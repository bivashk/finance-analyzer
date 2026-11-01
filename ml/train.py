"""Actual full-encoder fine-tuning plus frozen sentence embeddings and a TF-IDF baseline."""
import argparse
import copy
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]


def metrics(y, pred, labels):
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,average='macro')),
            'per_category':classification_report(y,pred,labels=labels,output_dict=True,zero_division=0),
            'labels':labels,'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist()}


def read_data(path=None):
    with (Path(path) if path else ROOT/'data/public/prepared.csv').open() as source:
        rows = list(csv.DictReader(source))
    parts = {s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        if {r['group'] for r in parts[left]} & {r['group'] for r in parts[right]}:
            raise ValueError('Evaluation leakage: groups overlap')
    return parts


def train(epochs=8, dataset=None, output=None, report_path=None, metadata=None):
    import torch
    from huggingface_hub import model_info
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    parts = read_data(dataset)
    text = {s:[r['description'] for r in rows] for s,rows in parts.items()}
    y = {s:[r['category'] for r in rows] for s,rows in parts.items()}
    labels = sorted(set(y['train']))
    ids = {label:i for i,label in enumerate(labels)}
    provenance = json.loads((Path(metadata) if metadata else ROOT/'reports/public_dataset.json').read_text())
    report = {'dataset':provenance,'seed':42,'training_labels':len(y['train']),
              'test_examples':len(y['test']), 'results':{}, 'epochs':[]}
    baseline = make_pipeline(TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True),LogisticRegression(C=4,max_iter=1000,random_state=42))
    baseline.fit(text['train'],y['train'])
    report['results']['tfidf_logistic_regression'] = metrics(y['test'],baseline.predict(text['test']),labels)
    embedding_id = 'sentence-transformers/all-MiniLM-L6-v2'
    previous = ROOT/'reports/transfer_learning_evaluation.json'
    pinned = json.loads(previous.read_text())['results'] if previous.exists() else {}
    embedding_revision = pinned.get('sentence_transformers',{}).get('revision') or model_info(embedding_id).sha
    encoder = SentenceTransformer(embedding_id,revision=embedding_revision,device='cpu',trust_remote_code=False)
    embedding_train = encoder.encode(text['train'],batch_size=64,normalize_embeddings=True,show_progress_bar=False)
    embedding_model = LogisticRegression(C=4,max_iter=1000,random_state=42).fit(embedding_train,y['train'])
    before = time.perf_counter()
    embedding_test = encoder.encode(text['test'],batch_size=64,normalize_embeddings=True,show_progress_bar=False)
    embedding_pred = embedding_model.predict(embedding_test)
    report['results']['sentence_transformers'] = {**metrics(y['test'],embedding_pred,labels),
        'milliseconds_per_transaction':(time.perf_counter()-before)*1000/len(embedding_pred),
        'encoder':embedding_id,'revision':embedding_revision,'encoder_fine_tuned':False,
        'latency_scope':'CPU embedding encoding and logistic prediction; tokenizer included'}
    dest = (Path(output)/'sentence') if output else ROOT/'models/sentence'
    dest.mkdir(parents=True,exist_ok=True)
    # Portable JSON head avoids loading pickle files during serving.
    (dest/'head.json').write_text(json.dumps({'classes':embedding_model.classes_.tolist(),
        'coef':embedding_model.coef_.tolist(),'intercept':embedding_model.intercept_.tolist(),
        'encoder':embedding_id,'revision':embedding_revision,'provenance':provenance}))
    del encoder
    model_id = 'prajjwal1/bert-tiny'
    revision = pinned.get('fine_tuned_transformer',{}).get('revision') or model_info(model_id).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,revision=revision,
        num_labels=len(labels),id2label=dict(enumerate(labels)),label2id=ids,trust_remote_code=False)
    initial_hash = hashlib.sha256(model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy().tobytes()).hexdigest()
    encoded = {s:tokenizer(values,padding='max_length',truncation=True,max_length=64,return_tensors='pt') for s,values in text.items()}
    targets = {s:torch.tensor([ids[v] for v in values]) for s,values in y.items()}
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    best, best_state = -1,None
    training_start = time.perf_counter()

    def infer(split):
        model.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0,len(text[split]),64):
                batch = {k:v[start:start+64] for k,v in encoded[split].items()}
                chunks.extend(model(**batch).logits.argmax(-1).tolist())
        return [labels[i] for i in chunks]

    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        order = torch.randperm(len(y['train']))
        for start in range(0,len(order),32):
            batch_ids = order[start:start+32]
            batch = {k:v[batch_ids] for k,v in encoded['train'].items()}
            optimizer.zero_grad()
            loss = model(**batch,labels=targets['train'][batch_ids]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            loss_sum += loss.item()*len(batch_ids)
        val = f1_score(y['validation'],infer('validation'),average='macro')
        entry = {'epoch':epoch+1,'train_loss':loss_sum/len(order),'validation_macro_f1':float(val)}
        report['epochs'].append(entry)
        print(json.dumps(entry),flush=True)
        if val>best:
            best,best_state = val,copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    before = time.perf_counter()
    predicted = infer('test')
    latency = (time.perf_counter()-before)*1000/len(predicted)
    final_hash = hashlib.sha256(model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy().tobytes()).hexdigest()
    if initial_hash==final_hash:
        raise RuntimeError('Encoder weights did not change; fine-tuning claim would be invalid')
    report['results']['fine_tuned_transformer'] = {**metrics(y['test'],predicted,labels),
        'model':model_id,'revision':revision,'milliseconds_per_transaction':latency,
        'latency_scope':'CPU forward inference on pretokenized batches; tokenizer excluded',
        'encoder_weights_changed':True,'initial_embedding_sha256':initial_hash,'final_embedding_sha256':final_hash,
        'best_validation_macro_f1':best,'training_seconds':time.perf_counter()-training_start}
    report['limitations'] = ['Synthetic corpus; no real customer accuracy claim.',
        'Text normalization groups disjoint, but merchants and templates can overlap.',
        'Hyperparameters fixed before test evaluation; best epoch selected on validation macro-F1.',
        'BERT-tiny is a compact pretrained encoder, not a generative LLM.',
        'No human insight-discovery study has been conducted.']
    dest = (Path(output)/'transformer') if output else ROOT/'models/transformer'
    dest.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(dest,safe_serialization=True)
    tokenizer.save_pretrained(dest)
    (dest/'provenance.json').write_text(json.dumps(provenance,indent=2))
    (Path(report_path) if report_path else ROOT/'reports/transfer_learning_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{'accuracy':v['accuracy'],'macro_f1':v['macro_f1']} for k,v in report['results'].items()},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--dataset')
    parser.add_argument('--output')
    parser.add_argument('--report-path')
    parser.add_argument('--metadata')
    args = parser.parse_args()
    if not 1<=args.epochs<=30:
        parser.error('epochs must be between 1 and 30')
    train(args.epochs,args.dataset,args.output,args.report_path,args.metadata)
