"""Load local fine-tuned weights or a pinned embedding model; never run remote code."""
import json
import os
import hashlib
from pathlib import Path
import numpy as np
from backend.model import Classifier, ROOT



class TransformerClassifier:
    def __init__(self, path=ROOT/'models/transformer'):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).eval()
        self.provenance = json.loads((Path(path)/'provenance.json').read_text())
        self.version = 'bert-tiny-finetuned-'+self.provenance['prepared_sha256'][:12]
        calibration = Path(path)/'calibration.json'
        self.calibration = json.loads(calibration.read_text()) if calibration.exists() else None
        if self.calibration and self.calibration['checkpoint_sha256']!=hashlib.sha256((Path(path)/'model.safetensors').read_bytes()).hexdigest():
            raise ValueError('Calibration does not match this checkpoint')

    def predict(self, descriptions):
        result = []
        for offset in range(0,len(descriptions),64):
            batch = self.tokenizer(descriptions[offset:offset+64],padding=True,truncation=True,max_length=64,return_tensors='pt')
            with self.torch.inference_mode():
                probabilities = (self.model(**batch).logits/(self.calibration['temperature'] if self.calibration else 1)).softmax(-1)
            for probability in probabilities:
                confidence,index = probability.max(-1)
                result.append({'category':self.model.config.id2label[int(index)],
                    'confidence':round(float(confidence),4),'needs_review':float(confidence)<(self.calibration['threshold'] if self.calibration else .65),
                    'calibrated':bool(self.calibration),
                    'model_version':self.version,'training_data':self.provenance['provenance']})
        return result


class SentenceClassifier:
    def __init__(self,path=ROOT/'models/sentence/head.json'):
        from sentence_transformers import SentenceTransformer
        head = json.loads(Path(path).read_text())
        self.encoder = SentenceTransformer(head['encoder'],revision=head['revision'],device='cpu',local_files_only=True)
        self.head = head

    def predict(self,descriptions):
        embeddings = self.encoder.encode(descriptions,batch_size=64,normalize_embeddings=True,show_progress_bar=False)
        logits = embeddings @ np.array(self.head['coef']).T + np.array(self.head['intercept'])
        logits -= logits.max(axis=1,keepdims=True)
        probabilities = np.exp(logits)/np.exp(logits).sum(axis=1,keepdims=True)
        return [{'category':self.head['classes'][int(p.argmax())], 'confidence':round(float(p.max()),4),
                 'needs_review':bool(p.max()<.65),'model_version':'minilm-frozen-logistic',
                 'training_data':self.head['provenance']['provenance']} for p in probabilities]


def load_classifier(name):
    if name=='transformer':
        return TransformerClassifier(os.environ.get('FINANCE_MODEL_PATH',str(ROOT/'models/transformer')))
    if name=='sentence':
        return SentenceClassifier()
    if name=='tfidf':
        return Classifier()
    raise ValueError('FINANCE_CLASSIFIER must be transformer, sentence, or tfidf')
