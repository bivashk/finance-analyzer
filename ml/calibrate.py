"""Calibrate on validation data only; report untouched-test reliability and error examples."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from backend.calibration import fit_temperature, calibration_metrics, choose_review_threshold
from ml.prepare import ROOT


def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(4)
    path = ROOT/'models/merchant/transformer'
    tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).eval()
    with (ROOT/'data/public/merchant_prepared.csv').open() as f:
        rows = list(csv.DictReader(f))
    def predict(subset):
        logits = []
        with torch.inference_mode():
            for start in range(0,len(subset),64):
                batch = tokenizer([r['description'] for r in subset[start:start+64]],padding=True,truncation=True,max_length=64,return_tensors='pt')
                logits.extend(model(**batch).logits.tolist())
        return np.asarray(logits)
    validation = [r for r in rows if r['split']=='validation']
    # Disjoint merchant groups for fitting temperature and selecting the review policy.
    fit = [r for r in validation if int(hashlib.sha256(r['group'].encode()).hexdigest()[:8],16)%2==0]
    policy_rows = [r for r in validation if r not in fit]
    labels = model.config.label2id
    logits = predict(fit)
    temperature = fit_temperature(logits,[labels[r['category']] for r in fit])
    policy = choose_review_threshold(predict(policy_rows),[labels[r['category']] for r in policy_rows],temperature)
    test = [r for r in rows if r['split']=='test']
    test_logits = predict(test)
    y = np.array([labels[r['category']] for r in test])
    probabilities = softmax(test_logits/temperature,axis=1)
    predicted = probabilities.argmax(axis=1)
    accepted = probabilities.max(axis=1)>=policy['threshold']
    errors = [{'description':row['description'],'merchant_family':row['group'],'actual':row['category'],
               'predicted':model.config.id2label[int(index)],'confidence':round(float(probability.max()),4)}
              for row,index,probability in zip(test,predicted,probabilities) if labels[row['category']]!=index]
    confusion = {}
    for error in errors:
        key = error['actual']+' -> '+error['predicted']
        confusion[key] = confusion.get(key,0)+1
    report = {'calibration_fit_rows':len(fit),'threshold_selection_rows':len(policy_rows),'test_rows':len(test),
              'temperature':temperature,'review_policy':policy,
              'uncalibrated_test':calibration_metrics(test_logits,y),
              'calibrated_test':calibration_metrics(test_logits,y,temperature),
              'test_auto_accept_coverage':float(accepted.mean()),
              'test_accepted_accuracy':float((predicted[accepted]==y[accepted]).mean()) if accepted.any() else None,
              'test_review_rate':float(1-accepted.mean()),'test_error_count':len(errors),
              'top_confusions':sorted(confusion.items(),key=lambda item:-item[1])[:10],
              'error_examples':errors[:30],
              'limitations':['Synthetic matched-merchant benchmark; vocabulary matching can miss aliases.',
                             'Validation used for epoch selection as well as post-hoc calibration; independent calibration data would be stronger.',
                             'Threshold target is empirical; reliability and accepted accuracy can deteriorate on new domains.']}
    configuration = {'temperature':temperature,**policy,
                     'calibration_source':'merchant-disjoint synthetic validation data',
                     'checkpoint_sha256':hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest()}
    (path/'calibration.json').write_text(json.dumps(configuration,indent=2)+'\n')
    (ROOT/'reports/calibration_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('error_examples','uncalibrated_test','calibrated_test')},indent=2),flush=True)


if __name__=='__main__':
    main()
