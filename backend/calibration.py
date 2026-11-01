"""Validation-only temperature fitting and review policy, with explicit uncertainty."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss


def fit_temperature(logits, labels):
    logits,labels = np.asarray(logits),np.asarray(labels)
    result = minimize_scalar(lambda log_t:log_loss(labels,softmax(logits/np.exp(log_t),axis=1),labels=np.arange(logits.shape[1])),
                             bounds=(-2,3),method='bounded')
    return float(np.exp(result.x))


def calibration_metrics(logits, labels, temperature=1):
    probabilities = softmax(np.asarray(logits)/temperature,axis=1)
    labels = np.asarray(labels)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1)==labels
    bins, ece = [],0.
    bucket = np.minimum((confidence*10).astype(int),9)
    for index in range(10):
        lower = index/10
        selected = bucket==index
        if selected.any():
            accuracy,average = float(correct[selected].mean()),float(confidence[selected].mean())
            ece += selected.mean()*abs(accuracy-average)
            bins.append({'lower':round(float(lower),1),'count':int(selected.sum()),'accuracy':accuracy,'confidence':average})
    onehot = np.eye(probabilities.shape[1])[labels]
    return {'negative_log_likelihood':float(log_loss(labels,probabilities,labels=np.arange(probabilities.shape[1]))),
            'brier_score':float(np.mean(np.sum((probabilities-onehot)**2,axis=1))),
            'expected_calibration_error_10_bins':float(ece),'reliability_bins':bins}


def choose_review_threshold(logits, labels, temperature, target_precision=.95, minimum=30):
    probabilities = softmax(np.asarray(logits)/temperature,axis=1)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1)==np.asarray(labels)
    for threshold in np.arange(.5,1,.01):
        accepted = confidence>=threshold
        if accepted.sum()>=minimum and correct[accepted].mean()>=target_precision:
            return {'threshold':round(float(threshold),2),'validation_accepted':int(accepted.sum()),
                    'validation_precision':float(correct[accepted].mean()),'target_precision':target_precision,
                    'policy':'Empirical validation target, not a guaranteed production error rate.'}
    return {'threshold':1.,'validation_accepted':0,'validation_precision':None,'target_precision':target_precision,
            'policy':'No validation threshold met the target with sufficient examples; review all predictions.'}
