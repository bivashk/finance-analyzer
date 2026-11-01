"""Descriptive spending patterns and unsupervised review flags; never fraud labels."""
from collections import defaultdict
from datetime import date
import math
import numpy as np
from sklearn.ensemble import IsolationForest


def detect(rows):
    expenses = [r for r in rows if r['amount']<0]
    if len(expenses)<20:
        return {'available':False,'reason':'At least 20 expense transactions are required.','flags':[],'patterns':[]}
    amounts = np.array([abs(r['amount']) for r in expenses],dtype=float)
    log_amounts = np.log1p(amounts)
    by_category = defaultdict(list)
    for i,row in enumerate(expenses):
        by_category[row['category']].append(i)
    robust = np.zeros(len(expenses))
    for positions in by_category.values():
        reference = log_amounts[positions] if len(positions)>=8 else log_amounts
        median = np.median(reference)
        mad = np.median(np.abs(reference-median))
        # Floor avoids division by zero when repeated payments dominate.
        robust[positions] = .6745*(log_amounts[positions]-median)/max(float(mad),.1)
    features = np.column_stack([log_amounts, [date.fromisoformat(r['date']).weekday() for r in expenses]])
    model = IsolationForest(n_estimators=150,contamination='auto',random_state=42).fit(features)
    scores = -model.score_samples(features)
    # Fixed score threshold, not a promise that a certain percentage is fraudulent.
    isolation = scores>.65
    flags = []
    for i,row in enumerate(expenses):
        if robust[i]>3.5 or isolation[i]:
            reasons = []
            if robust[i]>3.5:
                reasons.append('Amount unusually high relative to available expense history')
            if isolation[i]:
                reasons.append('Isolation Forest flags an unusual amount/weekday combination')
            flags.append({'id':row['id'],'date':row['date'],'description':row['description'],
                          'amount':row['amount'],'category':row['category'],
                          'robust_z':round(float(robust[i]),3),'isolation_score':round(float(scores[i]),3),
                          'reasons':reasons})
    flags.sort(key=lambda r:r['isolation_score'],reverse=True)
    patterns = [{'category':category,'transactions':len(indices),
                 'total_expense':round(float(amounts[indices].sum()),2),
                 'median_expense':round(float(np.median(amounts[indices])),2)}
                for category,indices in sorted(by_category.items())]
    return {'available':True,'flags':flags,'patterns':patterns,
            'limitations':['Flags are unsupervised review suggestions, not confirmed fraud or errors.',
                           'Model is fitted on the submitted batch; it is not a prospective risk score.']}
