"""TF-IDF classifier with a fixed, merchant-disjoint benchmark; no pickle loading."""
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
RULES = {
 'Income': 'salary|payroll|dividend|bonus', 'Transport': 'uber|ola|cab|metro|bus|train',
 'Food & Drink': 'restaurant|cafe|pizza|food|swiggy|zomato', 'Groceries': 'grocery|supermarket|bigbasket|milk',
 'Utilities': 'electricity|water|broadband|internet|utility', 'Shopping': 'shopping|clothing|myntra',
 'Entertainment': 'netflix|spotify|cinema|movie', 'Health': 'hospital|clinic|pharmacy|medicine',
 'Fuel': 'petrol|diesel|fuel', 'Rent': 'rent|landlord', 'Fitness': 'gym|yoga|fitness',
 'Education': 'tuition|school|course', 'Fees': 'fee|charge', 'Travel': 'hotel|resort',
 'Electronics': 'laptop|electronics|computer'}


def rule_predict(text):
    return next((c for c, pattern in RULES.items() if re.search(pattern, text, re.I)), 'Other')


class Classifier:
    def __init__(self, path=ROOT/'data/transactions_synthetic.csv'):
        with Path(path).open() as source:
            self.rows = list(csv.DictReader(source))
        self.version = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
        self.pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                      LogisticRegression(max_iter=1000, C=4, random_state=42))
        self.pipeline.fit([r['description'] for r in self.rows], [r['category'] for r in self.rows])

    def predict(self, descriptions):
        probabilities = self.pipeline.predict_proba(descriptions)
        classes = self.pipeline.classes_
        return [{'category': str(classes[p.argmax()]), 'confidence': round(float(p.max()),4),
                 'needs_review': bool(p.max() < .65), 'model_version': self.version,
                 'training_data': 'synthetic demonstration data'} for p in probabilities]

    def evaluate(self):
        # The final two merchants in each category are never seen during fitting.
        test = [r for r in self.rows if r['split'] == 'test']
        train = [r for r in self.rows if r['split'] == 'train']
        assert not ({r['merchant'] for r in train} & {r['merchant'] for r in test})
        model = make_pipeline(TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True),
                              LogisticRegression(max_iter=1000, C=4, random_state=42))
        model.fit([r['description'] for r in train], [r['category'] for r in train])
        x, y = [r['description'] for r in test], [r['category'] for r in test]
        labels = sorted(set(y))
        results = {}
        for name, predict in [('keyword_baseline', lambda x: [rule_predict(t) for t in x]),
                              ('tfidf_logistic_regression', model.predict)]:
            start = time.perf_counter()
            pred = predict(x)
            elapsed = (time.perf_counter()-start)*1000/len(x)
            results[name] = {'macro_f1': f1_score(y,pred,average='macro'),
                'milliseconds_per_transaction': elapsed,
                'per_category': classification_report(y,pred,output_dict=True,zero_division=0),
                'confusion_matrix': confusion_matrix(y,pred,labels=labels).tolist()}
        return {'dataset': 'Synthetic template-based demonstration; not evidence of real banking accuracy.',
                'seed':42, 'train_rows':len(train), 'test_rows':len(test), 'labels':labels,
                'split':'Fixed merchant-disjoint holdout; templates overlap, so this is an optimistic demo.',
                'model_version':self.version, 'results':results,
                'eda':{'rows':len(self.rows), 'merchant_groups':len({r['merchant'] for r in self.rows}),
                       'class_counts':{label:sum(r['category']==label for r in self.rows) for label in labels},
                       'missing_descriptions':sum(not r['description'].strip() for r in self.rows),
                       'features':'Word TF-IDF unigrams and bigrams; merchant identity is not an explicit feature.'},
                'limitations':['No real customer data or business-impact measurement.',
                               'Confidence scores are uncalibrated; 0.65 is a demonstration review threshold.',
                               'No paid LLM comparison has been run.']}


if __name__ == '__main__':
    report = Classifier().evaluate()
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/model_evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v['macro_f1'] for k,v in report['results'].items()},indent=2))
