"""Behavioral customer intelligence. No protected attributes or eligibility decisions."""
from collections import defaultdict
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,adjusted_rand_score
from backend.analytics import connect,ingest,transactions

FEATURES = ['transactions_per_month','average_monthly_expenses','savings_ratio',
            'monthly_expense_variability','discretionary_share','recurring_expense_share']
DISCRETIONARY = {'Restaurants','Food & Drink','Entertainment','Shopping','Travel','Subscription'}


def profile(rows):
    db = connect(':memory:')
    try:
        quality = ingest(db,rows)
        normalized = transactions(db)
        monthly = [dict(r) for r in db.execute('''SELECT substr(date,1,7) month, count(*) n,
          SUM(CASE WHEN amount_paise>0 THEN amount_paise ELSE 0 END)/100.0 income,
          SUM(CASE WHEN amount_paise<0 THEN -amount_paise ELSE 0 END)/100.0 expenses
          FROM transactions GROUP BY 1 ORDER BY 1''')]
    finally:
        db.close()
    if not monthly:
        raise ValueError('Customer needs at least one valid transaction')
    income = sum(m['income'] for m in monthly)
    expense = sum(m['expenses'] for m in monthly)
    values = np.array([m['expenses'] for m in monthly])
    discretionary = sum(-r['amount'] for r in normalized if r['amount']<0 and r['category'] in DISCRETIONARY)
    descriptions = defaultdict(list)
    for row in normalized:
        if row['amount']<0:
            descriptions[row['description'].casefold()].append(row)
    recurring = []
    for description,items in descriptions.items():
        amounts = np.array([-r['amount'] for r in items])
        months = {r['date'][:7] for r in items}
        if len(months)>=3 and amounts.std()/max(amounts.mean(),.01)<.1:
            recurring.extend(items)
    features = {'transactions_per_month':len(normalized)/len(monthly),
        'average_monthly_expenses':expense/len(monthly),
        'savings_ratio':(income-expense)/income if income else 0.,
        'monthly_expense_variability':float(values.std()/max(values.mean(),.01)),
        'discretionary_share':discretionary/expense if expense else 0.,
        'recurring_expense_share':sum(-r['amount'] for r in recurring)/expense if expense else 0.}
    if income and features['savings_ratio']<.1:
        rule,action = 'Low recorded surplus','Review the largest expense categories before setting a monthly budget.'
    elif features['monthly_expense_variability']>.35:
        rule,action = 'Variable spending','Review month-to-month changes and plan a buffer for irregular expenses.'
    elif features['discretionary_share']>.45:
        rule,action = 'Discretionary spending focus','Review optional purchases and subscriptions against your own goals.'
    elif income and features['savings_ratio']>=.3:
        rule,action = 'Consistent recorded surplus','Set a savings goal using the surplus observed in the recorded period.'
    else:
        rule,action = 'Balanced recorded spending','Monitor monthly category totals and review recurring payments.'
    return {'features':{k:round(float(v),4) for k,v in features.items()},'rule_segment':rule,
            'recommended_action':action,'observed_months':len(monthly),'transactions':len(normalized),
            'income_recorded':income>0,'quality':quality,
            'limitations':['Behavioral descriptions only; no credit, fraud, eligibility, or retention prediction.',
                           'Missing months/accounts and transfers counted as income can distort these features.',
                           'Recurring detection uses repeated descriptions and similar amounts across at least 3 months.']}


def segment(customers):
    if not isinstance(customers,list) or not 8<=len(customers)<=100:
        raise ValueError('Provide 8-100 customer records for cohort segmentation')
    if any(not isinstance(c,dict) or not isinstance(c.get('transactions'),list) for c in customers):
        raise ValueError('Each customer requires transactions')
    if sum(len(c['transactions']) for c in customers)>10000:
        raise ValueError('Cohort is limited to 10000 transactions')
    identities = [str(c.get('customer_id','')).strip() for c in customers]
    if any(not identity or len(identity)>80 for identity in identities) or len(set(identities))!=len(identities):
        raise ValueError('Use distinct customer IDs of 1-80 characters')
    profiles = [profile(c['transactions']) for c in customers]
    if any(p['observed_months']<3 or p['transactions']<10 for p in profiles):
        raise ValueError('Each customer needs at least 3 observed months and 10 valid transactions')
    x = np.array([[p['features'][f] for f in FEATURES] for p in profiles])
    # Log-scale monetary size and frequency before standardization, preserving ratio meaning.
    x[:,:2] = np.log1p(x[:,:2])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    if len(np.unique(scaled,axis=0))<3:
        raise ValueError('Insufficient behavioral variation to form meaningful clusters')
    candidates = []
    for k in range(2,min(5,len(customers)-1)+1):
        if k>=len(np.unique(scaled,axis=0)):
            break
        model = KMeans(n_clusters=k,n_init=20,random_state=42).fit(scaled)
        if len(set(model.labels_))<2:
            continue
        candidates.append((float(silhouette_score(scaled,model.labels_)),model))
    score,model = max(candidates,key=lambda item:item[0])
    agreement = []
    for seed in (7,99,2026):
        alternate = KMeans(n_clusters=model.n_clusters,n_init=20,random_state=seed).fit_predict(scaled)
        agreement.append(float(adjusted_rand_score(model.labels_,alternate)))
    members = [{'customer_id':identity,'cluster':int(cluster),**p}
               for identity,cluster,p in zip(identities,model.labels_,profiles)]
    summaries = []
    for cluster in range(model.n_clusters):
        group = [m for m in members if m['cluster']==cluster]
        summaries.append({'cluster':cluster,'customers':len(group),
            'mean_features':{f:round(float(np.mean([m['features'][f] for m in group])),4) for f in FEATURES},
            'common_rule_segment':max(sorted({m['rule_segment'] for m in group}),key=lambda label:sum(m['rule_segment']==label for m in group))})
    return {'customers':members,'clusters':summaries,'chosen_k':model.n_clusters,'silhouette_score':score,
            'mean_seed_stability_adjusted_rand':float(np.mean(agreement)),
            'rule_comparison_adjusted_rand':float(adjusted_rand_score([m['rule_segment'] for m in members],model.labels_)),
            'limitations':['KMeans is fitted on this submitted cohort; these are descriptive clusters, not validated predictive customer outcomes.',
                           'Silhouette measures geometric separation; seed stability does not establish business value.',
                           'No acquisition, engagement, retention uplift, or credit-risk claim is made.']}
