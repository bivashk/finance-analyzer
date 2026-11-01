"""Evaluate paired, consented usability observations; never manufacture improvement metrics."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def evaluate(rows):
    people = defaultdict(dict)
    for row in rows:
        person = row['participant_id'].strip()
        condition = row['condition']
        if not person or condition not in ('baseline','enhanced') or condition in people[person]:
            raise ValueError('One baseline and one enhanced observation required per anonymous participant')
        total, correct = int(row['tasks_total']), int(row['correct_insights'])
        seconds = float(row['seconds'])
        if total<1 or not 0<=correct<=total or not np.isfinite(seconds) or seconds<=0:
            raise ValueError('Invalid task counts or elapsed time')
        people[person][condition] = {'total':total,'correct':correct,'seconds':seconds}
    if len(people)<10:
        raise ValueError('At least 10 complete paired participants are required by the study protocol')
    pairs = []
    for person,conditions in people.items():
        if set(conditions)!= {'baseline','enhanced'}:
            raise ValueError(f'Missing paired condition for {person}')
        baseline, enhanced = conditions['baseline'],conditions['enhanced']
        if baseline['total']!=enhanced['total']:
            raise ValueError('Matched conditions must have equal task counts')
        pairs.append((baseline,enhanced))
    rates = np.array([[b['correct']/b['total'],e['correct']/e['total']] for b,e in pairs])
    times = np.array([[b['seconds'],e['seconds']] for b,e in pairs])
    baseline, enhanced = rates.mean(axis=0)
    if baseline==0:
        relative = None
    else:
        relative = float((enhanced-baseline)/baseline*100)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(5000):
        sample = rates[rng.integers(0,len(rates),len(rates))].mean(axis=0)
        boot.append(float((sample[1]-sample[0])*100))
    return {'participants':len(pairs),'baseline_correct_insight_rate':float(baseline),
            'enhanced_correct_insight_rate':float(enhanced),'relative_improvement_percent':relative,
            'absolute_improvement_percentage_points':float((enhanced-baseline)*100),
            'paired_bootstrap_95_percent_ci_percentage_points':np.percentile(boot,[2.5,97.5]).tolist(),
            'mean_task_time_reduction_percent':float((times[:,0].mean()-times[:,1].mean())/times[:,0].mean()*100),
            'limitations':['Observational study outcomes depend on recruitment, task design, and ordering.',
                           'Report actual outcomes and uncertainty; this does not prove a predefined 40% uplift.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    args = parser.parse_args()
    with Path(args.csv).open() as source:
        report = evaluate(list(csv.DictReader(source)))
    Path('reports/usability_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
