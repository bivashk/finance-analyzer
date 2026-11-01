"""Conservative matched-merchant holdout, independent of the previous row-group benchmark."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from ml.prepare import ROOT


def compact(text):
    return re.sub('[^a-z0-9]','',text.casefold())


def vocabulary(source):
    block = source.split('const merchants = {',1)[1].split('\n};',1)[0]
    names = sorted({compact(json.loads(value)) for value in re.findall(r'"(?:\\.|[^"\\])*"',block)
                    if len(compact(json.loads(value)))>=4})
    # Unify overlapping brand aliases conservatively. Unknown/ambiguous matches are excluded.
    parent = {name:name for name in names}
    def find(name):
        while parent[name]!=name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    for i,left in enumerate(names):
        for right in names[i+1:]:
            if left in right or right in left:
                a,b = find(left),find(right)
                parent[max(a,b)] = min(a,b)
    return {name:find(name) for name in names}


def match_merchant(text, aliases):
    value = compact(text)
    matches = {family for name,family in aliases.items() if name in value}
    return next(iter(matches)) if len(matches)==1 else None


def main():
    dest = ROOT/'data/public'
    generator = dest/'merchant_generator.js'
    aliases = vocabulary(generator.read_text())
    (ROOT/'reports/merchant_vocabulary.json').write_text(json.dumps({
        'source':'https://github.com/wnstnb/foliome/blob/main/scripts/generate-training-data.js',
        'source_sha256':hashlib.sha256(generator.read_bytes()).hexdigest(), 'aliases':aliases},indent=2)+'\n')
    with (dest/'transactions-synthetic.csv').open() as f:
        raw = list(csv.DictReader(f))
    candidates = defaultdict(list)
    omitted = 0
    for row in raw:
        family = match_merchant(row['description'],aliases)
        if family is None:
            omitted += 1
            continue
        bucket = int(hashlib.sha256(('42|'+family).encode()).hexdigest()[:8],16)%10
        split = 'test' if bucket<2 else 'validation' if bucket<4 else 'train'
        candidates[(split,row['category'])].append({**row,'group':family,'split':split})
    labels = sorted({r['category'] for r in raw})
    valid = [label for label in labels if all(len(candidates[(split,label)])>=20 for split in ('train','validation','test'))]
    rows = []
    for split,cap in [('train',100),('validation',50),('test',50)]:
        for label in valid:
            # Deterministic sample independent of CSV order. Prioritize broad merchant coverage.
            groups = defaultdict(list)
            for row in candidates[(split,label)]:
                groups[row['group']].append(row)
            for group in groups:
                groups[group].sort(key=lambda r:hashlib.sha256(r['description'].encode()).hexdigest())
            chosen = []
            for depth in range(max(map(len,groups.values()))):
                for group in sorted(groups):
                    if depth<len(groups[group]):
                        chosen.append(groups[group][depth])
                if len(chosen)>=cap:
                    break
            rows.extend(chosen[:cap])
    with (dest/'merchant_prepared.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['description','category','group','split'])
        writer.writeheader()
        writer.writerows(rows)
    sets = {split:{r['group'] for r in rows if r['split']==split} for split in ('train','validation','test')}
    assert not (sets['train']&sets['test'] or sets['train']&sets['validation'] or sets['test']&sets['validation'])
    original = json.loads((ROOT/'reports/public_dataset.json').read_text())
    metadata = {**original,'prepared_sha256':hashlib.sha256((dest/'merchant_prepared.csv').read_bytes()).hexdigest(),
        'split':'Seed 42 global merchant-family hash split; matched aliases only; no family overlap across train/validation/test.',
        'split_counts':dict(Counter(r['split'] for r in rows)),
        'class_counts':{s:dict(Counter(r['category'] for r in rows if r['split']==s)) for s in sets},
        'merchant_groups':{s:len(values) for s,values in sets.items()},
        'included_categories':valid,'excluded_categories':[label for label in labels if label not in valid],
        'unmatched_or_ambiguous_source_rows_excluded':omitted,
        'limits':'Up to 100 train/50 validation/50 test per category; only unambiguous vocabulary-matched families. Unknown aliases may be missed; no real-customer validation.'}
    (ROOT/'reports/merchant_dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ['split_counts','merchant_groups','included_categories','excluded_categories']},indent=2),flush=True)


if __name__=='__main__':
    main()
