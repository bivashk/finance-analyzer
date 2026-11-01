"""Download a revision-pinned MIT synthetic corpus and create disjoint text-group splits."""
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'DoDataThings/us-bank-transaction-categories-v2'


def text_group(text):
    text = re.sub(r'\[(?:debit|credit)\]', '', text.casefold())
    # Remove volatile reference/store numbers and mixed alphanumeric reference tokens.
    text = re.sub(r'\b\w*\d\w*\b', ' ', text)
    text = re.sub(r'[^a-z]+', ' ', text)
    return ' '.join(text.split())


def make_splits(rows, train_cap=100, validation_cap=30, test_cap=50, seed=42):
    grouped = defaultdict(list)
    for row in rows:
        description, label = row['description'].strip(), row['category'].strip()
        if description and label:
            grouped[text_group(description)].append({'description':description,'category':label})
    conflicts = sum(len({r['category'] for r in values})>1 for values in grouped.values())
    by_label = defaultdict(list)
    for group, values in grouped.items():
        if group and len({r['category'] for r in values})==1:
            by_label[values[0]['category']].append((group,values))
    result = []
    for label in sorted(by_label):
        groups = sorted(by_label[label])
        random.Random(seed).shuffle(groups)
        if len(groups)<10:
            raise ValueError(f'Too few independent normalized text groups for {label}')
        ntest = max(1,len(groups)//5)
        nval = max(1,len(groups)//5)
        partitions = {'test':groups[:ntest], 'validation':groups[ntest:ntest+nval], 'train':groups[ntest+nval:]}
        caps = {'train':train_cap,'validation':validation_cap,'test':test_cap}
        for split, partition in partitions.items():
            # Round-robin samples independent groups before taking extra variants.
            chosen = []
            for depth in range(max(len(v) for _,v in partition)):
                for group, values in partition:
                    if depth<len(values):
                        chosen.append({**values[depth],'group':hashlib.sha256(group.encode()).hexdigest()[:20],'split':split})
                if len(chosen)>=caps[split]:
                    break
            result.extend(chosen[:caps[split]])
    group_sets = {s:{r['group'] for r in result if r['split']==s} for s in ('train','validation','test')}
    assert not (group_sets['train'] & group_sets['validation'] or group_sets['train'] & group_sets['test'] or group_sets['validation'] & group_sets['test'])
    return result, conflicts


def main():
    dest = ROOT/'data/public'
    dest.mkdir(parents=True,exist_ok=True)
    with urlopen(f'https://huggingface.co/api/datasets/{DATASET}?blobs=true',timeout=30) as response:
        info = json.load(response)
    revision = info['sha']
    files = [f['rfilename'] for f in info['siblings'] if f['rfilename'].endswith('.csv')]
    if len(files)!=1:
        raise ValueError(f'Expected one dataset CSV, found {files}')
    sources = {}
    names = {f['rfilename'] for f in info['siblings']}
    if info.get('cardData',{}).get('license') != 'mit':
        raise ValueError('Dataset license metadata changed; review before download')
    for name in [files[0], 'README.md'] + (['LICENSE'] if 'LICENSE' in names else []):
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{name}'
        target = dest/Path(name).name
        remote = next(f for f in info['siblings'] if f['rfilename']==name)
        content = target.read_bytes() if target.exists() else b''
        def matches(blob):
            if remote.get('lfs'):
                return hashlib.sha256(blob).hexdigest()==remote['lfs']['sha256']
            git_blob = b'blob '+str(len(blob)).encode()+b'\0'+blob
            return hashlib.sha1(git_blob).hexdigest()==remote.get('blobId')
        if not matches(content):
            with urlopen(url,timeout=60) as response:
                chunks = []
                while block := response.read(65536):
                    chunks.append(block)
                content = b''.join(chunks)
            if not matches(content):
                raise ValueError(f'Checksum mismatch for {name}')
            target.write_bytes(content)
        sources[name] = hashlib.sha256(content).hexdigest()
    with (dest/Path(files[0]).name).open() as source:
        raw = list(csv.DictReader(source))
    rows, conflicts = make_splits(raw)
    target = dest/'prepared.csv'
    with target.open('w',newline='') as output:
        writer = csv.DictWriter(output,fieldnames=['description','category','group','split'])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {'dataset':DATASET,'source_url':f'https://huggingface.co/datasets/{DATASET}',
        'revision':revision,'license':'MIT','provenance':'Synthetic bank-format descriptions, not real customer transactions.',
        'source_rows':len(raw),'source_sha256':sources,'prepared_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'split':'Seed 42; normalized-description-group-disjoint train/validation/test. Merchant overlap may remain.',
        'split_counts':dict(Counter(r['split'] for r in rows)),
        'class_counts':{s:dict(Counter(r['category'] for r in rows if r['split']==s)) for s in ('train','validation','test')},
        'discarded_conflicting_groups':conflicts,
        'limits':'Up to 100 train, 30 validation, 50 test examples per category. No claim of minimal sufficient labels.'}
    (ROOT/'reports/public_dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__ == '__main__':
    main()
