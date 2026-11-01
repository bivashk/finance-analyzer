"""Session-scoped operational evidence; no descriptions, amounts, or customer IDs."""
import os
import sqlite3
import time
import uuid
import numpy as np


def connect(path):
    db = sqlite3.connect(path,timeout=20)
    db.row_factory = sqlite3.Row
    db.executescript('''CREATE TABLE IF NOT EXISTS ops_batches(
      id TEXT PRIMARY KEY,session TEXT NOT NULL,created REAL NOT NULL,elapsed_ms REAL NOT NULL,
      n INTEGER NOT NULL,estimated_cost REAL,model TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS ops_predictions(
      id TEXT PRIMARY KEY,batch TEXT NOT NULL,session TEXT NOT NULL,created REAL NOT NULL,
      transaction_id TEXT NOT NULL,category TEXT NOT NULL,confidence REAL NOT NULL,
      needs_review INTEGER NOT NULL,reviewed_category TEXT);
    CREATE INDEX IF NOT EXISTS ops_session ON ops_predictions(session,created);''')
    with db:
        cutoff = time.time()-86400
        db.execute('DELETE FROM ops_batches WHERE created<?',(cutoff,))
        db.execute('DELETE FROM ops_predictions WHERE created<?',(cutoff,))
    return db


def record(db,session,rows,predictions,elapsed_ms,model):
    now,batch = time.time(),uuid.uuid4().hex
    rate = os.environ.get('FINANCE_COST_PER_COMPUTE_SECOND')
    rate = float(rate) if rate else None
    if rate is not None and (not np.isfinite(rate) or rate<0):
        raise ValueError('Invalid configured compute cost')
    cost = rate*elapsed_ms/1000 if rate is not None else None
    with db:
        db.execute('INSERT INTO ops_batches VALUES (?,?,?,?,?,?,?)',(batch,session,now,elapsed_ms,len(rows),cost,model))
        for row,prediction in zip(rows,predictions):
            identity = uuid.uuid4().hex
            prediction['prediction_id'] = identity
            db.execute('INSERT INTO ops_predictions VALUES (?,?,?,?,?,?,?,?,NULL)',
                (identity,batch,session,now,row['id'],prediction['category'],prediction['confidence'],int(prediction['needs_review'])))


def correction(db,session,transaction_id,category,prediction_id=None):
    if prediction_id:
        row = db.execute('SELECT id FROM ops_predictions WHERE id=? AND session=? AND transaction_id=?',
                         (prediction_id,session,transaction_id)).fetchone()
    else:
        row = db.execute('SELECT id FROM ops_predictions WHERE session=? AND transaction_id=? ORDER BY created DESC LIMIT 1',
                         (session,transaction_id)).fetchone()
    if row:
        with db:
            db.execute('UPDATE ops_predictions SET reviewed_category=? WHERE id=?',(category,row['id']))
    return row is not None


def summary(db,session):
    rows = list(db.execute('SELECT * FROM ops_predictions WHERE session=?',(session,)))
    batches = list(db.execute('SELECT * FROM ops_batches WHERE session=?',(session,)))
    total = len(rows)
    reviewed = [r for r in rows if r['reviewed_category'] is not None]
    changes = sum(r['category']!=r['reviewed_category'] for r in reviewed)
    review = sum(r['needs_review'] for r in rows)
    per_item = [r['elapsed_ms']/r['n'] for r in batches if r['n']]
    cost_known = bool(batches) and all(r['estimated_cost'] is not None for r in batches)
    return {'prediction_events':total,'batches':len(batches),'auto_accept_coverage':(total-review)/total if total else None,
        'manual_review_rate':review/total if total else None,'reviewed_predictions':len(reviewed),
        'correction_rate_among_reviewed':changes/len(reviewed) if reviewed else None,
        'reviewed_prediction_agreement':1-changes/len(reviewed) if reviewed else None,
        'batch_amortized_inference_ms_p50':float(np.percentile(per_item,50)) if per_item else None,
        'batch_amortized_inference_ms_p95':float(np.percentile(per_item,95)) if per_item else None,
        'estimated_compute_cost_per_transaction':sum(r['estimated_cost'] for r in batches)/total if cost_known and total else None,
        'cost_status':'Configured compute-time estimate; not measured cloud billing' if cost_known else 'Not configured; no cost claim available',
        'retention_hours':24,'limitations':['Rates count prediction events, including repeated categorization.',
            'Manual reviews are a selected subset; agreement is not unbiased model accuracy.',
            'Latency covers model inference and tokenization, not full network/user waiting time.']}
