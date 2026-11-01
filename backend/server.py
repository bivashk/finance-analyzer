"""Single-user, loopback-only server. Serves only public frontend assets."""
import json
import os
import hashlib
import hmac
import re
import time
import threading
from collections import defaultdict,deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from backend.analytics import CATEGORIES, connect, ingest, parse_csv, snapshot, transactions
from backend.model import Classifier
from backend.grounding import explain
from backend.forecast import forecast
from backend.anomalies import detect
from backend.inference import load_classifier
from backend.customer import profile,segment
from backend import operations

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = os.environ.get('FINANCE_DB', str(ROOT/'data/finance.sqlite3'))
MODEL = None
CLASSIFIER_NAME = os.environ.get('FINANCE_CLASSIFIER','tfidf')
PUBLIC_DEMO = os.environ.get('FINANCE_PUBLIC_DEMO')=='1'
METRICS_PATH = os.environ.get('FINANCE_METRICS_DB',str(ROOT/'data/operations.sqlite3'))
RATE_LOCK = threading.Lock()
RATE_WINDOWS = defaultdict(deque)
ASSETS = {'/':('index.html','text/html'), '/index.html':('index.html','text/html'),
          '/script.js':('script.js','text/javascript'), '/styles.css':('styles.css','text/css'),
          '/logo.png':('logo.png','image/png')}


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(body)

    def session(self):
        value = self.headers.get('X-Finance-Session','local')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',value):
            raise ValueError('Invalid session identifier')
        return hashlib.sha256(value.encode()).hexdigest()

    def authorize(self):
        token = os.environ.get('FINANCE_PROXY_TOKEN')
        if token and not hmac.compare_digest(self.headers.get('X-Finance-Proxy',''),token):
            self.reply(401,{'error':'Access the API through the configured frontend'})
            return False
        if PUBLIC_DEMO:
            if not re.fullmatch(r'[a-zA-Z0-9_-]{32,100}',self.headers.get('X-Finance-Session','')):
                self.reply(400,{'error':'A browser session is required'})
                return False
            with RATE_LOCK:
                now = time.monotonic()
                # Key by socket address, never an untrusted forwarded header.
                key = self.client_address[0]
                window = RATE_WINDOWS[key]
                while window and window[0]<now-60:
                    window.popleft()
                if len(window)>=120:
                    self.reply(429,{'error':'Please wait before sending more requests'})
                    return False
                window.append(now)
                if len(RATE_WINDOWS)>10000:
                    for old_key in list(RATE_WINDOWS):
                        if not RATE_WINDOWS[old_key] or RATE_WINDOWS[old_key][-1]<now-60:
                            del RATE_WINDOWS[old_key]
        return True

    def do_GET(self):
        if self.path == '/api/health':
            return self.reply(200, {'status':'ok','mode':'stateless public demo' if PUBLIC_DEMO else 'local single-user','model_data':'synthetic', 'classifier':CLASSIFIER_NAME})
        if self.path in ('/api/metrics','/api/customer-demo'):
            if not self.authorize():
                return
            if self.path=='/api/customer-demo':
                from ml.customer_demo import customers
                return self.reply(200,{'customers':customers(),'provenance':'Synthetic demonstration only'})
            try:
                metrics_db = operations.connect(METRICS_PATH)
                try:
                    return self.reply(200,operations.summary(metrics_db,self.session()))
                finally:
                    metrics_db.close()
            except ValueError as exc:
                return self.reply(400,{'error':str(exc)})
        if self.path == '/api/evaluation':
            path = ROOT/'reports'/('merchant_evaluation.json' if os.environ.get('FINANCE_MODEL_PATH') else 'transfer_learning_evaluation.json' if CLASSIFIER_NAME!='tfidf' else 'model_evaluation.json')
            return self.reply(200, json.loads(path.read_text())) if path.exists() else self.reply(404, {'error':'Run python -m backend.model first'})
        if self.path in ASSETS:
            name, content_type = ASSETS[self.path]
            self.send_response(200)
            self.send_header('Content-Type',content_type)
            self.end_headers()
            self.wfile.write((ROOT/name).read_bytes())
            return
        self.reply(404, {'error':'Not found'})

    def do_POST(self):
        if not self.authorize():
            return
        # Reject cross-origin writes to the local service, including opaque file origins.
        allowed_origin = os.environ.get('FINANCE_ALLOWED_ORIGIN')
        if self.headers.get('Origin') not in (None, f'http://{self.headers.get("Host")}', f'https://{self.headers.get("Host")}', allowed_origin):
            return self.reply(403, {'error':'Use the dashboard served by this service'})
        if self.headers.get('Content-Type','').split(';')[0] != 'application/json':
            return self.reply(415, {'error':'Expected application/json'})
        db = None
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 5_000_000:
                return self.reply(413, {'error':'Request must contain 1 byte to 5 MB'})
            data = json.loads(self.rfile.read(size))
            if not isinstance(data,dict):
                raise ValueError('Expected JSON object')
            db = connect(':memory:' if PUBLIC_DEMO else DB_PATH)
            if self.path=='/api/cohort-import':
                rows = parse_csv(data['csv'])
                if not rows or len(rows)>10000 or any(not row.get('customer_id') for row in rows):
                    raise ValueError('CSV requires Customer_ID and 1-10000 transaction rows')
                grouped = defaultdict(list)
                for row in rows:
                    grouped[row['customer_id']].append(row)
                return self.reply(200,segment([{'customer_id':identity,'transactions':values} for identity,values in grouped.items()]))
            if self.path=='/api/segments':
                return self.reply(200,segment(data.get('customers')))
            if self.path == '/api/import':
                rows = parse_csv(data['csv'])
                result = ingest(db, rows)
                from backend.analytics import normalize
                imported_ids = set()
                for row in rows:
                    try:
                        imported_ids.add(normalize(row)['id'])
                    except (ValueError, TypeError, OverflowError):
                        pass
                return self.reply(200, {**result,'transactions':[t for t in transactions(db) if t['id'] in imported_ids]})
            if self.path in ('/api/analyze', '/api/chat'):
                rows = data.get('transactions')
                if not isinstance(rows,list) or len(rows)>10000 or any(not isinstance(r,dict) for r in rows):
                    raise ValueError('Expected at most 10000 transaction objects')
                # Analyze exactly the current browser selection; never mix it with persisted imports.
                memory = connect(':memory:')
                try:
                    quality = ingest(memory,rows)
                    result = snapshot(memory)
                    result['forecast'] = forecast(result['monthly'])
                    result['anomalies'] = detect(transactions(memory))
                    result['customer_profile'] = profile(transactions(memory)) if result['count'] else None
                finally:
                    memory.close()
                if self.path == '/api/chat':
                    return self.reply(200, explain(result,data.get('message','')))
                return self.reply(200, {**result,'quality':quality})
            if self.path == '/api/classify':
                rows = data.get('transactions')
                if not isinstance(rows,list) or not rows or len(rows)>10000 or any(not isinstance(r,dict) for r in rows):
                    raise ValueError('Expected 1-10000 transaction objects')
                if PUBLIC_DEMO and len(rows)>256:
                    raise ValueError('Public inference is limited to 256 transactions per batch')
                # Validate before prediction. Identities match the ingestion policy.
                from backend.analytics import normalize
                normalized = [normalize(r) for r in rows]
                descriptions = [r['description'] for r in normalized]
                if CLASSIFIER_NAME in ('transformer','sentence'):
                    descriptions = [description if re.match(r'^\[(debit|credit)\]',description,re.I)
                                    else ('[credit] ' if row['amount_paise']>0 else '[debit] ')+description
                                    for row,description in zip(normalized,descriptions)]
                started = time.perf_counter()
                predictions = MODEL.predict(descriptions)
                elapsed = (time.perf_counter()-started)*1000
                metrics_db = operations.connect(METRICS_PATH)
                try:
                    operations.record(metrics_db,self.session(),normalized,predictions,elapsed,CLASSIFIER_NAME)
                finally:
                    metrics_db.close()
                with db:
                    for row, pred in zip(normalized,predictions):
                        db.execute('INSERT INTO predictions(transaction_id,category,confidence,model_version) VALUES (?,?,?,?)',
                                   (row['id'],pred['category'],pred['confidence'],pred['model_version']))
                return self.reply(200, {'predictions':predictions})
            if self.path == '/api/corrections':
                from backend.analytics import normalize
                row = normalize(data['transaction'])
                category = data['category']
                if category not in CATEGORIES:
                    raise ValueError('Unknown category')
                with db:
                    db.execute('INSERT INTO corrections(transaction_id,old_category,new_category) VALUES (?,?,?)',
                               (row['id'],row['category'],category))
                    db.execute('UPDATE transactions SET category=? WHERE id=?',(category,row['id']))
                metrics_db = operations.connect(METRICS_PATH)
                try:
                    operations.correction(metrics_db,self.session(),row['id'],category,
                                          (data['transaction'].get('prediction') or {}).get('prediction_id'))
                finally:
                    metrics_db.close()
                return self.reply(200, {'saved':True})
            self.reply(404, {'error':'Not found'})
        except (ValueError, KeyError, TypeError) as exc:
            self.reply(400, {'error':str(exc)})
        except Exception:
            self.reply(500, {'error':'Internal error; check local service configuration'})
        finally:
            if db is not None:
                db.close()


if __name__ == '__main__':
    MODEL = load_classifier(CLASSIFIER_NAME)
    server = ThreadingHTTPServer((os.environ.get('FINANCE_HOST','127.0.0.1'),int(os.environ.get('PORT','8000'))),Handler)
    print(f'Finance Analyzer: http://127.0.0.1:{server.server_port} ({CLASSIFIER_NAME}, single-user local demo)')
    server.serve_forever()
