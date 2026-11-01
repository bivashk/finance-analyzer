"""HTTP contract tests against an ephemeral loopback server and temporary database."""
import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
from backend import server
from backend.model import Classifier
from backend.analytics import connect


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        server.DB_PATH = str(Path(cls.temp.name)/'test.sqlite3')
        server.MODEL = Classifier()
        cls.old_metrics = server.METRICS_PATH
        server.METRICS_PATH = str(Path(cls.temp.name)/'metrics.sqlite3')
        cls.http = ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever,daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.http.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.temp.cleanup()
        server.METRICS_PATH = cls.old_metrics

    def post(self,path,body,origin=None):
        headers = {'Content-Type':'application/json'}
        if origin:
            headers['Origin'] = origin
        request = urllib.request.Request(self.url+path,data=json.dumps(body).encode(),headers=headers)
        with urllib.request.urlopen(request) as response:
            return json.load(response)

    def test_import_predict_correct_and_analyze_current_rows(self):
        result = self.post('/api/import',{'csv':'Date,Description,Amount\n2026-01-01,Salary payroll,1000\n2026-01-02,Cafe food,-20\n'})
        self.assertEqual(result['accepted'],2)
        rows = result['transactions']
        predicted = self.post('/api/classify',{'transactions':rows})
        self.assertEqual(len(predicted['predictions']),2)
        self.post('/api/corrections',{'transaction':rows[1],'category':'Food & Drink'})
        db = connect(server.DB_PATH)
        self.assertEqual(db.execute('SELECT count(*) FROM corrections').fetchone()[0],1)
        self.assertEqual(db.execute('SELECT count(*) FROM predictions').fetchone()[0],2)
        db.close()
        analysis = self.post('/api/analyze',{'transactions':[rows[1]]})
        self.assertEqual(analysis['count'],1)
        self.assertEqual(analysis['metrics']['net_cash_flow'],-20)
        self.assertFalse(analysis['forecast']['available'])
        answer = self.post('/api/chat',{'transactions':[rows[1]],'message':'expenses?'})
        self.assertIn('20.00',answer['content'])

    def test_reject_cross_origin_and_invalid_payload(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post('/api/import',{'csv':'x'},'https://example.com')
        self.assertEqual(error.exception.code,403)
        error.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post('/api/analyze',{'transactions':['not a row']})
        self.assertEqual(error.exception.code,400)
        error.exception.close()

    def test_private_files_not_public(self):
        for path in ('/.env','/data/finance.sqlite3','/backend/server.py','/config.js'):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(self.url+path)
            self.assertEqual(error.exception.code,404)
            error.exception.close()
