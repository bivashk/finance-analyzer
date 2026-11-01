import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from backend import server
from backend.model import Classifier


class PublicAPITests(unittest.TestCase):
    def test_public_imports_are_ephemeral_and_metrics_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = (server.PUBLIC_DEMO,server.DB_PATH,server.METRICS_PATH,server.MODEL)
            server.PUBLIC_DEMO = True
            server.DB_PATH = str(Path(directory)/'must-not-exist.sqlite3')
            server.METRICS_PATH = str(Path(directory)/'metrics.sqlite3')
            server.MODEL = Classifier()
            http = ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
            threading.Thread(target=http.serve_forever,daemon=True).start()
            base = f'http://127.0.0.1:{http.server_port}'
            def request(path,payload=None,session='a'*32):
                req = Request(base+path,data=json.dumps(payload).encode() if payload is not None else None,
                    headers={'Content-Type':'application/json','X-Finance-Session':session})
                with urlopen(req) as response:
                    return json.load(response)
            try:
                a = request('/api/import',{'csv':'Date,Description,Amount\n2026-01-01,Salary payroll,100\n'})
                b = request('/api/import',{'csv':'Date,Description,Amount\n2026-01-02,Cafe food,-10\n'},'b'*32)
                self.assertEqual(len(b['transactions']),1)
                self.assertEqual(b['transactions'][0]['description'],'Cafe food')
                request('/api/classify',{'transactions':a['transactions']})
                self.assertEqual(request('/api/metrics')['prediction_events'],1)
                self.assertEqual(request('/api/metrics',session='b'*32)['prediction_events'],0)
                self.assertFalse(Path(server.DB_PATH).exists())
                with self.assertRaises(HTTPError) as error:
                    request('/api/classify',{'transactions':a['transactions']},'bad')
                self.assertEqual(error.exception.code,400)
                error.exception.close()
            finally:
                http.shutdown()
                http.server_close()
                server.PUBLIC_DEMO,server.DB_PATH,server.METRICS_PATH,server.MODEL = previous
