import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from backend import server
from backend.model import Classifier


class StreamlitTests(unittest.TestCase):
    def test_sample_dashboard_calls_api_and_renders_charts(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest('Install requirements-ml.txt for Streamlit integration tests')
        with tempfile.TemporaryDirectory() as directory:
            old_db, old_model, old_metrics = server.DB_PATH, server.MODEL, server.METRICS_PATH
            server.DB_PATH = str(Path(directory)/'test.sqlite3')
            server.MODEL = Classifier()
            server.METRICS_PATH = str(Path(directory)/'metrics.sqlite3')
            http = ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
            thread = threading.Thread(target=http.serve_forever,daemon=True)
            thread.start()
            try:
                with patch.dict(os.environ,{'FINANCE_API_URL':f'http://127.0.0.1:{http.server_port}'}):
                    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'streamlit_app.py'),default_timeout=30).run()
                    self.assertEqual(len(app.exception),0)
                    sample = next(button for button in app.sidebar.button if button.label=='Load sample transactions')
                    sample.click().run()
                    self.assertEqual(len(app.exception),0)
                    self.assertEqual(len(app.metric),4) # income, expense, cash flow, forecast
                    self.assertEqual(len(app.get('plotly_chart')),2)
                    self.assertTrue(any('Unusual spending' in item.value for item in app.subheader))
            finally:
                http.shutdown()
                http.server_close()
                server.DB_PATH, server.MODEL, server.METRICS_PATH = old_db, old_model, old_metrics
