"""Runs only after local training; never downloads models in the test suite."""
import unittest
from pathlib import Path
from backend.inference import TransformerClassifier, SentenceClassifier


class TrainedInferenceTests(unittest.TestCase):
    @unittest.skipUnless(Path('models/transformer/model.safetensors').exists(),'Run ml.train first')
    def test_saved_transformer_loads_and_returns_reviewable_predictions(self):
        model = TransformerClassifier()
        output = model.predict(['[credit] SALARY PAYROLL','[debit] NETFLIX'])
        self.assertEqual(len(output),2)
        self.assertTrue(all(0<=row['confidence']<=1 for row in output))
        self.assertTrue(all('bert-tiny-finetuned' in row['model_version'] for row in output))
        self.assertEqual(output[0]['category'],'Income')

    @unittest.skipUnless(Path('models/sentence/head.json').exists(),'Run ml.train first')
    def test_sentence_embeddings_and_json_head_load_offline(self):
        model = SentenceClassifier()
        output = model.predict(['[credit] SALARY PAYROLL','[debit] NETFLIX'])
        self.assertEqual(len(output),2)
        self.assertTrue(all(0<=row['confidence']<=1 for row in output))
        self.assertEqual(output[0]['model_version'],'minilm-frozen-logistic')
