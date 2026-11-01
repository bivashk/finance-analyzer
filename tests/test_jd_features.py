import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from backend.calibration import fit_temperature,calibration_metrics,choose_review_threshold
from backend.customer import profile,segment
from backend import operations
from ml.customer_demo import customers
from ml.merchant_prepare import vocabulary,match_merchant


class CalibrationTests(unittest.TestCase):
    def test_temperature_is_fitted_without_test_data_and_policy_can_abstain(self):
        logits = np.array([[8.,0.],[8.,0.],[0.,8.],[0.,8.]])
        y = np.array([0,1,1,0])
        temperature = fit_temperature(logits,y)
        self.assertGreater(temperature,1)
        self.assertLess(calibration_metrics(logits,y,temperature)['negative_log_likelihood'],
                        calibration_metrics(logits,y)['negative_log_likelihood'])
        policy = choose_review_threshold(logits,y,temperature,minimum=2)
        self.assertEqual(policy['threshold'],1)
        self.assertEqual(sum(b['count'] for b in calibration_metrics(logits,y,temperature)['reliability_bins']),len(y))

    def test_merchant_aliases_share_a_family_and_unknowns_are_excluded(self):
        aliases = vocabulary('const merchants = {\n places: ["UBER", "UBER EATS", "STARBUCKS"]\n};')
        self.assertEqual(match_merchant('UBER EATS #123',aliases),match_merchant('Uber ride',aliases))
        self.assertIsNone(match_merchant('unmatched merchant',aliases))


class CustomerTests(unittest.TestCase):
    def test_profile_uses_observed_finances_and_no_income_is_explicit(self):
        p = profile([{'date':f'2025-0{i}-01','description':'Rent','amount':-100,'category':'Rent'} for i in range(1,4)])
        self.assertFalse(p['income_recorded'])
        self.assertEqual(p['features']['recurring_expense_share'],1)
        self.assertEqual(p['features']['average_monthly_expenses'],100)

    def test_cohort_has_reproducible_descriptive_clusters_and_input_guards(self):
        report = segment(customers())
        self.assertEqual(sum(c['customers'] for c in report['clusters']),36)
        self.assertTrue(-1<=report['silhouette_score']<=1)
        self.assertTrue(0<=report['mean_seed_stability_adjusted_rand']<=1)
        with self.assertRaises(ValueError):
            segment(customers()[:2])
        data = customers()
        data[1]['customer_id'] = data[0]['customer_id']
        with self.assertRaises(ValueError):
            segment(data)


class OperationsTests(unittest.TestCase):
    def test_metrics_are_session_scoped_and_corrections_are_not_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            db = operations.connect(str(Path(directory)/'metrics.sqlite3'))
            predictions = [{'category':'Rent','confidence':.9,'needs_review':False},
                           {'category':'Other','confidence':.4,'needs_review':True}]
            operations.record(db,'session-a',[{'id':'a'},{'id':'b'}],predictions,20,'test')
            self.assertFalse(operations.correction(db,'session-b','a','Rent',predictions[0]['prediction_id']))
            operations.correction(db,'session-a','a','Rent',predictions[0]['prediction_id'])
            operations.correction(db,'session-a','b','Groceries',predictions[1]['prediction_id'])
            metrics = operations.summary(db,'session-a')
            self.assertEqual(metrics['manual_review_rate'],.5)
            self.assertEqual(metrics['correction_rate_among_reviewed'],.5)
            self.assertEqual(metrics['batch_amortized_inference_ms_p50'],10)
            self.assertIsNone(metrics['estimated_compute_cost_per_transaction'])
            self.assertEqual(operations.summary(db,'session-b')['prediction_events'],0)
            db.close()
