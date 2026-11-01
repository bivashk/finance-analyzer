import unittest
from ml.prepare import make_splits, text_group
from ml.demo import demo_rows
from backend.anomalies import detect
from ml.evaluate_study import evaluate


class WorkflowTests(unittest.TestCase):
    def test_normalized_variants_never_cross_split(self):
        rows = []
        for label in ('Food','Transport'):
            for merchant in range(20):
                identity = chr(97+merchant)
                for variation in range(4):
                    rows.append({'description':f'[debit] {label} merchant{identity} #{variation}', 'category':label})
        split, conflicts = make_splits(rows,train_cap=20,validation_cap=8,test_cap=8)
        groups = {s:{r['group'] for r in split if r['split']==s} for s in ('train','validation','test')}
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])
        self.assertEqual(conflicts,0)
        self.assertEqual(text_group('[debit] CAFE #123'),text_group('[credit] cafe #567'))

    def test_spending_outlier_flag_and_small_batch_guard(self):
        result = detect(demo_rows())
        self.assertTrue(result['available'])
        self.assertIn('demo-outlier',[r['id'] for r in result['flags']])
        self.assertFalse(detect(demo_rows()[:3])['available'])

    def test_study_requires_observations_and_reports_actual_values(self):
        with self.assertRaises(ValueError):
            evaluate([])
        # Synthetic unit-test inputs, never written as study evidence.
        observations = [{'participant_id':str(i),'condition':c,'tasks_total':'10',
                         'correct_insights':str(n),'seconds':str(t)} for i in range(10)
                        for c,n,t in [('baseline',5,100),('enhanced',6,90)]]
        result = evaluate(observations)
        self.assertAlmostEqual(result['relative_improvement_percent'],20)
        self.assertAlmostEqual(result['mean_task_time_reduction_percent'],10)
        observations[-1]['tasks_total']='9'
        with self.assertRaises(ValueError):
            evaluate(observations)
