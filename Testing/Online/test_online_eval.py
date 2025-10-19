import unittest
from datetime import datetime, timedelta
import numpy as np
from online_eval import OnlineEvaluator

class TestOnlineEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = OnlineEvaluator(log_path='test_metrics.json')
    
    def test_recommendation_logging(self):
        self.evaluator.log_recommendation('u1', ['m1', 'm2'], 0.1)
        metrics = self.evaluator.compute_online_metrics()
        self.assertIn('avg_response_time', metrics)
        self.assertEqual(metrics['avg_response_time'], 0.1)
    
    def test_interaction_logging(self):
        self.evaluator.log_interaction('u1', 'm1', 'watch', 30)
        self.evaluator.log_interaction('u1', 'm2', 'skip', 0)
        metrics = self.evaluator.compute_online_metrics()
        self.assertIn('avg_watch_time', metrics)
        self.assertEqual(metrics['avg_watch_time'], 30)
    
    def test_time_window_filtering(self):
        # Log old data
        self.evaluator.metrics['recommendations'] = [{
            'timestamp': (datetime.now() - timedelta(hours=25)).isoformat(),
            'user_id': 'u1',
            'items': ['m1', 'm2'],
            'response_time': 0.1
        }]
        
        # Log recent data
        self.evaluator.log_recommendation('u2', ['m3', 'm4'], 0.2)
        
        metrics = self.evaluator.compute_online_metrics(window_hours=24)
        self.assertEqual(len(metrics), 4)  # Should only include recent data

if __name__ == '__main__':
    unittest.main()
