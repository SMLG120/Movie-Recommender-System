import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class OnlineEvaluator:
    def __init__(self, log_path='logs/online_metrics.json'):
        self.log_path = log_path
        self.metrics = {
            'response_times': [],
            'recommendations': [],
            'user_interactions': [],
            'watch_times': []
        }
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def log_recommendation(self, user_id, recommended_items, response_time):
        """Log a recommendation event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'items': recommended_items,
            'response_time': response_time
        }
        self.metrics['recommendations'].append(event)
        self.metrics['response_times'].append(response_time)
        self._save_metrics()
    
    def log_interaction(self, user_id, item_id, action_type, watch_time=None):
        """Log user interaction with recommended item"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'item_id': item_id,
            'action_type': action_type,
            'watch_time': watch_time
        }
        self.metrics['user_interactions'].append(event)
        if watch_time is not None:
            self.metrics['watch_times'].append(watch_time)
        self._save_metrics()
    
    def compute_online_metrics(self, window_hours=24):
        """Compute metrics for recent data"""
        now = datetime.now()
        recent_recs = [r for r in self.metrics['recommendations'] 
                      if (now - datetime.fromisoformat(r['timestamp'])).total_seconds() < window_hours * 3600]
        
        recent_interactions = [i for i in self.metrics['user_interactions']
                             if (now - datetime.fromisoformat(i['timestamp'])).total_seconds() < window_hours * 3600]
        
        if not recent_recs or not recent_interactions:
            return {}
        
        metrics = {
            'avg_response_time': np.mean(self.metrics['response_times'][-1000:]),
            'p95_response_time': np.percentile(self.metrics['response_times'][-1000:], 95),
            'interaction_rate': len(recent_interactions) / len(recent_recs),
            'avg_watch_time': np.mean(self.metrics['watch_times'][-1000:]) if self.metrics['watch_times'] else 0
        }
        
        return metrics
    
    def _save_metrics(self):
        """Save metrics to disk"""
        with open(self.log_path, 'w') as f:
            json.dump(self.metrics, f)

def get_evaluator():
    """Singleton pattern to get evaluator instance"""
    if not hasattr(get_evaluator, 'instance'):
        get_evaluator.instance = OnlineEvaluator()
    return get_evaluator.instance