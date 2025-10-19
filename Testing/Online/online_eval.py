import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from collections import defaultdict

class OnlineEvaluator:
    def __init__(self, log_path='logs/online_metrics.json'):
        self.log_path = log_path
        self.metrics = {
            'response_times': [],
            'recommendations': [],
            'user_interactions': [],
            'watch_times': [],
            'errors': [],
            'model_versions': [],
            'recommendation_quality': []
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
    
    def log_error(self, error_type, error_message):
        """Log error events"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message
        }
        self.metrics['errors'].append(event)
        self._save_metrics()
    
    def log_model_deployment(self, model_version, model_type):
        """Log model deployment events"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'version': model_version,
            'type': model_type
        }
        self.metrics['model_versions'].append(event)
        self._save_metrics()
    
    def log_recommendation_quality(self, user_id, recommended_items, selected_item, satisfaction_score):
        """Log recommendation quality metrics"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'recommendations': recommended_items,
            'selected': selected_item,
            'satisfaction': satisfaction_score
        }
        self.metrics['recommendation_quality'].append(event)
        self._save_metrics()
    
    def compute_online_metrics(self, window_hours=24):
        """Enhanced metrics computation"""
        now = datetime.now()
        window_start = now - timedelta(hours=window_hours)
        
        # Filter recent data
        recent_data = {
            key: [
                event for event in events
                if datetime.fromisoformat(event['timestamp']) > window_start
            ]
            for key, events in self.metrics.items()
        }
        
        if not recent_data['recommendations']:
            return {}
        
        basic_metrics = {
            'avg_response_time': np.mean(self.metrics['response_times'][-1000:]),
            'p95_response_time': np.percentile(self.metrics['response_times'][-1000:], 95),
            'interaction_rate': len(recent_data['user_interactions']) / len(recent_data['recommendations']),
            'avg_watch_time': np.mean(self.metrics['watch_times'][-1000:]) if self.metrics['watch_times'] else 0
        }
        
        # Add error rate
        error_rate = len(recent_data['errors']) / len(recent_data['recommendations'])
        basic_metrics['error_rate'] = error_rate
        
        # Add recommendation quality metrics
        if recent_data['recommendation_quality']:
            satisfaction_scores = [
                event['satisfaction']
                for event in recent_data['recommendation_quality']
            ]
            basic_metrics['avg_satisfaction'] = np.mean(satisfaction_scores)
        
        return basic_metrics
    
    def _save_metrics(self):
        """Save metrics to disk"""
        with open(self.log_path, 'w') as f:
            json.dump(self.metrics, f)

def get_evaluator():
    """Singleton pattern to get evaluator instance"""
    if not hasattr(get_evaluator, 'instance'):
        get_evaluator.instance = OnlineEvaluator()
    return get_evaluator.instance