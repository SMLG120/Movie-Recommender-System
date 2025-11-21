import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class OnlineEvaluator:
    """
    Simplified online evaluator for movie recommender system.
    Tracks recommendations, interactions, response times, errors, and model quality.
    """

    def __init__(self, log_path='Testing/Online/logs/online_metrics.json', alert_threshold=0.1):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Metrics storage
        self.metrics = {
            'recommendations': [],
            'user_interactions': [],
            'response_times': [],
            'errors': [],
            'model_versions': [],
            'recommendation_quality': []
        }

        # Prometheus metrics
        self.recommendation_counter = Counter('recommendations_total', 'Total recommendations made')
        self.response_time = Histogram('recommendation_response_time_seconds', 'Response time for recommendations')
        self.model_quality = Gauge('model_quality_score', 'Current model quality score')
        self.error_rate = Gauge('recommendation_error_rate', 'Current error rate')

        # Alert thresholds
        self.quality_threshold = alert_threshold
        self.last_alert = None
        self.alert_cooldown = timedelta(hours=1)

        # Start Prometheus server
        try:
            start_http_server(8000)
        except Exception as e:
            warnings.warn(f"Could not start Prometheus server: {e}")

    # ---------------- Logging functions ----------------

    def log_recommendation(self, user_id, recommended_items, response_time):
        """
        Log a recommendation event.
        """
        current_time = datetime.now()

        self.metrics['recommendations'].append({
            'timestamp': current_time.isoformat(),
            'user_id': user_id,
            'items': recommended_items
        })
        self.metrics['response_times'].append(response_time)

        # Update Prometheus
        self.recommendation_counter.inc()
        self.response_time.observe(response_time)

        # Save metrics periodically
        self._save_metrics()

    def log_interaction(self, user_id, item_id, action_type, watch_time=None):
        """
        Log a user interaction (click/watch).
        """
        current_time = datetime.now()
        self.metrics['user_interactions'].append({
            'timestamp': current_time.isoformat(),
            'user_id': user_id,
            'item_id': item_id,
            'action_type': action_type,
            'watch_time': watch_time
        })

        # Update model quality metric (hit rate)
        hit = self._compute_hit(user_id, item_id)
        self.model_quality.set(hit)

        self._save_metrics()

    def log_error(self, error_type, message):
        """
        Log an error event.
        """
        current_time = datetime.now()
        self.metrics['errors'].append({
            'timestamp': current_time.isoformat(),
            'type': error_type,
            'message': message
        })

        # Update Prometheus error rate
        total_recs = max(1, len(self.metrics['recommendations']))
        self.error_rate.set(len(self.metrics['errors']) / total_recs)
        self._save_metrics()

    def log_model_deployment(self, version, model_type):
        """Log model deployment events"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'version': version,
            'model_type': model_type
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

    # ---------------- Metrics computation ----------------

    def _compute_hit(self, user_id, item_id):
        """
        Compute if the latest recommendation for the user contained the interacted item.
        """
        user_recs = [r for r in self.metrics['recommendations'] if r['user_id'] == user_id]
        if not user_recs:
            return 0.0
        latest_items = user_recs[-1]['items']
        return float(item_id in latest_items)

    def compute_online_metrics(self, last_hours=24):
        """
        Compute aggregate online metrics for the last `last_hours` hours.
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=last_hours)

        # Filter events
        recs = [r for r in self.metrics['recommendations'] if datetime.fromisoformat(r['timestamp']) > cutoff]
        interactions = [i for i in self.metrics['user_interactions'] if datetime.fromisoformat(i['timestamp']) > cutoff]
        response_times = [r for t, r in zip(self.metrics['recommendations'], self.metrics['response_times'])
                          if datetime.fromisoformat(t['timestamp']) > cutoff]
        errors = [e for e in self.metrics['errors'] if datetime.fromisoformat(e['timestamp']) > cutoff]

        metrics = {}
        metrics['num_recommendations'] = len(recs)
        metrics['num_interactions'] = len(interactions)
        metrics['avg_response_time'] = np.mean(response_times) if response_times else 0.0
        metrics['p95_response_time'] = np.percentile(response_times, 95) if response_times else 0.0
        metrics['error_rate'] = len(errors) / max(1, len(recs)) if recs else 0.0

        # Hit rate
        hits = sum(float(i['item_id'] in next((r['items'] for r in recs if r['user_id'] == i['user_id']), []))
                   for i in interactions)
        metrics['hit_rate'] = hits / max(1, len(interactions))

        # Coverage
        metrics['user_coverage'] = len(set(r['user_id'] for r in recs))
        metrics['item_coverage'] = len(set(item for r in recs for item in r['items']))

        metrics['model_quality'] = metrics['hit_rate']  # Can be extended with more quality metrics

        return metrics

    # ---------------- Utilities ----------------

    def _save_metrics(self):
        """
        Save metrics to disk with error handling.
        """
        try:
            with open(self.log_path, 'w') as f:
                json.dump(self.metrics, f)
        except Exception as e:
            warnings.warn(f"Failed to save metrics: {e}")


# ---------------- Singleton accessor ----------------

def get_evaluator():
    """
    Singleton pattern for getting the online evaluator.
    """
    if not hasattr(get_evaluator, 'instance'):
        get_evaluator.instance = OnlineEvaluator()
    return get_evaluator.instance