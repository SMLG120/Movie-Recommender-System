from flask import Flask, request, jsonify
import random
import time
import os
from datetime import datetime

app = Flask(__name__)

CANARY_PERCENTAGE = 84

# Simulate two model versions
class ModelV1:
    def predict(self, features):
        """Simple model: sum of features"""
        time.sleep(0.1)
        return sum(features)

class ModelV2:
    def predict(self, features):
        """Improved model: weighted sum"""
        time.sleep(0.12)
        weights = [1.5, 2.0, 1.2]
        prediction = sum(f * w for f, w in zip(features, weights[:len(features)]))
        return prediction

model_v1 = ModelV1()
model_v2 = ModelV2()

# Metrics tracking
metrics = {
    "v1": {"requests": 0, "latency": [], "errors": 0},
    "v2": {"requests": 0, "latency": [], "errors": 0}
}

def select_model():
    """Route traffic based on canary percentage"""
    if random.random() * 100 < CANARY_PERCENTAGE:
        return model_v2, "v2"
    return model_v1, "v1"

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with canary routing"""
    try:
        data = request.json
        features = data.get('features', [])
        
        if not features:
            return jsonify({"error": "No features provided"}), 400
        
        model, version = select_model()
        
        start_time = time.time()
        prediction = model.predict(features)
        latency = time.time() - start_time
        
        metrics[version]["requests"] += 1
        metrics[version]["latency"].append(latency)
        
        return jsonify({
            "prediction": round(prediction, 2),
            "model_version": version,
            "latency_ms": round(latency * 1000, 2)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """View current metrics for both models"""
    total_requests = sum(m["requests"] for m in metrics.values())
    
    summary = {}
    for version in ["v1", "v2"]:
        latencies = metrics[version]["latency"]
        requests = metrics[version]["requests"]
        
        summary[version] = {
            "requests": requests,
            "traffic_percentage": round(requests / total_requests * 100, 1) if total_requests > 0 else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2) if latencies else 0,
            "errors": metrics[version]["errors"]
        }
    
    return jsonify({
        "canary_setting": f"{CANARY_PERCENTAGE}%",
        "total_requests": total_requests,
        "models": summary,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/reset', methods=['POST'])
def reset_metrics():
    """Reset all metrics"""
    global metrics
    metrics = {
        "v1": {"requests": 0, "latency": [], "errors": 0},
        "v2": {"requests": 0, "latency": [], "errors": 0}
    }
    return jsonify({"message": "Metrics reset successfully"})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "canary_percentage": CANARY_PERCENTAGE})

if __name__ == '__main__':
    print(f"Starting Canary Deployment Service with {CANARY_PERCENTAGE}% canary traffic")
    app.run(host='0.0.0.0', port=5000)
