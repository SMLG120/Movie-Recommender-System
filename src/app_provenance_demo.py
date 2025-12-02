import json
import pprint
from provenance_csv import (
    register_training_data,
    register_model,
    record_prediction,
    trace_prediction,
    get_model_by_version,
    get_all_models,
    get_predictions_by_model
)

def main():
    print("\n" + "="*70)
    print("CSV-Based Provenance Tracking System - Comprehensive Demo")
    print("="*70 + "\n")

    # Step 1: Register training data
    print("Step 1: Registering training data...")
    data_id = register_training_data({
        "data_source": "movie_ratings_db",
        "file_path": "data/training_data_v2.csv",
        "row_count": 5000,
        "column_count": 12,
        "missing_values_json": {"age": 15, "income": 30},
        "date_range_start": "2025-11-10",
        "date_range_end": "2025-11-15"
    })
    print(f"✓ Training data registered: {data_id}\n")

    # Step 2: Register Model A
    print("Step 2: Registering Model A...")
    model_a = register_model({
        "artifact_path": "models/xgb_model_v1.pkl",
        "training_data_id": data_id,
        "model_type": "XGBoost",
        "pipeline_version": "pipeline_v1.0",
        "training_row_count": 5000,
        "metrics_json": {
            "ndcg@20": 0.142,
            "recall@20": 0.18,
            "accuracy": 0.89
        },
        "framework_versions": {
            "xgboost": "1.7.5",
            "scikit-learn": "1.3.0",
            "pandas": "1.5.3"
        }
    })
    print(f"✓ Model A registered: {model_a}\n")

    # Step 3: Register Model B (Retrained version of Model A)
    print("Step 3: Registering Model B (Retrained version)...")
    model_b = register_model({
        "artifact_path": "models/xgb_model_v2.pkl",
        "training_data_id": data_id,
        "model_type": "XGBoost",
        "pipeline_version": "pipeline_v1.1",
        "training_row_count": 5000,
        "metrics_json": {
            "ndcg@20": 0.158,
            "recall@20": 0.22,
            "accuracy": 0.91
        },
        "framework_versions": {
            "xgboost": "1.7.5",
            "scikit-learn": "1.3.0",
            "pandas": "1.5.3"
        }
    })
    print(f"✓ Model B registered: {model_b}\n")

    # Step 4: Record predictions using Model A
    print("Step 4: Recording predictions with Model A...")
    request_ids_a = []
    for i in range(3):
        request_id = record_prediction({
            "userid": f"user_{i+1}",
            "model_version": model_a,
            "model_tag": "model_a_v1",
            "prediction": json.dumps([f"movie_{j}" for j in range(5)]),
            "input_data": {
                "user_history": [1, 2, 3],
                "user_features": {"age": 25, "country": "US"}
            },
            "inference_latency_ms": 42 + i*10,
            "extra_json": {
                "model_confidence": 0.85 + i*0.02,
                "cache_hit": i % 2 == 0
            }
        })
        request_ids_a.append(request_id)
    print(f"✓ Recorded {len(request_ids_a)} predictions with Model A\n")

    # Step 5: Record predictions using Model B
    print("Step 5: Recording predictions with Model B...")
    request_ids_b = []
    for i in range(3):
        request_id = record_prediction({
            "userid": f"user_{i+1}",
            "model_version": model_b,
            "model_tag": "model_b_v2",
            "prediction": json.dumps([f"movie_{j}" for j in range(5)]),
            "input_data": {
                "user_history": [1, 2, 3],
                "user_features": {"age": 25, "country": "US"}
            },
            "inference_latency_ms": 38 + i*10,
            "extra_json": {
                "model_confidence": 0.88 + i*0.02,
                "cache_hit": i % 2 == 0
            }
        })
        request_ids_b.append(request_id)
    print(f"✓ Recorded {len(request_ids_b)} predictions with Model B\n")

    # Step 6: Trace a prediction from Model A
    print("Step 6: Tracing prediction from Model A...")
    trace_result_a = trace_prediction(request_ids_a[0])
    if trace_result_a:
        print("Prediction Event:")
        pprint.pprint(trace_result_a["event"], width=70)
        print("\nModel Provenance (Model A):")
        pprint.pprint(trace_result_a["model_provenance"], width=70)
        print("\nTraining Data Provenance:")
        pprint.pprint(trace_result_a["data_provenance"], width=70)
    print()

    # Step 7: Trace a prediction from Model B
    print("Step 7: Tracing prediction from Model B...")
    trace_result_b = trace_prediction(request_ids_b[0])
    if trace_result_b:
        print("Prediction Event:")
        pprint.pprint(trace_result_b["event"], width=70)
        print("\nModel Provenance (Model B):")
        pprint.pprint(trace_result_b["model_provenance"], width=70)
    print()

    # Step 8: Compare Model A vs Model B metrics
    print("Step 8: Comparing Model A vs Model B...")
    model_a_info = get_model_by_version(model_a)
    model_b_info = get_model_by_version(model_b)

    metrics_a = json.loads(model_a_info["metrics_json"])
    metrics_b = json.loads(model_b_info["metrics_json"])

    print(f"\nModel A Metrics: {metrics_a}")
    print(f"Model B Metrics: {metrics_b}")
    print(f"NDCG@20 improvement: {(metrics_b['ndcg@20'] - metrics_a['ndcg@20']) / metrics_a['ndcg@20'] * 100:.2f}%")
    print()

    # Step 9: View all models
    print("Step 9: All registered models...")
    all_models = get_all_models()
    print(f"Total models registered: {len(all_models)}")
    for model in all_models:
        print(f"  - {model['model_version']}: {model['model_tag']} (git: {model['git_commit']})")
    print()

    # Step 10: Query predictions by model
    print("Step 10: Query predictions by model...")
    preds_a = get_predictions_by_model(model_a, limit=10)
    preds_b = get_predictions_by_model(model_b, limit=10)
    print(f"Model A predictions: {len(preds_a)}")
    print(f"Model B predictions: {len(preds_b)}")
    print()

    # Step 11: CSV file locations
    print("Step 11: Provenance tracking files...")
    from provenance_csv import _PROV_PATH, _DATA_PROV_PATH, _PRED_PATH
    print(f"Model provenance CSV: {_PROV_PATH}")
    print(f"Data provenance CSV: {_DATA_PROV_PATH}")
    print(f"Prediction events CSV: {_PRED_PATH}")
    print()

    print("="*70)
    print("✓ Provenance tracking demo completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
