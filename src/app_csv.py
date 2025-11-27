import json
import pprint
from provenance_csv import (
    register_model,
    record_prediction,
    trace_prediction,
    get_model_by_version,
    get_predictions_by_model,
    get_all_models,
    get_all_predictions
)

def main():
    print("\n" + "="*60)
    print("CSV-Based Provenance System - Test Run")
    print("="*60 + "\n")

    # Step 1: Register a model
    print("Step 1: Registering model...")
    model_version = register_model({
        "git_commit": "a7b9c3def456",
        "artifact_path": "/models/xgb_model_v1.pkl",
        "pipeline_version": "pipeline_v1.2",
        "python_env_hash": "py3.10-reqs-abc123",
        "training_data_id": "data_20251115",
        "training_data_range_start": "2025-11-10",
        "training_data_range_end": "2025-11-15",
        "training_row_count": 5000,
        "metrics_json": {"ndcg@20": 0.142, "recall@20": 0.18}
    })
    print(f"✓ Model registered: {model_version}\n")

    # Step 2: Record predictions
    print("Step 2: Recording predictions...")
    request_ids = []
    for i in range(3):
        request_id = record_prediction({
            "userid": f"user_{i+1}",
            "model_version": model_version,
            "prediction": json.dumps([f"movie_{j}" for j in range(5)]),
            "input_data": {
                "user_history": [1, 2, 3],
                "user_features": {"age": 25, "country": "US"}
            },
            "extra_json": {
                "inference_latency_ms": 42 + i*10,
                "model_confidence": 0.85 + i*0.02
            }
        })
        request_ids.append(request_id)
    print()

    # Step 3: Trace a prediction
    print("Step 3: Tracing first prediction...")
    trace_result = trace_prediction(request_ids[0])
    if trace_result:
        print("Prediction Event:")
        pprint.pprint(trace_result["event"], width=60)
        print("\nModel Provenance:")
        pprint.pprint(trace_result["provenance"], width=60)
    print()

    # Step 4: Query predictions by model
    print("Step 4: Querying predictions by model...")
    predictions = get_predictions_by_model(model_version, limit=10)
    print(f"✓ Found {len(predictions)} predictions for model {model_version}\n")

    # Step 5: View all models
    print("Step 5: All registered models...")
    all_models = get_all_models()
    print(f"✓ Total models: {len(all_models)}")
    for model in all_models:
        print(f"  - {model['model_version']}: {model['git_commit']}")
    print()

    # Step 6: View CSV file locations
    print("Step 6: CSV file locations...")
    from provenance_csv import _PROV_PATH, _PRED_PATH
    print(f"Model provenance CSV: {_PROV_PATH}")
    print(f"Prediction events CSV: {_PRED_PATH}")
    print()

    print("="*60)
    print("✓ CSV-based provenance system test completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
