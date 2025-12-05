# test_online_eval.py
from online_eval import get_evaluator
import time

def main():
    evaluator = get_evaluator()
    
    # --- Log some recommendations ---
    evaluator.log_recommendation(user_id="u1", recommended_items=["m1", "m2", "m3"], response_time=0.4)
    evaluator.log_recommendation(user_id="u2", recommended_items=["m2", "m4"], response_time=0.6)
    evaluator.log_recommendation(user_id="u3", recommended_items=["m5", "m1"], response_time=0.3)
    
    # --- Simulate user interactions ---
    evaluator.log_interaction(user_id="u1", item_id="m2", action_type="click")
    evaluator.log_interaction(user_id="u1", item_id="m4", action_type="click")
    evaluator.log_interaction(user_id="u2", item_id="m2", action_type="watch", watch_time=20)
    evaluator.log_interaction(user_id="u3", item_id="m5", action_type="watch", watch_time=15)
    
    # --- Log a model deployment ---
    evaluator.log_model_deployment("v1.0", "XGBoost")

    
    # --- Log recommendation quality metrics ---
    evaluator.log_recommendation_quality(
        user_id="u1",
        recommended_items=["m1", "m2", "m3"],
        selected_item="m2",
        rating_score=4.0
    )
    
    evaluator.log_recommendation_quality(
        user_id="u2",
        recommended_items=["m2", "m4"],
        selected_item="m2",
        rating_score=5.0
    )
    
    # --- Wait a few seconds to simulate real-time logging ---
    time.sleep(1)
    
    # --- Compute current online metrics ---
    metrics = evaluator.compute_online_metrics()
    print("=== Online Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
