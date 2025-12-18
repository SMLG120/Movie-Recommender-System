import json
from download_data import DataPipeline
from retrain_manager import RetrainingManager
from retrain_manager import DataPipeline, DataRefreshManager

def main():
    # 1. Refresh data
    pipeline = DataPipeline()
    refresh_mgr = DataRefreshManager(data_dir="data")
    report = refresh_mgr.refresh(pipeline)
    print(json.dumps(report, indent=2))
    new_users = report['stats']['users']
    new_interactions = report['stats']['interactions']

    # 2. Load retrain manager + state
    retrain_mgr = RetrainingManager(data_dir="data")
    state = retrain_mgr.load_retrain_state()

    # 3. Compute accumulated unseen data
    current = report["stats"]
    unseen_users = new_users - state["users_seen"]
    unseen_interactions = new_interactions - state["interactions_seen"]

    print(f"Unseen users since last retrain: {unseen_users}")
    print(f"Unseen interactions since last retrain: {unseen_interactions}")

    # 4. Gate retraining
    if unseen_users < 50 and unseen_interactions < 100:
        print("Insufficient new data for retraining. Exiting.")
        return

    # 5. Retrain
    promoted = retrain_mgr.run()

    if promoted:
        retrain_mgr.save_retrain_state(current)
        print("PROMOTED")
        sys.exit(10)
    else:
        print("NOT_PROMOTED")
        sys.exit(0)

    # if promoted:
    #     retrain_mgr.save_retrain_state(current)
    #     print("Model promoted → triggering canary deploy")
    #     subprocess.run(
    #         ["python", "deploy_app.py"],
    #         check=True,
    #     )
    # else:
    #     print("No promotion → skipping deployment")

if __name__ == "__main__":
    main()
