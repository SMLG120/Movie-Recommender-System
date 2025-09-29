from huggingface_hub import HfApi, HfFolder, Repository, upload_file

repo_id = "comp585Team6/recommender_model1"

upload_file(
    path_or_fileobj="src/models/xgb_recommender.joblib",
    path_in_repo="xgb_recommender.joblib",
    repo_id=repo_id,
    repo_type="model"
)
