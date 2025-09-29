# TODO for Installing
- [x] Create virtual environment (venv) in order to download all necessary python libraries(numpy, sklearn, pandas, etc.)

# TODO for MLP Training Implementation

- [x] Update requirements.txt to include tensorflow
- [x] Modify train_model.py to train MLP model for rating prediction
- [x] Modify app.py to load and use the trained MLP model for recommendations
- [x] Install updated dependencies
- [x] Run training script to generate model
- [ ] Test the inference service

# TODO for Integrating Watch Time Data

- [x] Update train_model.py to handle watch_time.csv: add MinMaxScaler for watch_time, update comments/variable names, change output file names
- [x] Update app.py to load updated model/mappings and adjust prediction logic for watch_time
- [x] Run training script with Data/watch_time.csv to generate watch_time model
- [x] Test the updated inference service with watch_time predictions

# TODO for Logistic Regression Implementation

- [x] Create src/train_model_logistic_regression.py for logistic regression training on binary watch_time classification
- [x] Update src/compare_complexities.py to include logistic regression in complexity comparisons
- [x] Test the updated compare_complexities.py script
