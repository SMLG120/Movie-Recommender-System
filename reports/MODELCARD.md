# Movie Recommendations Model Card
## Model Details
This project implements a movie recommendation system for a simulated streaming service with around 1M customers and around 27K movies. The system receives user activity logs (movie requests, ratings, and recommendation requests) and responds with personalized movie recommendations through an inference service.

### Model Date
October 2025

### Model Type
XGBoost Model

### Papers and Resources
[XGBoost Github](https://github.com/dmlc/xgboost)

## Model Use

### Primary Intended Uses
The primary intended use for this model is generating a list of 20 movie recommendations to a specific user given the data of 1 million customers and 27k movies available and the demographic information of the specified user.

### Primary Intended Users

The primary intended users of this model are people engaging with the movie streaming service who are looking for personalized movie recommendations.

### Out of Scope Use Cases

This model is meant to be applied for entertainment purposes only. It should not be used to make interpretations of users' personal details or as a ranking system for movies.

## Factors

The user data includes age, occupation, and gender attributes. These 3 factors are considered to find users with similar demographic information. The movie data includes the title, original language, release date, runtime, popularity rating, vote average, vote count, and genres. Additionally, rating and watch time data is provided which includes the user and movie ids and rating and duration of watch, repesctively, to demonstrate users' level of engagement with a film. Based on the provided data, the model is meant to predict the rating which the specified user would give each film. Then, the inference service is meant to generate the top 20 predicted movies and print the results in descending order by predicted rating.

## Metrics

The metrics used to measure model performance are Mean Squared Error (MSE) and Mean Absolute Error (MAE). [MSE](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error) measures the square of the difference between the predicted and actual value, so a successful model will have a low MSE. [MAE](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error) measure the absolute value of the difference between the actual and predicted value. The difference between MSE and MAE is that since MSE squares the difference between actual and predicted, a large error is weighted more heavily. Thus, MSE is better for identifying the presence of outlier values. 

The [r2](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error) score is also taken which evaluates how the model will perform on data not yet tested.

## Evaluation Data

The model states that 20% of the provided data will be set aside for testing by sklearn.model_selection train_test_split. 

### Preprocessing

The preprocess method divides the data into 3 sections, categorical, numeric, and binary-hot and handles encoding. The tune method performs hyperparamter tuning and returns the best estimator.

## Training Data

The dataset is split 80-20 training and testing data, so the training data originates from the same dataset which testing is performed.

## Ethical Considerations

The dataset from which this model is based does not reveal any personal information, as information is anonymized and users are identified through a user id. This model could not influence matters central to human life. Although this system does provide real-time movie recommendations to users, the worst risks and harms associated with this model is that a user chooses not to watch a certain movie because it is not on their recommended list. There is no risk of injury or harm to human life or the artistic work of the filmmakers involved. 

## Caveats and Recommendations

One caveat to consider is that we did not evaluate the demographic distribution of the user data available. Thus, for folks on the edge of the age range avaialble or those who have a less common occupation, we have not tested by introducing additional data to fill in these gaps whether the system's recommendations would be impacted.









