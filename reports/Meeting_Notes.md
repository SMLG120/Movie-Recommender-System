Milestone 1  
Samuel Ha, Sanghyun Hong, Jessica Ojo, Marie Check, Eric Qiu

## Meeting 1:

## Model

* Training data  
  * We have movie and user information \- we’ll just use the data provided   
  * What format of the data is needed to train the model? Json, csv?  
  * What features should be used from the data? Are we using all the features?  
* Model training methods  
  * What are recommended models?  
  * What methods are used to train them? And the effectiveness of each method?   
  * Are we fine tuning or training a new model?  
* Model evaluation   
  * Are there ethical concerns and should we be evaluating for that?  
  * How to evaluate recommender models?  
* Model deployments and model card (is model deployment locally, docker, huggingface)  
  * How should the model be deployed? Docker or server?

## Model Inference

Rest API \- inference with http://\<ip-of-your-virtual-machine\>:8080/recommend/\<userid\>  
**output** \- comma-separated list of 20 movie recommendations, from most to least highly recommended for `USER_ID movie IDs in a single line; we consider the first movie ID as the highest recommended movie.`

* Build a rest API? Does it have to be a rest API? 

## Tips from class

Host model in team id not personal id   
Information on how users interact with the movies are already provided  
Information about the movies and users are also provided   
No past recommendation information, only users patterns

## Conditions:

* Infra:  
  * Receives data through API & Can access log file through Kafka stream  
  * Prediction server on our VM  
  * You may build a distributed system with multiple machines where you use the provided virtual machine only as API broker or load balancer.  
* Provided Data:  
  * Read-only access to API & Both API provide info in JSON format   
  * Note that movie data, including "vote\_average", "vote\_count" and "popularity"fields come from an external database and do not reflect data on the movie recommendation service itself.  
* Languages, tools, and frameworks:  
  * Any Tech Stack is allowed ⇒ MUST USE Docker container   
  * Can use external data and services (i.g., cloud service) ⇒ makes sure to configure them with reasonable security measures. **must make them available to course staffs to evaluate**  
* Documentation:  
  * README.md or Wiki pages on GitLab  
  * you **must** document your design decision and implementation

## Meeting 2:

**Recommender systems** 

* Content-Based Approach  
  Content-based filtering methods are based on a description of the item and a profile of the user's preferences. To train a Machine Learning model with this approach we can use a k-NN model. For instance, if we know that user u bought an item i, we can recommend to u the available items with features most similar to i. Logistic Regression / Linear SVM, XGBoost or MLP  
  Example: User 1 is a 34-year-old male in sales/marketing; movie is a thriller with certain runtime, languages, etc.  
  → Predicts based on similarity between item attributes and user profile/history.  
* useful for coldstart scenarios like ours  
* disadvantage is that each user is treated independently  
    
* Collaborative Filtering Approach  
  leverage the feedbacks or activity history of all users in order to predict the rating of a user on a given item. It works with the assumption that users who agreed in the past will agree in the future, and that they will like similar kinds of items as they liked in the past.  
  Uses user–item interaction history. Matrix Factorization, Neural CF  
  → Requires enough overlap between users and movies. (which we don’t have). We could use [MovieLens 100k](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)   
    
* Hybrid Approaches  
  use both the known metadata and the set of observed user-item interactions. combines advantages of both Content-Based and Collaborative Filtering methods, and allows to obtain the best results. Recommended package \- [LightFM](https://github.com/lyst/lightfm) , Paper; [DeepFM](https://arxiv.org/abs/1703.04247)


Types of Hybrid Approaches

* Weighted \- can be built with XGBoost  
* Switching \-  
* Mixed  
* Feature combination \- For example, we can inject features of a collaborative recommendation model into an content-based recommendation model. The hybrid model is capable to consider the collaborative data from the sub system with relying on one model exclusively.  
* Feature augmentation \- A contributing recommendation model is employed to generate a rating or classification of the user/item profile, which is further used in the main recommendation system to produce the final predicted result.  
* Joint Models (Deep Hybrid): Train a single neural architecture with two branches:  
  Branch A: learns collaborative embeddings (user\_id, movie\_id).  
  Branch B: learns content embeddings (user demographics, movie metadata, text embeddings).

Concatenate both → pass through MLP → predict rating/interaction.  
Example: Wide & Deep, DeepFM, or custom PyTorch/TF hybrid.  
Pros: End-to-end, scalable, captures interactions.	  
Cons: Harder to tune, needs more data.

**Final Decision:**  
Approach of choice: Content-based; XGBoost, logistic regression and MLP  
	(Keep in mind: final model shouldn’t down-sample too much for the **final model**)  
Any ethical consideration \- None so far, movie data is public  
Model deployment \- upload model to huggingface, display learning from model card class  
Create a restAPI to run our inference logic and get output  
Docker to deploy the API configurations, which allows access to our configured API \- docker image.  
Report \- final stage 

## Work Distribution

| Goal | Tasks | Deadline | Assignee(s) |
| :---- | :---- | :---- | :---- |
| Train Content based model | \- format the data. Collect all information into a CSV  | Wednesday \- Sep 24th | Jess |
|  | \- Train XGBoost model (Jess) \- Train logistic regression (Eric) \- Train MLP (Samuel, Harry) | Friday \- sep 26th | Jess, Eric, Samuel, Harry |
| Model deployment  | \- deploy model to huggingface (Jess) \- model card (Marie) | Saturday/Sunday \- Sep 27th/28th | Jess, Marie |
| Inference Logic API | \- Create a working flaskAPI or RestAPI (Jess) \- Write unit test to check inputs, outputs, error handling (Marie) |  Monday Sep 29th | Jess, Marie |
| Docker deployment | Create dockerfile and docker image  | Tuesday 30th | Harry, Samuel. Jess on assist |
| Report writing | \-Take down notes as you code/test \-Compile them into report | As we go- final product Tuesday Sep 30th | Everyone |
| Report Submission | Double-checking report writing | Wednesday Oct 1st | Eric |

## Team Task Breakdown in Detail

**1\. Data & Feature Engineering** 

* Collect dataset(s).

* Clean/transform into user–item interaction matrix.

* Engineer basic features:

  * user history (ratings, watch counts),

  * item metadata (genre, year).

* Document how the dataset was prepared.

* Push cleaned dataset and preprocessing scripts into repo.

Deliverables: `data/` folder, preprocessing script(s), notes for report.

---

 **2\. Model Training** 

* Choose a simple baseline recommender:

  * Collaborative filtering (e.g., matrix factorization, nearest-neighbors), OR

  * Content-based (cosine similarity by genres).

* Train the model using prepared data.

* Save model weights or matrices in a serialized format.

* Provide code & notebook for training.

Deliverables: `models/train.py`, stored trained model files.

---

**3\. Inference Service** 

* Write an HTTP service (e.g., Flask, FastAPI, Node/Express).

Endpoint:

 `GET /recommend/<userid>`

*  returns up to 20 movie IDs, ranked best → worst.

* Load trained model artifacts at startup.

* Implement ranking logic from Person B’s model.

* Test locally with curl.

 Deliverables: `service/app.py` (or `main.js`), Dockerfile for deployment.

**4\. Deployment & Integration** 

* Containerize the service with Docker.

* Set up port 8080 exposure inside container.

* Deploy to the team’s VM.

* Verify API works from external curl.

* Check Kafka logs for requests.

* Write instructions in `README.md` for how to run service.

 Deliverables: `Dockerfile`, deployment scripts, verified API endpoint running.

---

**5\. Team Workflow & Report** 

* Act as **project manager**

* Take notes at each team meeting:

  * what progress was made,

  * what ideas discussed,

  * ToDos for each member.

* Collect commit history & link to evidence of contributions.

* Write/report team process & reflections (1 page).

* Assemble the final PDF with sections from others.

* Push into `reports/M1.pdf`.

