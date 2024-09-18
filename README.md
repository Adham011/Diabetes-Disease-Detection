![hq720](https://github.com/user-attachments/assets/ec1dab2f-7809-4ab7-a817-1cd0d4109220)


# Diabetes Detection

A diabetes detection machine learning project involves using data and algorithms to train a model to accurately predict the likelihood of an individual having diabetes based on various features such as Glucose, age, and blood Pressure.

## Description

Diabetes detection machine learning project is a system that uses algorithms, statistical models and historical data to predict the likelihood of an individual having diabetes. The goal is to use features like Glucose, age, and Blood Pressure to identify individuals who have diabetes or are at risk of developing it. Machine learning models are trained using this data and can then be used to make predictions on new individuals.


## About Dataset

### Context
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.

### Content
Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1)

### Sources
(a) Original owners: National Institute of Diabetes and Digestive and
Kidney Diseases
(b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
Research Center, RMI Group Leader
Applied Physics Laboratory
The Johns Hopkins University
Johns Hopkins Road
Laurel, MD 20707
(301) 953-6231
(c) Date received: 9 May 1990

## Feature Selection

The process of feature selection in machine learning is used to identify and select the most relevant features from a dataset to improve the performance and efficiency of a machine learning model. It is an important step in the model building process as it can help to reduce overfitting, increase model interpretability, and improve the accuracy of predictions.

### Feature Selection methods use in this projects

#### correlation matrix

![image](https://github.com/user-attachments/assets/6fdebe29-a690-40ab-86bc-4230015f10e3)


A correlation matrix can be used to identify highly correlated features, which can then be removed or consolidated. Highly correlated features can cause a problem in machine learning models as they can introduce multicollinearity, which can lead to unstable and unreliable model estimates.

![con variable](https://github.com/nileshparab42/Diabetes-Detection/blob/master/Assets/sns.png)


## Outlier Treatment

Outlier treatment is the process of identifying and handling extreme values or observations that are significantly different from other observations in a dataset. Outliers can have a significant impact on the results of data analysis and modeling, and can skew the mean and standard deviation of a dataset.

### Methods used to identify outliers

- **Visualization:** Outliers can be identified by creating visualizations such as box plots, scatter plots, and histograms to identify observations that fall outside of the typical range.

- **Interquartile range (IQR):** Outliers can be identified by calculating the interquartile range (IQR), which is the difference between the first and third quartile. Values that fall outside of 1.5 * IQR above the third quartile or 1.5 * IQR below the first quartile are considered outliers.

![Outlier](https://github.com/nileshparab42/Diabetes-Detection/blob/master/Assets/outlier.png)


### Methods used to treat outliers

- **Imputing the missing values:** This method can be used to replace outliers with the mean, median, or mode of the dataset.

![Outlier Tret](https://github.com/nileshparab42/Diabetes-Detection/blob/master/Assets/outlier-tret.png)

## Features transformation and scaling

**Feature transformation** is the process of applying mathematical functions to the features in a dataset in order to change their distribution or to extract additional information from the data. Feature transformations are often used to improve the performance of machine learning models by making the features more suitable for the model.

- **Label encoding** is a method of converting categorical variables, represented as text values, into numerical values. It assigns a unique numerical value to each category or level of a categorical feature. This is often used as a preprocessing step before training a machine learning model.

- **Target encoding** is a technique used in machine learning to encode categorical variables. It replaces each category with the average value of the target variable for that category. This can improve the performance of a model by allowing it to better handle categorical variables. It is also known as mean encoding or probability encoding.

**Feature scaling** is the process of normalizing the range of values for each feature in a dataset. This is often done to ensure that all features are on a similar scale and to prevent some features from having a greater impact on the outcome of a machine learning model than others.

- **StandardScaler** is a pre-processing method in machine learning used to standardize a dataset by subtracting the mean and scaling to unit variance. It is commonly used for feature scaling before applying a supervised learning algorithm to a dataset. 

## Model Selection

### Selection of model

- **Logistic regression** is a statistical method used for predicting binary outcomes (i.e. outcomes with two possible results, such as success or failure). It is a type of generalized linear model (GLM) that is used to model a binary dependent variable based on one or more independent variables. 

- **Naive Bayes classifier** is a probabilistic machine learning algorithm that is based on the Bayes' theorem, which states that the probability of a hypothesis (in this case, a class label) given some observed evidence (in this case, a feature vector) is equal to the probability of the evidence given the hypothesis, multiplied by the prior probability of the hypothesis, divided by the overall probability of the evidence.

- **K-Nearest Neighbors (KNN)** is a type of instance-based, or lazy, learning algorithm. It is a classification algorithm that is used to assign a class label to an unlabeled observation based on the class labels of the k-nearest observations to it in feature space.

- **A Decision Tree Classifier** is a type of supervised learning algorithm (having a pre-defined target variable) that is mostly used in classification problems. It works by recursively partitioning the dataset into subsets based on the values of the input features.

- **Support Vector Classifier (SVC)** is a type of supervised learning algorithm that can be used for classification and regression tasks. The main idea behind SVC is to find the best hyperplane (a decision boundary) that separates the different classes in the feature space. The best hyperplane is the one that maximizes the margin, which is the distance between the hyperplane and the closest data points from each class, also known as support vectors.

### Evaluation of algorithm

![image](https://github.com/user-attachments/assets/0ac05eb4-2db1-439d-a838-49c974c886dd)

![image](https://github.com/user-attachments/assets/b8fef941-2ba6-42b6-955c-f84ecc68dbaa)

- **Confusion Matrix**

A confusion matrix is a table that is used to define the performance of a classification algorithm. Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabeling one as another). The diagonal elements represent the number of points for which the predicted label is equal to the true label, while off-diagonal elements are those that are mislabeled by the classifier. It is a useful tool for understanding the performance of a classification algorithm, including the types of errors that the classifier is making.




**Confusion Matrix accuracy of Algorithms:**

- K-Nearest Neighbors (KNN): 0.77
- Neural Network: 0.82

We chose the Neural Network because the highest Confusion matrix accuracy.

### Hyperparameter Tuning

Hyperparameter tuning is the process of selecting the best set of hyperparameters for a machine learning model. Hyperparameters are parameters that are not learned from the data, but are set before training the model.

**Grid search:** is a technique used to tune the hyperparameters of a machine learning model. It is a systematic way of going through multiple combinations of parameter settings, cross-validating as it goes, and returning the best set of parameters that yield the highest performance for a given model. The technique involves specifying a set of values for each hyperparameter, creating a "grid" of all possible combinations of those values, and then training and evaluating a model for each combination of values. 

**Hyperparameters for KNN :**

- solvers: ['newton-cg', 'lbfgs', 'liblinear']
- penalty: ['l2']
- c_values: [100, 10, 1.0, 0.1, 0.01]

![image](https://github.com/user-attachments/assets/c6658f10-0bbb-416d-a532-1896bb5fa5e2)
![image](https://github.com/user-attachments/assets/1413469a-94aa-4793-b2d7-2782949c3098)

Confusion matrix accuracy for KNN nearist neighbours After Hyper Tuning(Complete dataset): 0.80 when K= 20

# Future Improvements

Experiment with more complex models like Gradient Boosting Machines (e.g., XGBoost, LightGBM) to improve prediction accuracy.
Implement ensemble methods to combine the strengths of multiple models.
Real-Time Data Ingestion and Prediction:

Allow users to upload their own data for real-time predictions through the Streamlit interface.
Enable real-time model updates when new data is introduced, ensuring the system remains accurate over time.
Model Explainability:

Incorporate model interpretability tools like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to help users understand how the model makes predictions.
Provide visual explanations of which features most strongly influence the prediction.
User Authentication:

Add a secure user authentication system for the Streamlit app, ensuring that only authorized users can access or upload data.
Optionally, create a user dashboard where individuals can track their health over time and see historical predictions.
Deployment Scaling:

Improve scalability by deploying the Streamlit app on cloud platforms like AWS, Google Cloud, or Heroku for faster load times and handling of larger datasets.
Optimize the deployment to support more concurrent users.
Mobile-Friendly Interface:

Optimize the Streamlit application for mobile browsers to provide a better experience on smartphones and tablets.
Database Integration:

Integrate a database to store patient data and prediction results securely.
Use this data for future research and model retraining, ensuring the system evolves with new insights.
Incorporating Additional Health Metrics:

Extend the model to include other health metrics such as blood pressure, cholesterol levels, and physical activity, making the prediction more comprehensive.
Long-Term Predictions and Health Recommendations:

Add a feature to predict long-term diabetes risk based on continuous data collection and provide personalized health recommendations based on the predictions.
Multilingual Support:

Add support for multiple languages to make the application more accessible to non-English speakers.
