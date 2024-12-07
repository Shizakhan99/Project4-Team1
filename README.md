**Stroke Prediction Model**

This project aims to build, train, and evaluate various machine learning models to predict the likelihood of a person having a stroke based on their health data. The models include Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, Random Forest, Naive Bayes, and a Deep Neural Network (DNN) with hyperparameter tuning.

**Table of Contents**

Overview
Dataset
Modeling Process
Results
Requirements
How to Run
License

**Overview**

This project focuses on the classification of stroke prediction based on a set of health-related attributes, such as age, BMI, hypertension, and others. We explore different resampling techniques (e.g., undersampling and oversampling) to handle class imbalance and evaluate multiple machine learning algorithms to optimize prediction accuracy.

**Dataset**

The dataset used in this project is the Stroke Prediction dataset. It contains several health attributes of individuals, including:

Age
Hypertension
Heart Disease
Ever Married
Work Type
Residence Type
BMI
Smoking Status
Stroke (target variable)

The dataset is loaded from a CSV file and cleaned by removing rows with missing BMI values.

**Data Preprocessing**

Missing Data Handling: Rows with missing BMI values are dropped.
Categorical Data Encoding: Categorical variables such as work_type, residence_type, and smoking_status are converted into numeric features using one-hot encoding.

Resampling: The dataset is resampled using both undersampling and oversampling techniques (using RandomUnderSampler and SMOTE).

**Modeling Process**

The following models are trained and evaluated on the dataset:

Logistic Regression:
A baseline model for classification using logistic regression.

K-Nearest Neighbors (KNN):
Classification model using KNN with 3 neighbors.

Decision Tree Classifier:
A decision tree model to classify based on the most important features.

Random Forest Classifier:
An ensemble method using Random Forest for classification.

Naive Bayes:
A probabilistic model used for classification, evaluated with hyperparameter tuning using GridSearchCV.

Deep Neural Network (DNN):
A neural network model built using TensorFlow and Keras, optimized with KerasTuner for hyperparameter tuning.

**Preprocessing Steps:**

Train-Test Split: The dataset is split into 85% training and 15% testing data using train_test_split from scikit-learn.
Scaling: Feature scaling is done using StandardScaler to normalize the data before training.

**Model Evaluation:**

All models are evaluated based on accuracy, confusion matrix, and classification report.

The confusion matrix shows the true positives, true negatives, false positives, and false negatives for model performance.

The classification report provides precision, recall, F1-score, and accuracy metrics.

Hyperparameter Tuning (Naive Bayes & DNN):
Naive Bayes: Hyperparameter tuning is done using GridSearchCV for optimal var_smoothing value.

DNN: The neural network is optimized using KerasTuner to find the best configuration of hyperparameters, including the number of layers, units per layer, and activation functions.

**Results**

After evaluating all models, performance metrics such as accuracy, R-squared score, and confusion matrices are reported for each model. The DNN model performs the best after hyperparameter tuning, with detailed evaluation through confusion matrices and accuracy scores.

**Example Outputs**

Accuracy Score: The models' accuracy scores are displayed for both the initial and optimized models.
Confusion Matrix: A confusion matrix is plotted for visual performance comparison of the models.
