Nearest Earth Objects (NEO) Classification

Overview

This project aims to classify Nearest Earth Objects (NEOs) as hazardous or non-hazardous using various machine learning models. The dataset contains information about asteroids and their orbital characteristics.

The project follows a structured workflow:

Data Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Model Training and Evaluation

Hyperparameter Tuning

Dataset

The dataset used in this project is nearest-earth-objects(1910-2024).csv, containing historical data on NEOs.

Target Variable:

is_hazardous: Indicates whether an object is hazardous (1) or non-hazardous (0).

Key Features:

Various physical and orbital parameters of asteroids.

orbiting_body and name were removed as they are not predictive features.


Data Preprocessing

Load the dataset using Pandas.

Handle missing values by dropping NaN entries.

Convert is_hazardous to numerical format (0 or 1).

Perform feature selection by removing non-relevant columns.

Normalize numerical features using StandardScaler.

Machine Learning Models

The following classification models were trained and evaluated:

Logistic Regression

Random Forest Classifier

K-Nearest Neighbors (KNN)

XGBoost Classifier

Each model was trained using a 70-30 train-test split. Feature scaling was applied using StandardScaler.

Model Evaluation

The models were evaluated using the following metrics:

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

AUC-ROC Curve

Hyperparameter Tuning

To optimize the Random Forest Classifier, GridSearchCV was used to find the best parameters:

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
print("Best Random Forest Parameters:", rf_grid.best_params_)

Results & Best Model Selection

The Random Forest Classifier achieved the best performance with an accuracy of ~91%.

AUC-ROC analysis showed that Random Forest was the best model in differentiating between hazardous and non-hazardous objects.

XGBoost also performed well
