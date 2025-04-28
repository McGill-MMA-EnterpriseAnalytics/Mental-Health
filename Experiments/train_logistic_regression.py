"""
Script for training a Logistic Regression model to predict mental health treatment likelihood.

Steps:
1. Load cleaned features and labels.
2. Define categorical and numerical columns.
3. Split the data into training and testing sets with stratification.
4. Build a preprocessing pipeline (OneHotEncode categorical features, pass through numerical).
5. Build a modeling pipeline with preprocessing and Logistic Regression (with class balancing).
6. Train the model.
7. Evaluate model performance using classification metrics.

Author: Carol
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load Data
# X: Feature dataframe
# y: Target series (flattened)
X = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/X_clean.csv")
y = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/y_clean.csv").squeeze()

# 2. Define Feature Columns
# cat_cols: List of categorical feature names
# num_cols: List of numerical feature names
cat_cols = [
    'Gender', 'Country_grouped', 'self_employed', 'family_history',
    'work_interfere', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'leave',
    'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]
num_cols = ['Age']

# 3. Split Data
# Stratified train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4. Preprocessor
# ColumnTransformer to OneHotEncode categorical features and pass through numerical features
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

# 5. Model Pipeline
# Create a full pipeline: preprocessing + Logistic Regression with balanced class weights
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# 6. Train Model
# Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# 7. Evaluate Model
# Predict and print evaluation metrics on the test set
y_pred = pipeline.predict(X_test)
print("Logistic Regression:\n", classification_report(y_test, y_pred))
