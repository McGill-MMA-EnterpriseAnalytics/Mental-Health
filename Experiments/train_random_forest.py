"""
Script for training a Logistic Regression model to predict mental health treatment likelihood.

This script loads cleaned features and labels, applies preprocessing with one-hot encoding 
for categorical variables, builds a Logistic Regression model with class balancing, trains 
on the training set, and evaluates on the test set.

Steps:
1. Load cleaned features and labels.
2. Define categorical and numerical feature columns.
3. Perform train-test split with stratification.
4. Build preprocessing pipeline (OneHotEncode categorical, passthrough numerical).
5. Build modeling pipeline with Logistic Regression (with class_weight="balanced").
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
X = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/X_clean.csv")  # Features
y = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/y_clean.csv").squeeze()  # Target

# 2. Define Feature Columns
cat_cols = [
    'Gender', 'Country_grouped', 'self_employed', 'family_history',
    'work_interfere', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'leave',
    'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]
num_cols = ['Age']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4. Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),  # One-hot encode categorical features
    ('num', 'passthrough', num_cols)                             # Keep numerical features as-is
])

# 5. Full Modeling Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# 6. Train Model
pipeline.fit(X_train, y_train)

# 7. Evaluate Model
y_pred = pipeline.predict(X_test)
print("===== Logistic Regression Results =====")
print(classification_report(y_test, y_pred))
