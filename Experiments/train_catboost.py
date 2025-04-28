"""
Script for training a CatBoostClassifier to predict mental health treatment likelihood.

Steps:
1. Load preprocessed feature and label datasets.
2. Define categorical and numerical feature columns.
3. Split the data into training and testing sets.
4. Train a CatBoost model directly on the raw categorical features without one-hot encoding.
5. Evaluate the model using classification metrics.

Author: Carol Wang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

# 1. Load data
# X: Features dataframe
# y: Target series
X = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/X_clean.csv")
y = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/y_clean.csv").squeeze()

# 2. Define feature columns
# cat_cols: List of categorical feature column names
# num_cols: List of numerical feature column names
cat_cols = [
    'Gender', 'Country_grouped', 'self_employed', 'family_history',
    'work_interfere', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'leave',
    'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]
num_cols = ['Age']

# 3. Split data
# Split features and target into training and testing sets (80% train, 20% test),
# preserving the distribution of the target variable with stratification.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4. Model training
# Initialize CatBoostClassifier:
# - CatBoost can natively handle categorical features without needing one-hot encoding.
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    eval_metric='AUC',
    random_seed=42,
    verbose=False  # Suppress training output
)

# Train the model, specifying categorical features by their column names.
model.fit(
    X_train, y_train,
    cat_features=cat_cols,
    eval_set=(X_test, y_test)
)

# 5. Model evaluation
# Predict labels on the test set
y_pred = model.predict(X_test)

# Print classification metrics
print("CatBoost:\n", classification_report(y_test, y_pred))
