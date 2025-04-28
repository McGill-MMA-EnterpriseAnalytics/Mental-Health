"""
Script for training a Self-Training model with Logistic Regression as the base estimator
to predict mental health treatment likelihood using semi-supervised learning.

This script loads cleaned features and labels, applies preprocessing with one-hot encoding 
for categorical variables, masks a portion of training labels to simulate semi-supervised learning, 
trains a Self-Training classifier, and evaluates performance on the test set.

Steps:
1. Load cleaned features and labels.
2. Define categorical and numerical feature columns.
3. Perform train-test split with stratification.
4. Build a preprocessing pipeline (OneHotEncode categorical, passthrough numerical).
5. Encode features.
6. Mask 50% of training labels for semi-supervised learning.
7. Build Self-Training model using Logistic Regression as base learner.
8. Train the model.
9. Predict and evaluate on the test set.

Author: Carol
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
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

# 5. Encode Features
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

# 6. Create Semi-supervised Labels
# Randomly mask 50% of the training labels (-1 means unlabeled)
rng = np.random.RandomState(42)
mask = rng.rand(len(y_train)) < 0.5
y_train_semi = y_train.copy()
y_train_semi[mask] = -1

# 7. Define Self-Training Model
self_train = SelfTrainingClassifier(
    base_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
    threshold=0.8  # Confidence threshold for pseudo-labeling
)

# 8. Train Self-Training Model
self_train.fit(X_train_enc, y_train_semi)

# 9. Predict on Test Set
y_pred = self_train.predict(X_test_enc)

# 10. Evaluation
print("===== Self-Training Logistic Regression Results =====")
print(classification_report(y_test, y_pred))
