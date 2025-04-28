"""
Script for training a Label Propagation model on semi-supervised mental health survey data.

Steps:
1. Load preprocessed features and labels.
2. Define categorical and numerical columns.
3. Apply one-hot encoding to categorical features.
4. Mask 50% of the training labels to simulate semi-supervised learning.
5. Train a Label Propagation model using a k-Nearest Neighbors (k-NN) kernel.
6. Evaluate the model's performance on the test set.

Author: Carol
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.semi_supervised import LabelPropagation

# 1. Load Data
# X: Feature dataframe
# y: Target series (flattened to 1D)
X = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/X_clean.csv")
y = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/y_clean.csv").squeeze()

# 2. Feature Columns
# Define categorical and numerical feature columns
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
# Split the data into training and testing sets (80% train, 20% test),
# maintaining label distribution with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4. Define Preprocessor
# Use OneHotEncoder for categorical features and passthrough for numerical
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

# 5. Encode Features
# Fit on training data and transform both train and test sets
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

# 6. Create Semi-supervised Labels
# Randomly mask 50% of training labels (set to -1) for semi-supervised learning
rng = np.random.RandomState(42)
mask = rng.rand(len(y_train)) < 0.5
y_train_semi = y_train.copy()
y_train_semi[mask] = -1

# 7. Define Label Propagation Model
# Using k-Nearest Neighbors kernel and setting max iterations
