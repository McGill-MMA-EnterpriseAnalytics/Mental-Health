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
y = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_
