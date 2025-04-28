"""
Full training pipeline for mental health treatment prediction using XGBoost,
including:
- Data loading and preprocessing
- Semi-supervised learning with pseudo-labeling
- Hyperparameter tuning via RandomizedSearchCV
- Model and artifact saving (pipeline, classifier, explainer)
- MLflow experiment tracking

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from pseudo_labeling import apply_pseudo_labeling  

# =======================
# 1. Load Data
# =======================
X = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/X_clean.csv")
y = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/y_clean.csv")

# Ensure y is a Series
if isinstance(y, pd.DataFrame):
    y = y.squeeze()

# Define categorical and numerical columns
cat_cols = [
    'Gender', 'Country_grouped', 'self_employed', 'family_history',
    'work_interfere', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'leave',
    'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]
num_cols = ['Age']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Further split training into labeled and unlabeled
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
    X_train, y_train, stratify=y_train, test_size=0.2, random_state=42
)

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

# =======================
# 2. Pseudo-Labeling
# =======================
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

X_aug_xgb, y_aug_xgb = apply_pseudo_labeling(
    xgb,
    X_labeled,
    y_labeled,
    X_unlabeled,
    preprocessor=preprocessor,
    threshold=0.95
)

# Final split after pseudo-labeling
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_aug_xgb, y_aug_xgb, stratify=y_aug_xgb, test_size=0.2, random_state=42
)

# =======================
# 3. Model Training & Tuning
# =======================
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

param_dist = {
    'classifier__n_estimators': [300, 400, 500, 600, 800],
    'classifier__max_depth': [3, 4, 5, 6],
    'classifier__learning_rate': [0.01, 0.03, 0.05, 0.07],
    'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'classifier__gamma': [0, 0.05, 0.1],
    'classifier__reg_alpha': [0, 0.01, 0.1],
    'classifier__reg_lambda': [1, 1.5, 2.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit model
random_search.fit(X_train_final, y_train_final)

# Best pipeline
best_xgb_pipeline = random_search.best_estimator_

# Test set predictions
y_pred = best_xgb_pipeline.predict(X_test_final)

# =======================
# 4. Evaluation
# =======================
print("\n===== Best Parameters =====")
print(random_search.best_params_)

print("\n===== Best Cross-Validation Accuracy =====")
print(random_search.best_score_)

print("\n===== Test Set Classification Report =====")
print(classification_report(y_test_final, y_pred))

# =======================
# 5. Save Artifacts
# =======================
save_dir = "/Users/qianzhao/Desktop/Enterprise/formal version/modeling"
os.makedirs(save_dir, exist_ok=True)

# Save pipeline, preprocessor, classifier
joblib.dump(best_xgb_pipeline, os.path.join(save_dir, "best_xgb_pipeline.pkl"))
joblib.dump(best_xgb_pipeline.named_steps['preprocessor'], os.path.join(save_dir, "preprocessor.pkl"))
joblib.dump(best_xgb_pipeline.named_steps['classifier'], os.path.join(save_dir, "classifier.pkl"))

# Save best parameters
with open(os.path.join(save_dir, "best_xgb_params.json"), "w") as f:
    json.dump(random_search.best_params_, f, indent=4)

print(f"✅ Pipeline, Preprocessor, Classifier saved to {save_dir}")

# Save SHAP explainer
classifier = best_xgb_pipeline.named_steps['classifier']
explainer = shap.TreeExplainer(classifier)
joblib.dump(explainer, os.path.join(save_dir, "explainer.pkl"))

print(f"✅ SHAP Explainer saved to {save_dir}")

# Save train/test data for deployment
X_train_final.to_csv(os.path.join(save_dir, "X_train_final.csv"), index=False)
X_test_final.to_csv(os.path.join(save_dir, "X_test_final.csv"), index=False)
y_train_final.to_csv(os.path.join(save_dir, "y_train_final.csv"), index=False)
y_test_final.to_csv(os.path.join(save_dir, "y_test_final.csv"), index=False)

# =======================
# 6. MLflow Tracking
# =======================
mlflow.set_experiment("mental_health_xgb_experiment")

with mlflow.start_run():
    mlflow.log_params(random_search.best_params_)
    mlflow.log_metric("cross_val_accuracy", random_search.best_score_)
    test_acc = (y_pred == y_test_final).mean()  # Test set accuracy
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.sklearn.log_model(best_xgb_pipeline, artifact_path="model")
    print("✅ MLflow logging completed.")
