
Experiments

This folder contains model experimentation scripts for the Mental Health Treatment Prediction project.
Each script trains a different machine learning model to evaluate and compare predictive performance.

Contents

- train_catboost.py	Train a CatBoost classifier on preprocessed survey data.
- train_label_propagation.py	Apply semi-supervised Label Propagation using partially labeled training data.
- train_logistic_regression.py	Train a Logistic Regression baseline model.
- train_random_forest.py	Train a Random Forest classifier to capture non-linear feature interactions.
- train_self_training.py	Implement a semi-supervised Self-Training model to boost performance.
- train_xgboost.py	Train an XGBoost model with hyperparameter tuning for final deployment.
Purpose

Model Benchmarking: Compare traditional, ensemble, and semi-supervised learning methods.
Final Model Selection: Choose the best-performing model (based on accuracy, F1-score, fairness).
Deployment Readiness: The best model will be integrated into the FastAPI service.
How to Run

Inside your activated virtual environment:

python train_catboost.py
python train_label_propagation.py
python train_logistic_regression.py
python train_random_forest.py
python train_self_training.py
python train_xgboost.py
