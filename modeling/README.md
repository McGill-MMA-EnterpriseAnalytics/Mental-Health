
Modeling

This folder contains all scripts, saved models, and artifacts related to model training, hyperparameter tuning, and pseudo-labeling for the Mental Health Treatment Prediction project.

**Files Overview**

File	Description
- train_model.py	Main script for training the XGBoost model and saving the final pipeline.
- pseudo_labeling.py	Implements a semi-supervised learning approach using pseudo-labeling.
- best_xgb_params.json	Stores the best hyperparameters found during Optuna tuning.
- best_xgb_pipeline.pkl	Complete trained model pipeline, including preprocessing steps.
- classifier.pkl	Trained model object (classifier only).
- preprocessor.pkl	Preprocessing pipeline used before model training (encoding, scaling).
- explainer.pkl	SHAP explainer object for model interpretability.
- X_train_final.csv	Final preprocessed training feature set.
- X_test_final.csv	Final preprocessed testing feature set.
- y_train_final.csv	Final preprocessed training labels.
- y_test_final.csv	Final preprocessed testing labels.

 **Key Components**

- Training Pipeline: Data preprocessing ➔ model fitting ➔ model serialization.
- Hyperparameter Tuning: Optimized XGBoost parameters using Optuna for maximum predictive performance.
- Semi-Supervised Learning: Leveraged pseudo-labeling to enrich training data when labels were partially missing.
- Explainability: Built SHAP-based interpretability pipeline (explainer.pkl) to analyze feature contributions.

**Notes**

- Models and preprocessors were serialized using joblib.
- Artifacts in this folder are essential for deployment (used later by the API).
- No manual modification is recommended unless retraining is needed.
