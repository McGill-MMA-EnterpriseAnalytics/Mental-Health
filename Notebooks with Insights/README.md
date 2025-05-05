
# Notebooks with Insights

This folder contains full Jupyter Notebook versions of the project scripts and analysis outputs, including detailed markdown cells explaining the reasoning, methodologies, results, and key insights at each stage.

## Purpose
- Provide a transparent view of the project development process.
- Document the thought process, data exploration, model development, drift and fairness evaluation, and inference results.
- Facilitate easier review, understanding, and presentation of the project by including structured markdown commentary alongside executable code.

## Contents
- **Data Preprocessing and EDA** (`Data_Preprocessing_and_EDA.ipynb`, `Data_Preprocessing_and_EDA_v2.ipynb`):  
  Data cleaning, feature engineering, and exploratory analysis.
  
- **Drift Analysis** (`drift_analysis.ipynb`, `XGBoost_Optuna_ModelDrift.ipynb`):  
  Detection of data drift using EvidentlyAI, including HTML report generation and drift interpretation.
  XGBoost model tuning using Optuna, followed by performance evaluation and drift detection analysis based on ROC and classification metrics.

- **Fairness Analysis** (`Fairness_Anaysis.ipynb`):  
  Fairness evaluation across demographic groups using Fairlearn metrics.

- **Modeling** (`best_xgboost_with_tuning.py`, `Model_Tuning.ipynb` in notebook form):  
  XGBoost model training with hyperparameter tuning and evaluation.
  Full modeling process including data preprocessing, model selection, semi-supervised learning improvement, and hyperparameter tuning of Linear Regression, Random Forest, XGboost, and Catboost models, including performance evaluation and model saving.

- **Causal Inference** (`Causal_Inference.ipynb`):  
  Investigation of potential causal relationships between features and the treatment outcome.

- **Clustering** (`Clustering.ipynb`):  
  Unsupervised clustering to explore latent groupings in the survey population.

- **Survey Dataset** (`survey_encoded_v2.csv`):  
  The cleaned and encoded dataset used throughout the notebooks.

## Notes
- The logic and results are consistent with the main project scripts (`data_preprocessing/`, `modeling/`, `drift_fairness/`), but extended with markdown-based explanations for better interpretability.
- These notebooks are intended for detailed internal review, instructor evaluation, and team documentation purposes.

## Folder Structure Reminder
This folder is a complement to the main project structure, not a replacement. All production-ready code remains organized separately in modular Python scripts.

---

