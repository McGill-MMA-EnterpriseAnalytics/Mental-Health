# Mental Health Prediction from Online Survey
Team Members: Phoebe Gao, Yina Liang Li, Carol Wang, Yuri Xu, and Qian Zhao.

**Problem Statement**

Mental health issues, such as anxiety and depression, are increasingly prevalent, especially in the workplace. Early prediction and intervention can significantly improve individuals' well-being and productivity.
In this project, we aim to predict whether someone is likely to seek treatment for mental health conditions based on their lifestyle, work environment, and demographic information.

**Dataset Description**

Source: Kaggle – Mental Health in Tech Survey

Key Features:  
- Demographics: Age, Gender, Country 
- Workplace Factors: Remote work, Company size, Mental health benefits, Work interference
- Medical History: Family mental illness, Previous diagnosis, Past treatment
- Perceptions and Awareness: Comfort discussing mental health, Employer support
  
Target Variable:
- treatment: Whether the individual has sought treatment for a mental health condition (Yes/No)

 **Project Workflow**
 Data Collection (survey.csv)
    ↓
Data Preprocessing (cleaning_utils.py, preprocess.py)
    ↓
Train/Test Split (X_train_final.csv, X_test_final.csv)
    ↓
Model Training (Logistic Regression, Random Forest, XGBoost)
    ↓
Hyperparameter Tuning (Optuna)
    ↓
Model Evaluation (Accuracy, Fairness, Explainability)
   ↙                   ↘
Model Saving        Fairness & Drift Monitoring (Fairlearn + Evidently)
(classifier.pkl, preprocessor.pkl)
    ↓
API Deployment (FastAPI app.py)
    ↓
Dockerize Application (dockerfile, docker-compose.yml)
    ↓
CI/CD Integration (GitHub Actions)
