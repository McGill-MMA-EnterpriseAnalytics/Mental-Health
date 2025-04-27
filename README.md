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

## Project Structure Overview

```mermaid
flowchart TD
    A1[mental-health-prediction/] --> B1[data_preprocessing/]
    B1 --> B2[preprocess.py]

    A1 --> C1[modeling/]
    C1 --> C2[train_model.py]

    A1 --> D1[inference/]
    D1 --> D2[api.py]

    A1 --> E1[drift_fairness/]
    E1 --> E2[monitor_drift.py]
    E1 --> E3[check_fairness.py]

    A1 --> F1[tests/]
    F1 --> F2[test_preprocessing.py]
    F1 --> F3[test_model.py]

    A1 --> G1[notebooks/]
    G1 --> G2[explainability.ipynb]

    A1 --> H[Dockerfile]
    A1 --> I[requirements.txt]
    A1 --> J[.pre-commit-config.yaml]
    A1 --> K[.github/workflows/ci.yml]
    A1 --> L[mlruns/]
    A1 --> M[README.md]
```


**Quick Start Guide**

- Install Requirements: pip install -r requirements.txt

- Local Development
1. Preprocess Data: python data_preprocessing/preprocess.py
2. Train Model: python modeling/train_model.py
3. Run Fairness and Drift Monitoring: python drift_fairness/check_fairness.py
                                      python drift_fairness/monitor_drift.py
4. Launch FastAPI App: cd inference ,uvicorn api:app --reload, The API will be available at: http://127.0.0.1:8000/docs

- Docker Deployment
1. Build Docker Image:docker build -t mental-health-api .
2. Run Docker Container: docker run -p 8000:8000 mental-health-api

-  Continuous Integration (CI)  
Every push to the main branch triggers the GitHub Actions workflow to:

  1.Automatically build the Docker image
  
  2. Run unit tests in /tests/
     
  3. Validate the FastAPI server  
You can find the workflow file in .github/workflows/ci.yml.

**Key Results**

- Achieved 90% overall test accuracy on the mental health treatment prediction task.
- Implemented fairness auditing with demographic parity and equalized odds difference metrics.
- Deployed Dockerized FastAPI API and integrated CI/CD with GitHub Actions.
- Set up drift detection using Evidently AI for continuous monitoring.


**Project End-to-End Pipeline**

```mermaid
flowchart TD
    A[Data Collection 📋\nSurvey CSV data collection] --> B[Data Preprocessing 🛠️\nCleaning and encoding data]
    B --> C[Train-Test Split ✂️\nSplit into training and testing sets]
    C --> D[Model Training 🤖\nTrain Logistic Regression, Random Forest, XGBoost]
    D --> E[Hyperparameter Tuning 🎯\nOptimize with Optuna study]
    E --> F[Model Evaluation 📊\nEvaluate Accuracy, Fairness, Explainability]
    F --> G[Model Saving 💾\nSave classifier and preprocessor]
    G --> H[API Deployment 🚀\nDeploy with FastAPI app]
    H --> I[Containerization 🐳\nBuild Docker image and compose]
    I --> J[CI-CD Automation ⚙️\nGitHub Actions workflow for CI-CD]
    F --> K[Monitoring 🧭\nFairness and Drift monitoring scripts]
```


