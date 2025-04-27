# tests/test_model.py

import pandas as pd
import joblib
import pytest

def test_model_predict():
    # Load model and preprocessor
    model = joblib.load('/Users/qianzhao/Desktop/Enterprise/formal version/modeling/classifier.pkl')
    preprocessor = joblib.load('/Users/qianzhao/Desktop/Enterprise/formal version/modeling/preprocessor.pkl')

    # Create a small sample input
    sample_data = pd.DataFrame({
    'Gender': ['Male'],
    'Country_grouped': ['Canada'],
    'self_employed': ['No'],
    'family_history': ['Yes'],
    'work_interfere': ['Often'],
    'remote_work': ['No'],
    'tech_company': ['Yes'],
    'benefits': ['Yes'],
    'care_options': ['Yes'],
    'wellness_program': ['No'],
    'seek_help': ['No'],
    'leave': ['Somewhat easy'],
    'mental_health_consequence': ['No'],
    'phys_health_consequence': ['No'],
    'coworkers': ['Some of them'],
    'supervisor': ['Yes'],
    'mental_health_interview': ['No'],
    'phys_health_interview': ['No'],
    'mental_vs_physical': ['Mental health'],
    'obs_consequence': ['No'],
    'Age': [30]
})


    # Preprocessing
    X_enc = preprocessor.transform(sample_data)

    # Predict
    try:
        y_pred = model.predict(X_enc)
    except Exception as e:
        pytest.fail(f"Model prediction failed: {e}")

    # Assert shape
    assert y_pred.shape[0] == 1
