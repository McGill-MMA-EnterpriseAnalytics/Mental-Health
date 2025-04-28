"""
Unit test for verifying model prediction functionality.

This test:
- Loads the saved classifier and preprocessor.
- Preprocesses a sample input.
- Predicts output using the model.
- Checks prediction success and output shape.

Author: Your Name
Date: YYYY-MM-DD
"""

import pandas as pd
import joblib
import numpy as np  # ✅ Add this line
import pytest

def test_model_predict():
    """
    Test whether the trained model can make a prediction on sample input.
    """
    # Load model and preprocessor
    try:
        model = joblib.load('modeling/classifier.pkl')
        preprocessor = joblib.load('modeling/preprocessor.pkl')
    except Exception as e:
        pytest.fail(f"❌ Failed to load model or preprocessor: {e}")

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
    try:
        X_enc = preprocessor.transform(sample_data)
    except Exception as e:
        pytest.fail(f"❌ Preprocessing failed: {e}")

    # Predict
    try:
        y_pred = model.predict(X_enc)
    except Exception as e:
        pytest.fail(f"❌ Model prediction failed: {e}")

    # Assert output shape
    assert y_pred.shape[0] == 1, f"❌ Prediction output shape mismatch: {y_pred.shape}"

    # Optional: check that prediction is numeric (0 or 1)
    assert isinstance(y_pred[0], (int, np.integer, float, np.floating)), "❌ Prediction is not numeric."
