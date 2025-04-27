# tests/test_preprocessing.py

import pandas as pd
import joblib
import pytest

def test_preprocessor_fit_transform():
    # Load preprocessor
    preprocessor = joblib.load('modeling/preprocessor.pkl')

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
    'supervisor': ['Yes'],                # 补充字段
    'mental_health_interview': ['No'],     # 补充字段
    'phys_health_interview': ['No'],       # 补充字段
    'mental_vs_physical': ['Mental health'],  # 补充字段
    'obs_consequence': ['No'],             # 补充字段
    'Age': [30]                            # 补充字段
})


    # Try to transform
    try:
        transformed = preprocessor.transform(sample_data)
    except Exception as e:
        pytest.fail(f"Preprocessing failed: {e}")

    # Assert transformed output is 2D
    assert len(transformed.shape) == 2
    assert transformed.shape[0] == 1

