"""
Unit test for verifying preprocessing pipeline functionality.

This test:
- Loads the saved preprocessor.
- Applies it to a sample input.
- Ensures the transformed output is 2D and has expected shape.

"""

import pandas as pd
import joblib
import pytest

def test_preprocessor_fit_transform():
    """
    Test whether the saved preprocessor can successfully transform a sample input.
    """
    # Load preprocessor
    try:
        preprocessor = joblib.load('modeling/preprocessor.pkl')
    except Exception as e:
        pytest.fail(f"❌ Failed to load preprocessor: {e}")

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

    # Try to transform
    try:
        transformed = preprocessor.transform(sample_data)
    except Exception as e:
        pytest.fail(f"❌ Preprocessing transform failed: {e}")

    # Assert transformed output is 2D
    assert len(transformed.shape) == 2, f"❌ Transformed output should be 2D, got shape {transformed.shape}"
    assert transformed.shape[0] == 1, f"❌ Transformed output should have 1 row, got {transformed.shape[0]}"
