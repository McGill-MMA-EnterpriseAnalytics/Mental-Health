"""
Integration test for the FastAPI /predict endpoint.

This test:
- Sends a sample payload to the /predict endpoint.
- Verifies that the response status code is 200 (OK).
- Can be skipped on GitHub Actions where no server is running.

"""

import pytest
import requests
import os

@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping API test on GitHub Actions without running server"
)
def test_api_predict():
    """
    Test the /predict endpoint with a sample payload.
    """
    url = "http://127.0.0.1:8000/predict"

    payload = {
        "Gender": "Male",
        "Country_grouped": "Canada",
        "self_employed": "No",
        "family_history": "Yes",
        "work_interfere": "Often",
        "remote_work": "No",
        "tech_company": "Yes",
        "benefits": "Yes",
        "care_options": "Yes",
        "wellness_program": "No",
        "seek_help": "No",
        "leave": "Somewhat easy",
        "mental_health_consequence": "No",
        "phys_health_consequence": "No",
        "coworkers": "Some of them",
        "supervisor": "Yes",
        "mental_health_interview": "No",
        "phys_health_interview": "No",
        "mental_vs_physical": "Mental health",
        "obs_consequence": "No",
        "Age": 30
    }

    try:
        response = requests.post(url, json=payload)
    except Exception as e:
        pytest.fail(f"❌ Failed to connect to API: {e}")

    # Check that the server responds correctly
    assert response.status_code == 200, f"❌ Expected 200 OK, got {response.status_code}"

    # Optional: Check if the response is valid JSON
    try:
        response_json = response.json()
    except ValueError:
        pytest.fail("❌ Response is not valid JSON.")

    # Optional: Check required fields exist in response
    assert "prediction" in response_json, "❌ 'prediction' field missing in response."
    assert "confidence" in response_json, "❌ 'confidence' field missing in response."
