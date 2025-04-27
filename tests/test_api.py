
import pytest
import requests
import os

@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", 
    reason="Skipping API test on GitHub Actions without running server"
)
def test_api_predict():
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
        pytest.fail(f"Failed to connect to API: {e}")

    assert response.status_code == 200
