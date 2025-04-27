# tests/test_api.py

import pytest
import requests

def test_api_predict():

    # 假设你的 API 本地在 8000 端口跑
    url = "http://127.0.0.1:8000/predict"

    # 准备一个测试请求 payload
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


    # 发 POST 请求
    try:
        response = requests.post(url, json=payload)
    except Exception as e:
        pytest.fail(f"Failed to connect to API: {e}")

    # 检查 status_code
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    # 检查返回 JSON
    result = response.json()
    assert "prediction" in result, "Response JSON must contain 'prediction'"
