 Tests

This folder contains unit tests and integration tests for validating the core functionalities of the Mental Health Treatment Prediction project.

 **Files Overview**


**File	Description**
- test_preprocessing.py	Tests data cleaning and preprocessing functions to ensure correct transformations.
- test_model.py	Tests the trained model’s prediction pipeline and model loading.
- test_api.py	Integration test for the FastAPI endpoint to verify the API server returns valid predictions.

**Key Testing Aspects**

- Preprocessing Validation: Ensures that data cleaning utilities (e.g., gender standardization, age correction) work correctly.
- Model Prediction Check: Verifies that the trained classifier can generate outputs without error.
- API Health Check: Confirms that API /predict endpoint is live and returns predictions for sample inputs.

**How to Run Tests Locally**

- Install dependencies (if not already): pip install -r requirements.txt
- Make sure your FastAPI app is running locally (for API test).
- Run all tests: pytest tests/
