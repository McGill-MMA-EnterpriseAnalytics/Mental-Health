## Inference Module

This folder contains the API deployment code for the Mental Health Prediction project.  
It enables real-time inference through a FastAPI server.

---

##  Files

| File | Description |
|:---|:---|
| `api.py` | FastAPI application providing endpoints for mental health prediction |
| `__pycache__/` | Auto-generated cache files (can be ignored) |

---

##  How It Works

- Loads the trained model (`classifier.pkl`) and preprocessor (`preprocessor.pkl`)
- Accepts user input via API request
- Returns prediction results (e.g., treatment risk)

---

##  How to Run Locally

1. Install dependencies:

    ```bash
    pip install fastapi uvicorn joblib scikit-learn
    ```

2. Run the FastAPI server:

    ```bash
    uvicorn api:app --reload
    ```

3. Access the interactive API documentation at:

    ```
    http://127.0.0.1:8000/docs
    ```

---

##  Available API Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/predict` | POST | Accepts user information and returns mental health risk prediction |

Example payload for `/predict`:

```json
{
  "Gender": "Female",
  "Country_grouped": "Canada",
  "self_employed": "No",
  "family_history": "Yes",
  "work_interfere": "Often",
  "remote_work": "Yes",
  "tech_company": "Yes",
  "benefits": "Yes",
  "care_options": "Not sure",
  "wellness_program": "No",
  "seek_help": "Yes",
  "leave": "Somewhat difficult",
  "mental_health_consequence": "Yes",
  "phys_health_consequence": "No",
  "coworkers": "Some of them"
}

