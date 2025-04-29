"""
Mental Health Prediction API using FastAPI

This API serves a trained ML model to predict whether a person is likely to seek mental health 
treatment based on survey data. It also provides SHAP-based interpretability.

Endpoints:
- GET "/"         : Health check.
- POST "/predict" : Get prediction and confidence score.
- POST "/explain" : Get SHAP feature importance explanation.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model components
preprocessor = joblib.load("modeling/preprocessor.pkl")
model = joblib.load("modeling/classifier.pkl")
explainer = joblib.load("modeling/explainer.pkl")

# Define request schema
class PredictRequest(BaseModel):
    Gender: str
    Country_grouped: str
    self_employed: str
    family_history: str
    work_interfere: str
    remote_work: str
    tech_company: str
    benefits: str
    care_options: str
    wellness_program: str
    seek_help: str
    leave: str
    mental_health_consequence: str
    phys_health_consequence: str
    coworkers: str
    supervisor: str
    mental_health_interview: str
    phys_health_interview: str
    mental_vs_physical: str
    obs_consequence: str
    Age: float

FEATURE_COLUMNS = [
    'Gender', 'Country_grouped', 'self_employed', 'family_history',
    'work_interfere', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'leave',
    'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence',
    'Age'
]

def prepare_data(request: PredictRequest) -> pd.DataFrame:
    """
    Convert the request payload into a pandas DataFrame.
    
    Args:
        request (PredictRequest): Input features from API request.

    Returns:
        pd.DataFrame: Single-row DataFrame for prediction.
    """
    data = [[getattr(request, col) for col in FEATURE_COLUMNS]]
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)

def preprocess_data(df: pd.DataFrame):
    """
    Apply preprocessing pipeline to input data.
    
    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        np.ndarray: Transformed feature matrix.
    """
    return preprocessor.transform(df)

def make_prediction(processed_data):
    """
    Predict treatment likelihood and return confidence.
    
    Args:
        processed_data (np.ndarray): Preprocessed data.

    Returns:
        Tuple[int, float]: Binary prediction and probability score.
    """
    prediction = model.predict(processed_data)[0]
    confidence = model.predict_proba(processed_data)[0][1]
    return int(prediction), round(float(confidence), 4)

def explain_prediction(processed_data):
    """
    Compute SHAP values for feature importance explanation.
    
    Args:
        processed_data (np.ndarray): Transformed input.

    Returns:
        Dict[str, float]: SHAP importance scores per feature.
    """
    shap_values = explainer.shap_values(processed_data)
    feature_importance = dict(zip(FEATURE_COLUMNS, shap_values[0].tolist()))
    return feature_importance

@app.get("/")
def health_check():
    """
    Health check endpoint to verify API status.

    Returns:
        dict: Status confirmation message.
    """
    return {"message": "✅ API is up and running!"}

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict if an individual is likely to seek treatment.

    Args:
        request (PredictRequest): Input survey data.

    Returns:
        dict: Prediction label and confidence score.
    """
    raw_data = prepare_data(request)
    processed_data = preprocess_data(raw_data)
    prediction, confidence = make_prediction(processed_data)
    return {
        "prediction": prediction,
        "confidence": confidence
    }

@app.post("/explain")
def explain(request: PredictRequest):
    """
    Return SHAP feature importance values for input.

    Args:
        request (PredictRequest): Input survey data.

    Returns:
        dict: SHAP values for each feature.
    """
    raw_data = prepare_data(request)
    processed_data = preprocess_data(raw_data)
    feature_importance = explain_prediction(processed_data)
    return {
        "shap_values": feature_importance
    }
