from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 初始化 FastAPI app
app = FastAPI()

# 加载模型、预处理器、explainer
preprocessor = joblib.load("modeling/preprocessor.pkl")
model = joblib.load("modeling/classifier.pkl")
explainer = joblib.load("modeling/explainer.pkl")

# 定义请求体格式
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

# 公共变量
FEATURE_COLUMNS = [
    'Gender', 'Country_grouped', 'self_employed', 'family_history',
    'work_interfere', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'leave',
    'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence',
    'Age'
]

# 封装 pipeline functions
def prepare_data(request: PredictRequest) -> pd.DataFrame:
    data = [[getattr(request, col) for col in FEATURE_COLUMNS]]
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)

def preprocess_data(df: pd.DataFrame):
    return preprocessor.transform(df)

def make_prediction(processed_data):
    prediction = model.predict(processed_data)[0]
    confidence = model.predict_proba(processed_data)[0][1]
    return int(prediction), round(float(confidence), 4)

def explain_prediction(processed_data):
    shap_values = explainer.shap_values(processed_data)
    feature_importance = dict(zip(FEATURE_COLUMNS, shap_values[0].tolist()))
    return feature_importance

# 路由
@app.get("/")
def health_check():
    return {"message": "✅ API is up and running!"}

@app.post("/predict")
def predict(request: PredictRequest):
    raw_data = prepare_data(request)
    processed_data = preprocess_data(raw_data)
    prediction, confidence = make_prediction(processed_data)
    return {
        "prediction": prediction,
        "confidence": confidence
    }

@app.post("/explain")
def explain(request: PredictRequest):
    raw_data = prepare_data(request)
    processed_data = preprocess_data(raw_data)
    feature_importance = explain_prediction(processed_data)
    return {
        "shap_values": feature_importance
    }
