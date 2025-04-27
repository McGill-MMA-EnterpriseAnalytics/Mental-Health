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

@app.get("/")
def health_check():
    return {"message": "✅ API is up and running!"}

@app.post("/predict")
def predict(request: PredictRequest):
    # 组织数据
    data = [[
        request.Gender, request.Country_grouped, request.self_employed, request.family_history,
        request.work_interfere, request.remote_work, request.tech_company, request.benefits,
        request.care_options, request.wellness_program, request.seek_help, request.leave,
        request.mental_health_consequence, request.phys_health_consequence,
        request.coworkers, request.supervisor, request.mental_health_interview,
        request.phys_health_interview, request.mental_vs_physical, request.obs_consequence,
        request.Age
    ]]
    columns = [
        'Gender', 'Country_grouped', 'self_employed', 'family_history',
        'work_interfere', 'remote_work', 'tech_company', 'benefits',
        'care_options', 'wellness_program', 'seek_help', 'leave',
        'mental_health_consequence', 'phys_health_consequence',
        'coworkers', 'supervisor', 'mental_health_interview',
        'phys_health_interview', 'mental_vs_physical', 'obs_consequence',
        'Age'
    ]
    data_df = pd.DataFrame(data, columns=columns)

    # 特征工程
    data_processed = preprocessor.transform(data_df)

    # 预测
    prediction = model.predict(data_processed)[0]
    confidence = model.predict_proba(data_processed)[0][1]

    return {
        "prediction": int(prediction),
        "confidence": round(float(confidence), 4)
    }

@app.post("/explain")
def explain(request: PredictRequest):
    # 组织数据
    data = [[
        request.Gender, request.Country_grouped, request.self_employed, request.family_history,
        request.work_interfere, request.remote_work, request.tech_company, request.benefits,
        request.care_options, request.wellness_program, request.seek_help, request.leave,
        request.mental_health_consequence, request.phys_health_consequence,
        request.coworkers, request.supervisor, request.mental_health_interview,
        request.phys_health_interview, request.mental_vs_physical, request.obs_consequence,
        request.Age
    ]]
    columns = [
        'Gender', 'Country_grouped', 'self_employed', 'family_history',
        'work_interfere', 'remote_work', 'tech_company', 'benefits',
        'care_options', 'wellness_program', 'seek_help', 'leave',
        'mental_health_consequence', 'phys_health_consequence',
        'coworkers', 'supervisor', 'mental_health_interview',
        'phys_health_interview', 'mental_vs_physical', 'obs_consequence',
        'Age'
    ]
    data_df = pd.DataFrame(data, columns=columns)

    # 特征工程
    data_processed = preprocessor.transform(data_df)

    # 计算 SHAP
    shap_values = explainer.shap_values(data_processed)

    # 修改这里！！！
    feature_importance = dict(zip(columns, shap_values[0].tolist()))

    return {
        "shap_values": feature_importance
    }
