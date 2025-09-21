from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model 
model = joblib.load("model.joblib")

# FastAPI setup
app = FastAPI(title="Telco Churn Prediction API")

# Define input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str 

# Predict endpoint
@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.model_dump()])

    # Preprocess like in training
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", pd.NA)).fillna(0.0)

    # Predict
    pred_class = model.predict(df)[0]
    pred_proba = model.predict_proba(df)[0][1]

    return {"Churn_Prediction": int(pred_class), "Churn_Probability": float(pred_proba)}
