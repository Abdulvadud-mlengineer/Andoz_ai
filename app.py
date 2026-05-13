from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()



# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# загрузка модели
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    df = pd.read_excel(file.file)

    # feature engineering
    df["tax_ratio"] = df["tax_paid"] / df["income"]
    df["activity_per_employee"] = df["transactions"] / df["employees"]
    df["income_per_transaction"] = df["income"] / (df["transactions"] + 1)

    X = df[features]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df["prediction"] = preds
    df["risk_score"] = probs * 100

    return {
        "results": df[["prediction", "risk_score"]].to_dict(orient="records")
    }