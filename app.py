from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
from database import SessionLocal, FraudPrediction



MODEL_PATH = "model_pipeline.pkl"


# -------------------------
# Input Schema (Raw Features Only)
# -------------------------
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


# -------------------------
# Load Model on Startup
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    print("✔ Model Loaded")
    yield


app = FastAPI(title="Fraud API", lifespan=lifespan)


# -------------------------
# Enable CORS (required for UI)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow UI from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Feature Engineering (matches train.py)
# -------------------------
def add_engineered_features(df):

    # These engineered features must match EXACTLY train.py logic
    df["Amount_Scaled"] = df["Amount"]     # scaling happens inside pipeline
    df["log_amount"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] // 3600) % 24

    # For single prediction we cannot compute rolling stats → use safe defaults
    df["amount_zscore"] = 0
    df["rolling_amount_mean"] = df["Amount"]
    df["rolling_count"] = 0
    df["Amount_Outlier_Flag"] = (df["Amount"] > df["Amount"].quantile(0.99)).astype(int)

    return df


# -------------------------
# Predict Endpoint
# -------------------------
@app.post("/predict")
def predict(data: Transaction):

    try:
        df = pd.DataFrame([data.dict()])

        # add all engineered features required by the trained model pipeline
        df = add_engineered_features(df)

        probability = model.predict_proba(df)[0][1]
        decision = int(probability >= 0.5)

        # -------------------------
        # SAVE TO DATABASE
        # -------------------------
        from database import SessionLocal, FraudPrediction

        db = SessionLocal()
        record = FraudPrediction(
            time=data.Time,
            amount=data.Amount,
            probability=float(probability),
            fraud=decision
        )
        db.add(record)
        db.commit()
        db.close()
        # -------------------------

        return {
            "fraud_probability": float(probability),
            "fraud": decision
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
# -------------------------
# Run server manually
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
