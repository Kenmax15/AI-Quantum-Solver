from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title="AI Quantum Solver API", version="1.0")

# Load model once at startup
model = joblib.load("rf_model_E0.pkl")


class XXZInput(BaseModel):
    Jxy: float
    Jz: float
    W: float
    h: List[float]


def build_features(payload: XXZInput) -> pd.DataFrame:
    h_array = np.array(payload.h, dtype=float)

    data = {
        "Jxy": payload.Jxy,
        "Jz": payload.Jz,
        "W": payload.W,
        "h_mean": np.mean(h_array),
        "h_std": np.std(h_array, ddof=1) if len(h_array) > 1 else 0.0,
        "h_max": np.max(h_array),
        "h_min": np.min(h_array),
        "h_abs_mean": np.mean(np.abs(h_array)),
        "h_var": np.var(h_array, ddof=1) if len(h_array) > 1 else 0.0,
    }

    for i, val in enumerate(h_array, start=1):
        data[f"h_{i}"] = val

    return pd.DataFrame([data])


@app.get("/")
def root():
    return {"message": "AI Quantum Solver API is running"}


@app.post("/predict")
def predict(payload: XXZInput):
    X = build_features(payload)
    prediction = model.predict(X)[0]

    return {
        "predicted_E0": float(prediction),
        "N": len(payload.h),
        "input_summary": {
            "Jxy": payload.Jxy,
            "Jz": payload.Jz,
            "W": payload.W,
        }
    }