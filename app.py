# =========================
# FASTAPI APP
# =========================
from fastapi import FastAPI
import joblib
import pandas as pd

# Create app
app = FastAPI()

# Load trained model
model = joblib.load('model.pkl')


# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running"}


# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        return {
            "prediction": int(prediction),
            "meaning": "Survived" if prediction == 1 else "Did Not Survive"
        }

    except Exception as e:
        return {"error": str(e)}