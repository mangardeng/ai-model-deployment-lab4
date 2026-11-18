
import joblib
import numpy as np

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "iris_pipeline.joblib"

model = joblib.load(MODEL_PATH)

MODEL_VERSION = "iris_v1"

SPECIES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

app = FastAPI(
    title="Iris Flower Classification API",
    description="Machine learning API for Iris flower classification",
    version="1.0"
)

class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)

class Prediction(BaseModel):
    predicted_class: str
    predicted_class_id: int
    confidence: float
    model_version: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION
    }

@app.post("/v1/predict", response_model=Prediction)
def predict(flower: FlowerInput):
    try:
        X = np.array([[
            flower.sepal_length,
            flower.sepal_width,
            flower.petal_length,
            flower.petal_width
        ]], dtype=np.float32)

        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        confidence = float(probabilities[prediction])

        return Prediction(
            predicted_class=SPECIES[prediction],
            predicted_class_id=prediction,
            confidence=round(confidence, 4),
            model_version=MODEL_VERSION
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from error
