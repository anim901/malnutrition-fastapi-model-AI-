from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
from preprocess import preprocess_data
import numpy as np
import uvicorn

# Initialize FastAPI
app = FastAPI(debug=True)

# Load the TensorFlow model once at startup (faster inference)
MODEL_PATH = "./model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# Define request schema for automatic validation and docs
class PredictRequest(BaseModel):
    sex: str
    age: float
    birth_weight: float
    birth_length: float
    body_weight: float
    body_length: float
    asi_ekslusif: int

# Function to make predictions
def predict_stunting(normalized_data: np.ndarray):
    predictions = model.predict(normalized_data)
    return predictions

# Endpoint
@app.post("/predict")
async def predict(request: PredictRequest):
    # Extract data
    sex = request.sex
    age = request.age
    birth_weight = request.birth_weight
    birth_length = request.birth_length
    body_weight = request.body_weight
    body_length = request.body_length
    asi_ekslusif = request.asi_ekslusif

    # Preprocess input
    normalized_data = preprocess_data(
        sex, age, birth_weight, birth_length, body_weight, body_length, asi_ekslusif
    )

    # Predict
    predictions = predict_stunting(normalized_data)
    prediction_value = round(float(predictions[0][0]), 2)

    return {"prediction": prediction_value}


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
