from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class SoilInput(BaseModel):
    air_temperature: float
    soil_temperature: float
    humidity: float
    moisture: float
    nitrogen: float
    phosphorous: float
    potassium: float

@app.post("/predict")
def predict_soil_health(data: SoilInput):
    # Convert to 2D array for prediction
    features = np.array([[data.air_temperature, data.soil_temperature,
                          data.humidity, data.moisture, data.nitrogen,
                          data.phosphorous, data.potassium]])
    
    prediction = model.predict(features)[0]
    label = "Poor Soil Health" if prediction == 0 else "Moderate Soil Health"
    
    return {"prediction": int(prediction), "label": label}
