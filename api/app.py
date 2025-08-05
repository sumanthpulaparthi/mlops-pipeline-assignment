from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

# Load the best model from MLflow
MODEL_URI = "models:/BestCaliforniaHousingModel/Production"
#model = mlflow.pyfunc.load_model(MODEL_URI)
''' For docker to load the best model '''
model = mlflow.pyfunc.load_model("models/best_model")

app = FastAPI(title="California Housing Price Predictor")


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def read_root():
    return {"message": "California Housing Prediction API is running"}

@app.post("/predict")
def predict(features: HouseFeatures):
    data = [[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude
    ]]
    prediction = model.predict(data)
    return {"predicted_price": float(prediction[0])}

