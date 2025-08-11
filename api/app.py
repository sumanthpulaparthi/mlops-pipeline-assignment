from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow.pyfunc
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd
import io
import os

from api.database import init_db, log_to_db, get_logs
from pipeline.retrain import retrain_model

# Load the best model from MLflow
MODEL_URI = "models:/BestCaliforniaHousingModel/Production"
model = mlflow.pyfunc.load_model("models/best_model")

app = FastAPI(title="California Housing Price Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

init_db()


class HouseFeatures(BaseModel):
    MedInc: float = Field(
        ...,
        ge=0,
        description=(
            "Median income in the neighborhood "
            "(in tens of thousands of dollars)"
        )
    )
    HouseAge: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Median age of houses in the "
            "neighborhood (0-100 years)"
        )
    )
    AveRooms: float = Field(
        ...,
        ge=0,
        description="Average number of rooms per household"
    )
    AveBedrms: float = Field(
        ...,
        ge=0,
        description="Average number of bedrooms per household"
    )
    Population: float = Field(
        ...,
        ge=0,
        description="Total population in the neighborhood"
    )
    AveOccup: float = Field(
        ...,
        ge=0,
        description="Average number of household members"
    )
    Latitude: float = Field(
        ...,
        ge=32.0,
        le=42.0,
        description=(
            "Geographical latitude of the neighborhood "
            "(California: 32-42)"
        )
    )
    Longitude: float = Field(
        ...,
        ge=-125.0,
        le=-114.0,
        description=(
            "Geographical longitude of the neighborhood "
            "(California: -125 to -114)"
        )
    )


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
        features.Longitude,
    ]]
    prediction = model.predict(data)
    prediction_value = float(prediction[0])
    log_to_db(features, prediction_value)
    return {"predicted_price": prediction_value}


@app.get("/logs")
def get_log(limit: int = 10):
    results = get_logs(limit)
    print(results)  # Remove or comment out in production
    return {"logs": results}


instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/new_data.csv", index=False)

        await retrain_model(df)

        return {"message": "Retraining triggered successfully."}
    return {"error": "Please upload a valid .csv file"}
