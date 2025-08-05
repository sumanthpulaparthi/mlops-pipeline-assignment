from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
from prometheus_fastapi_instrumentator import Instrumentator
# Load the best model from MLflow
MODEL_URI = "models:/BestCaliforniaHousingModel/Production"
#model = mlflow.pyfunc.load_model(MODEL_URI)

from api.database import init_db, log_to_db,get_logs
''' For docker to load the best model '''
model = mlflow.pyfunc.load_model("models/best_model")

app = FastAPI(title="California Housing Price Predictor")

init_db()
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
    prediction_value = float(prediction[0])
    log_to_db(features,prediction_value)
    return {"predicted_price": prediction_value}


@app.get("/logs")
def get_log(limit: int = 10):
    #print("came here till this end point")
    #print(limit)
    results = get_logs(limit)
    print(results)
    return {"logs": results}


instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
