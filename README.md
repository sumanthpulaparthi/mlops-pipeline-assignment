
---


- Load and preprocess the **California Housing Dataset**
- Organize the project with a clean folder structure
- Track dataset versions using **DVC**
- Maintain code and data versioning with **Git**

---
 Folder Structure:

.
├── data
│   └── raw
│       ├── housing.csv
│       └── housing.csv.dvc
├── logs
├── models
├── notebooks
├── README.md
├── requirements.txt
└── src
    └── load_data.py


-----


1. **GitHub Repository Setup**
   - Initialized Git repository
   - Structured project directories for data, code, and notebooks

2. **Data Loading and Saving**
   - Loaded the California Housing dataset using `scikit-learn`
   - Saved the raw dataset to `data/raw/housing.csv`

3. **Dataset Tracking with DVC**
   - Initialized DVC (`dvc init`)
   - Added dataset to DVC tracking:  
     `dvc add data/raw/housing.csv`
   - Committed `.dvc` files to Git for version control

4. **.gitignore Fixes**
   - Ensured `.dvc/` and `*.dvc` files are **not ignored** by Git
   - Modified `.gitignore` to allow:
     ```gitignore
     !.dvc/
     !*.dvc
     ```

---

##  Requirements

Install required libraries:
```bash
pip install -r requirements.txt

---

##  Requirements

# 🧠 Part 2 – Model Development & Experiment Tracking

This stage focuses on training regression models and tracking experiments using MLflow on the California Housing Dataset.

---

## 🎯 Objectives

- Train two models: `LinearRegression` and `DecisionTreeRegressor`
- Log parameters, metrics (MSE), and models with **MLflow**
- Save trained models to the `models/` directory

---

## 📁 Key Files & Structure

# 🧠 Part 2 – Model Development & Experiment Tracking

This stage focuses on training regression models and tracking experiments using MLflow on the California Housing Dataset.

---

## 🎯 Objectives

- Train two models: `LinearRegression` and `DecisionTreeRegressor`
- Log parameters, metrics (MSE), and models with **MLflow**
- Save trained models to the `models/` directory

---

## 📁 Key Files & Structure

ml-pipeline-project/
├── data/raw/housing.csv # Input dataset (DVC tracked)
├── models/ # Output models
├── src/
│ ├── load_data.py # Loads dataset (used in Part 1)
│ └── train.py # ✅ Trains & tracks models
├── requirements.txt
├── mlruns/ # Auto-generated MLflow tracking directory


Train the Model

python3 src/train.py



MLFLOW UI:

mlflow ui --host 0.0.0.0 --port 5000


##### Open MLFlow in the browser: http://10.161.14.44:5000

🔧 Training LinearRegression...
✅ LinearRegression trained. MSE: 0.5558915986952444
📁 Model saved to: models/linearregression.pkl

🔧 Training DecisionTree...
✅ DecisionTree trained. MSE: 0.5245146178314735
📁 Model saved to: models/decisiontree.pkl






#########part3: API & Docker Packaging

Create an API for prediction using Flask or FastAPI.
Containerize the service using Docker.
Accept input via JSON and return model prediction.


API: FastAPI has been used for creating the model.

API contains 2 end points:

1. GET: (http://10.161.14.44:8000/)

Response: {
  "message": "California Housing Prediction API is running"
}


2. POST: (http://10.161.14.44:8000/predict)

Request Body:
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}


Response:

{
  "predicted_price": 4.778788739495789
}


Before Converting Container Saved the Best Model


Build the docker image housing-api:latest

created the docker-compose.yml and run the container by exposing the port 8000

Now docker ps -a gives the response:

CONTAINER ID   IMAGE                                                             COMMAND                   CREATED         STATUS         PORTS                                                   NAMES
b6820666cc79   housing-api:latest                                                "uvicorn api.app:app…"    9 minutes ago   Up 9 minutes   0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp             housing-api


##  Part 4: CI/CD with GitHub Actions

This project uses **GitHub Actions** to automate the CI/CD workflow, ensuring reliable and consistent deployment every time code is pushed to the `main` branch.

###  CI/CD Features

- **Linting**: Runs `flake8` on `src/` and `api/` directories to ensure code quality
- **Build**: Builds a Docker image of the FastAPI prediction service
- **Authentication**: Logs in to Docker Hub using secure GitHub secrets
- **Push**: Pushes the image to Docker Hub repository  
  → `docker.io/sumanth12345678/housing-api`
- **Deploy**: Executes a shell script to deploy the app locally via Docker Compose

###  Workflow Trigger

The CI/CD pipeline runs automatically on:
- Pushes to the `main` branch
- Manual trigger via the **"Run Workflow"** button (enabled via `workflow_dispatch`)

###  GitHub Secrets Used

| Secret Name        | Description                      |
|--------------------|----------------------------------|
| `DOCKER_USERNAME`  | Docker Hub username              |
| `DOCKER_PASSWORD`  | Docker Hub password or token     |

###  Key Files

| File                         | Purpose                          |
|------------------------------|----------------------------------|
| `.github/workflows/ci-cd.yml`| CI/CD pipeline definition        |
| `deploy_local.sh`            | Deploys the container locally    |
| `docker-compose.yml`         | Manages service deployment       |

###  Example Output

On a successful run:
- The Docker image is built and pushed
- Your FastAPI app is deployed and available at `http://10.161.14.44:8000/docs`



## Part 5: Logging and Monitoring

This section implements logging of prediction requests and monitoring of API performance metrics.

---

### ✅ Logging with SQLite (In-memory or file-based DB)

- All incoming prediction requests and their outputs are logged into a SQLite database named `logs.db`.
- Logs include:
  - Timestamp
  - Input features (MedInc, HouseAge, etc.)
  - Predicted price
- The log entries are stored in a table called `prediction_logs`.

#### 🧠 Log Schema:
| Column Name      | Type     |
|------------------|----------|
| id               | INTEGER (Auto Increment) |
| timestamp        | TEXT     |
| MedInc           | REAL     |
| HouseAge         | REAL     |
| AveRooms         | REAL     |
| AveBedrms        | REAL     |
| Population       | REAL     |
| AveOccup         | REAL     |
| Latitude         | REAL     |
| Longitude        | REAL     |
| predicted_price  | REAL     |

#### 🚀 Endpoint: `/logs`

- Purpose: Retrieve the latest predictions.
- Returns a list of logs in reverse chronological order.
- Accepts optional query param: `limit` (default = 10)

Example:

Response:
```json

{
  "logs": [
    [
      1,
      "2025-08-05T17:54:32.455535",
      "MedInc=8.3252 HouseAge=41.0 AveRooms=6.9841 AveBedrms=1.0238 Population=322.0 AveOccup=2.5556 Latitude=37.88 Longitude=-122.23",
      4.778788739495789
    ]
  ]
}


pip install prometheus-fastapi-instrumentator
from prometheus_fastapi_instrumentator import Instrumentator
app = FastAPI()
Instrumentator().instrument(app).expose(app)



#### 🚀 Endpoint: `/metrics`


