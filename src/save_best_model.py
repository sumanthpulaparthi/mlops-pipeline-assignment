import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.sklearn import load_model, save_model

import os

# Constants
EXPERIMENT_NAME = "California_Housing"
REGISTERED_MODEL_NAME = "BestCaliforniaHousingModel"
LOCAL_MODEL_PATH = "models/best_model"

# Set tracking URI if needed (e.g., for remote MLflow)
# mlflow.set_tracking_uri("http://your-tracking-server")

# Init MLflow client
client = MlflowClient()

# Step 1: Get experiment ID
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
experiment_id = experiment.experiment_id

# Step 2: Get best run based on MSE
runs = client.search_runs(
    experiment_ids=experiment_id,
    order_by=["metrics.mse ASC"],
    max_results=1
)

if not runs:
    raise ValueError("❌ No runs found in experiment.")

best_run = runs[0]
best_run_id = best_run.info.run_id
best_mse = best_run.data.metrics["mse"]
print(f"✅ Best model found: run_id={best_run_id}, MSE={best_mse}")

# Step 3: Prepare model URI
model_uri = f"runs:/{best_run_id}/model"

# Step 4: Register model version (skip if already exists)
try:
    client.create_registered_model(REGISTERED_MODEL_NAME)
    print(f"✅ Created registered model: {REGISTERED_MODEL_NAME}")
except MlflowException as e:
    if "already exists" in str(e):
        print(f"ℹ️ Registered model '{REGISTERED_MODEL_NAME}' already exists.")
    else:
        raise e

# Step 5: Register new model version
model_version = client.create_model_version(
    name=REGISTERED_MODEL_NAME,
    source=model_uri,
    run_id=best_run_id
)

print(f"📌 Registered new version: {model_version.version}")

# Step 6: Transition to 'Production'
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"🚀 Model version {model_version.version} is now in 'Production'")

# Step 7: Save best model locally for Docker/API
print("💾 Saving best model locally to 'models/best_model'...")
loaded_model = load_model(model_uri)

# Ensure models directory exists
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
save_model(loaded_model, LOCAL_MODEL_PATH)
print("✅ Model saved locally.")

