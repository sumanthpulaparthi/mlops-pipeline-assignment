import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "California_Housing"
REGISTERED_MODEL_NAME = "BestCaliforniaHousingModel"

client = MlflowClient()

# Step 1: Get experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id

# Step 2: Get runs sorted by lowest MSE
runs = client.search_runs(
    experiment_ids=experiment_id,
    order_by=["metrics.mse ASC"],
    max_results=1
)

if not runs:
    print("❌ No runs found in experiment.")
    exit()

best_run = runs[0]
best_run_id = best_run.info.run_id
best_mse = best_run.data.metrics["mse"]
print(f"✅ Best model: run_id={best_run_id}, MSE={best_mse}")

# Step 3: Register the model
model_uri = f"runs:/{best_run_id}/model"

# Create model if not exists
try:
    client.create_registered_model(REGISTERED_MODEL_NAME)
except mlflow.exceptions.RestException:
    pass  # model already exists

# Create new version
model_version = client.create_model_version(
    name=REGISTERED_MODEL_NAME,
    source=model_uri,
    run_id=best_run_id
)

print(f"📌 Registered as version: {model_version.version}")

# Step 4: Transition to 'Production' stage (optional)
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"🚀 Model version {model_version.version} is now in 'Production'")
mlflow.sklearn.save_model(model, "models/best_model")
