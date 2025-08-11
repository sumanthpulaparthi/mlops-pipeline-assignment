# save_best_model.py
import os,shutil
from mlflow.tracking import MlflowClient
from mlflow.sklearn import load_model, save_model
from mlflow.exceptions import MlflowException

def update_production_model():
    EXPERIMENT_NAME = "California_Housing"
    REGISTERED_MODEL_NAME = "BestCaliforniaHousingModel"
    LOCAL_MODEL_PATH = "models/best_model"
    if os.path.exists(LOCAL_MODEL_PATH):
       shutil.rmtree(LOCAL_MODEL_PATH)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=experiment_id,
        order_by=["metrics.mse ASC"],
        max_results=1
    )
    print(runs)
    if not runs:
        raise ValueError("❌ No runs found in experiment.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_mse = best_run.data.metrics["mse"]
    print(f"✅ Best model found: run_id={best_run_id}, MSE={best_mse}")

    model_uri = f"runs:/{best_run_id}/model"

    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
        print(f"✅ Created registered model: {REGISTERED_MODEL_NAME}")
    except MlflowException as e:
        if "already exists" in str(e):
            print(f"ℹ️ Registered model '{REGISTERED_MODEL_NAME}' already exists.")
        else:
            raise e

    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=model_uri,
        run_id=best_run_id
    )

    print(f"📌 Registered new version: {model_version.version}")

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"🚀 Model version {model_version.version} is now in 'Production'")

    print("💾 Saving best model locally to 'models/best_model'...")
    loaded_model = load_model(model_uri)
    # Ensure models directory is empty

    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
    save_model(loaded_model, LOCAL_MODEL_PATH)
    print("✅ Model saved locally.")

