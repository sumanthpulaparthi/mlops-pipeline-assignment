import os
import shutil
from mlflow import MlflowClient, pyfunc
from mlflow.pyfunc import load_model, save_model
from mlflow.exceptions import MlflowException
from mlflow.entities import ViewType

LOCAL_MODEL_PATH = "models/best_model"

def update_production_model():
    client = MlflowClient()

    # 🔍 Find the latest run with the best MSE
    experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    best_run = None
    best_mse = float("inf")
    best_experiment_name = None

    # 🔍 Loop through experiments to find the best run
    for experiment in experiments:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.mse ASC"],
            max_results=1,
        )
        if runs:
            run = runs[0]
            mse = run.data.metrics.get("mse")
            if mse is not None and mse < best_mse:
                best_run = run
                best_mse = mse
                best_experiment_name = experiment.name

    if best_run is None:
        raise ValueError("❌ No valid runs with 'mse' metric found.")

    print(f"✅ Best run found: {best_run.info.run_id} in experiment '{best_experiment_name}', MSE = {best_mse:.4f}")

    best_run_id = best_run.info.run_id
    model_uri = f"runs:/{best_run_id}/model"

    # 🏷️ Define registered model name from convention or tag (fallback to experiment name)
    REGISTERED_MODEL_NAME = best_experiment_name.replace(" ", "_") + "_Model"

    # 🧹 Clean up local model dir
    if os.path.exists(LOCAL_MODEL_PATH):
        shutil.rmtree(LOCAL_MODEL_PATH)
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

    # 🗂️ Create model registry entry
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
        print(f"✅ Created registered model: {REGISTERED_MODEL_NAME}")
    except MlflowException as e:
        if "already exists" in str(e):
            print(f"ℹ️ Registered model '{REGISTERED_MODEL_NAME}' already exists.")
        else:
            raise e

    # 📦 Register this specific model version
    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=f"{client.get_run(best_run_id).info.artifact_uri}/model",
        run_id=best_run_id
    )
    print(f"📌 Registered new version: {model_version.version}")

    # 🚀 Promote to Production (archive previous versions)
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"🚀 Model version {model_version.version} is now in 'Production'")

    # 💾 Save best model locally

    print("📥 Downloading and saving model artifacts locally...")
    client.download_artifacts(run_id=best_run_id, path="model", dst_path=LOCAL_MODEL_PATH)
    print(f"✅ Model artifacts downloaded to '{LOCAL_MODEL_PATH}'")
