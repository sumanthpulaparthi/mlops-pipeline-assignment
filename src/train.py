import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ========== Paths ==========
DATA_PATH = "data/raw/housing.csv"
MODEL_DIR = "models"
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_model")
MLFLOW_MODEL_NAME = "CaliforniaHousingBestModel"

os.makedirs(MODEL_DIR, exist_ok=True)
if os.path.exists(BEST_MODEL_DIR):
    shutil.rmtree(BEST_MODEL_DIR)

# ========== Load Dataset ==========
print(f"📥 Loading dataset from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== MLflow Setup ==========
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("California_Housing")
client = MlflowClient()

# ========== Model Training ==========
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5)
}

best_mse = float("inf")
best_model_name = None
best_run_id = None
best_model = None

for model_name, model in models.items():
    print(f"\n🔧 Training {model_name} ...")
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        mlflow.log_param("model_type", model_name)
        if model_name == "DecisionTree":
            mlflow.log_param("max_depth", 5)
        mlflow.log_metric("mse", float(mse))

        # Log the model (do not use artifact_path in new MLflow, use 'name')
        mlflow.sklearn.log_model(model, "model", input_example=X_test[:5])

        # Save locally as backup (optional, not required for MLflow)
        local_model_path = os.path.join(MODEL_DIR, f"{model_name.lower()}.pkl")
        mlflow.sklearn.save_model(model, local_model_path)
        print(f"✅ {model_name} MSE: {mse:.4f}")
        print(f"📁 Model saved to: {local_model_path}")

        # Track best model (by lowest MSE)
        if mse < best_mse:
            best_mse = mse
            best_model_name = model_name
            best_model = model
            best_run_id = run.info.run_id

# ========== Model Registration ==========
if best_model is not None and best_run_id is not None:
    print(f"\n🏆 Best model: {best_model_name} (MSE: {best_mse:.4f})")
    best_model_uri = f"runs:/{best_run_id}/model"

    # Register the model (outputs a ModelVersion object or version str)
    register_result = mlflow.register_model(
        model_uri=best_model_uri,
        name=MLFLOW_MODEL_NAME
    )

    # Get the version robustly (string for API call)
    version = getattr(register_result, "version", None) or register_result
    version = str(version)

    # --- Transition to 'Production' (if API is available/not deprecated) ---
    try:
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"✅ Best model registered and promoted to Production as '{MLFLOW_MODEL_NAME}' (v{version})")
    except Exception as e:
        print(f"⚠️  Could not transition model to 'Production'. Details: {e}")

    # Save best model locally (for backup)
    mlflow.sklearn.save_model(best_model, BEST_MODEL_DIR)
    print(f"📁 Backup saved locally to: {BEST_MODEL_DIR}")

else:
    print("No model was trained successfully.")

