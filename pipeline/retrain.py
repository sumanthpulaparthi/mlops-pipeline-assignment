import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def retrain_model(df: pd.DataFrame):
    model_dir = "models"
    best_model_dir = os.path.join(model_dir, "best_model")
    experiment_name = "California_Housing"
    mlflow_model_name = "CaliforniaHousingBestModel"

    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(best_model_dir):
        shutil.rmtree(best_model_dir)

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5)
    }

    best_mse = float("inf")
    best_model = None
    best_run_id = None

    for model_name, model in models.items():
        print(f"\n🔧 Retraining {model_name}...")

        with mlflow.start_run(run_name=f"Retrain_{model_name}") as run:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            mlflow.log_param("model_type", model_name)
            if model_name == "DecisionTree":
                mlflow.log_param("max_depth", 5)
            mlflow.log_metric("mse", float(mse))

            mlflow.sklearn.log_model(
                model,
                "model",
                input_example=X_test[:5]
            )

            local_model_path = os.path.join(
                model_dir, f"{model_name.lower()}_retrained.pkl"
            )
            mlflow.sklearn.save_model(model, local_model_path)
            print(f"✅ {model_name} retrained. MSE: {mse:.4f}")
            print(f"📁 Model saved to: {local_model_path}")

            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_run_id = run.info.run_id

    if best_model and best_run_id:
        print(
            f"\n🏆 Saving and registering best model with MSE: "
            f"{best_mse:.4f} to {best_model_dir}"
        )
        mlflow.sklearn.save_model(best_model, best_model_dir)

        best_model_uri = f"runs:/{best_run_id}/model"
        register_result = mlflow.register_model(
            model_uri=best_model_uri,
            name=mlflow_model_name
        )
        version = getattr(register_result, "version", None) or register_result
        version = str(version)
        try:
            client.set_registered_model_alias(
                name="CaliforniaHousingBestModel",
                alias="production",
                version=version
            )
            client.set_model_version_tag(
                name="CaliforniaHousingBestModel",
                version=version,
                key="deployment_note",
                value=f"MSE={best_mse:.4f}"
            )
            print(
                f"✅ Best model registered and promoted to Production as "
                f"'{mlflow_model_name}' (v{version})"
            )
        except Exception as e:
            print(f"⚠️ Could not transition model to 'Production': {e}")
    else:
        print("No model was retrained successfully.")

