import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# ========== Paths ==========
DATA_PATH = "data/raw/housing.csv"
MODEL_DIR = "models"

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ========== Load Dataset ==========
print(f"Loading dataset from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== MLflow Setup ==========
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("California_Housing")

# ========== Model Training ==========
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5)
}

for model_name, model in models.items():
    print(f"\n🔧 Training {model_name}...")

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Log to MLflow
        mlflow.log_param("model_type", model_name)
        if model_name == "DecisionTree":
            mlflow.log_param("max_depth", 5)
        mlflow.log_metric("mse", mse)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        local_model_path = os.path.join(MODEL_DIR, f"{model_name.lower()}.pkl")
        mlflow.sklearn.save_model(model, local_model_path)
        print(f"✅ {model_name} trained. MSE: {mse:.4f}")
        print(f"📁 Model saved to: {local_model_path}")

