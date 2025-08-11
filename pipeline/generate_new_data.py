
from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

# Ensure the processed folder exists
os.makedirs("data/processed", exist_ok=True)

# Load the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Save a subset as new training data
output_path = "data/processed/generated_data.csv"
df.sample(100).to_csv(output_path, index=False)

print(f"✅ {output_path} generated.")

