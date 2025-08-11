from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Save a subset as new training data
df.sample(100).to_csv("new_data.csv", index=False)

print("new_data.csv generated.")

