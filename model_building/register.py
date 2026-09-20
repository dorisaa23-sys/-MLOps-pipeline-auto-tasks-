from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "machine-failure-prediction.csv"

print(f"Reading dataset from: {RAW_PATH}")

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate the expected columns
expected_columns = [
    "UDI", "Type", "Air temperature", "Process temperature",
    "Rotational speed", "Torque", "Tool wear", "Failure",
]

missing = [column for column in expected_columns if column not in df.columns]

if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Failure distribution:")
print(df["Failure"].value_counts())
