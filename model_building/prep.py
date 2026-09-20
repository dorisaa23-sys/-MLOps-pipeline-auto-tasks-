from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "machine-failure-prediction.csv"

print(f"Reading dataset from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.drop(columns=["UDI"], inplace=True)

X = df.drop(columns=["Failure"])
y = df["Failure"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

Xtrain.to_csv(PROJECT_ROOT / "Xtrain.csv", index=False)
Xtest.to_csv(PROJECT_ROOT / "Xtest.csv", index=False)
ytrain.to_csv(PROJECT_ROOT / "ytrain.csv", index=False)
ytest.to_csv(PROJECT_ROOT / "ytest.csv", index=False)

print("Data prepared: train/test files written.")
print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape:", ytest.shape)
print("Type values retained as:", sorted(X["Type"].unique()))
