from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load files created by prep.py
Xtrain = pd.read_csv(PROJECT_ROOT / "Xtrain.csv")
Xtest = pd.read_csv(PROJECT_ROOT / "Xtest.csv")
ytrain = pd.read_csv(PROJECT_ROOT / "ytrain.csv").squeeze("columns")
ytest = pd.read_csv(PROJECT_ROOT / "ytest.csv").squeeze("columns")

numeric_features = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

categorical_features = ["Type"]

# Handle class imbalance
class_counts = ytrain.value_counts()
class_weight = class_counts[0] / class_counts[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=1
)

pipeline = make_pipeline(preprocessor, model)

param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1
)

grid.fit(Xtrain, ytrain)

best_model = grid.best_estimator_

print("Best parameters:", grid.best_params_)
print(classification_report(ytest, best_model.predict(Xtest)))

# Save the trained model inside the deployment folder
MODEL_PATH = (
    PROJECT_ROOT
    / "deployment"
    / "best_machine_failure_model_v1.joblib"
)

joblib.dump(best_model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
