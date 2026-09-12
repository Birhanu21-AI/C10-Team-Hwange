import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42

df = pd.read_parquet("data/processed/canonical_split.parquet")
X = np.load("data/embeddings/layer13_mean.npy")
y = df["label"].to_numpy()

train_val_mask = df["split"].isin(["train", "validation"]).to_numpy()

X_train_val = X[train_val_mask]
y_train_val = y[train_val_mask]

with open("best_config.json", "r") as f:
    best_config = json.load(f)

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            C=best_config["C"],
            class_weight=best_config["class_weight"],
            max_iter=2000,
            random_state=SEED,
        ),
    ),
])

model.fit(X_train_val, y_train_val)

os.makedirs("submission", exist_ok=True)

joblib.dump(model, "submission/trained_probe.joblib")

print("Saved submission/trained_probe.joblib")
