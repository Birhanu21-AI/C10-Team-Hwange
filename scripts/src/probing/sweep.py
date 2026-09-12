import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


SEED = 42

df = pd.read_parquet("data/processed/canonical_split.parquet")

X = np.load("data/embeddings/layer13_mean.npy")
y = df["label"].to_numpy()

train_mask = df["split"].eq("train").to_numpy()
val_mask = df["split"].eq("validation").to_numpy()

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

candidates = [
    {"C": 1.0, "class_weight": "balanced"},
    {"C": 0.1, "class_weight": "balanced"},
    {"C": 1.0, "class_weight": None},
    {"C": 0.1, "class_weight": None},
]

results = []

for config in candidates:
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(
        C=config["C"],
        class_weight=config["class_weight"],
        max_iter=2000,
        random_state=SEED,
    )

    clf.fit(X_train_scaled, y_train)

    preds = clf.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, preds)

    result = {
        **config,
        "validation_accuracy": float(accuracy),
    }

    results.append(result)

    print(result)

best = max(results, key=lambda x: x["validation_accuracy"])

with open("best_config.json", "w") as f:
    json.dump(best, f, indent=2)

print("\nBest configuration:")
print(json.dumps(best, indent=2))
