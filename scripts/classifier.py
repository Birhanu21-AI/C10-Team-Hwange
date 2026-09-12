from pathlib import Path

import joblib
import numpy as np


class Classifier:
    """
    Competition classifier using the pre-trained Gemma latent probe.
    """

    def __init__(self):
        model_path = Path(__file__).with_name("trained_probe.joblib")
        self.model = joblib.load(model_path)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X).astype(int)
