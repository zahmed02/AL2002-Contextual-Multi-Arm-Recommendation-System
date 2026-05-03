import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class ClusterAssigner:
    def __init__(self, model_dir: Path = Path("../models")):
        self.kmeans = joblib.load(model_dir / "kmeans_model.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.feature_cols = ['num_views', 'num_addtocart', 'num_transactions',
                             'unique_items', 'categories_viewed', 'duration_min']
    def assign_cluster(self, session_features: pd.Series) -> int:
        X = session_features[self.feature_cols].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)[0]