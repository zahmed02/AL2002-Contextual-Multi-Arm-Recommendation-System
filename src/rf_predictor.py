import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class PurchasePredictor:
    def __init__(self, model_dir: Path = Path("../models"), scaler=None):
        self.rf = joblib.load(model_dir / "random_forest.pkl")
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = joblib.load(model_dir / "scaler_context.pkl")
        self.feature_cols = ['num_views', 'num_addtocart', 'unique_items',
                             'categories_viewed', 'duration_min']
        self.cluster_dummies = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3']

    def predict_probability(self, session_df_row: pd.Series) -> float:
        num_features = session_df_row[self.feature_cols].values.reshape(1, -1)
        num_scaled = self.scaler.transform(num_features).flatten()
        cluster = int(session_df_row['cluster'])
        cluster_dummy = [1 if i == cluster else 0 for i in range(4)]
        X = np.concatenate([num_scaled, cluster_dummy]).reshape(1, -1)
        prob = self.rf.predict_proba(X)[0, 1]
        return prob