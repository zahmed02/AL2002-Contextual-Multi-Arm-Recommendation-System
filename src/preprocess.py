import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_context_vector(session_row: pd.Series, scaler: StandardScaler,
                       numeric_features: list) -> np.ndarray:
    """Build context vector for bandit (5 numeric features + cluster label)."""
    X_num = session_row[numeric_features].values.reshape(1, -1)
    X_num_scaled = scaler.transform(X_num).flatten()
    cluster_val = session_row['cluster']
    return np.concatenate([X_num_scaled, [cluster_val]])