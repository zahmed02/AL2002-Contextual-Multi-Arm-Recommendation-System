import numpy as np
import json
from pathlib import Path

class LinUCB:
    def __init__(self, n_arms=2, n_features=6, alpha=1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
    def select_arm(self, context):
        max_ucb = -np.inf
        best_arm = 0
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            p = np.dot(theta, context)
            ucb = p + self.alpha * np.sqrt(context @ A_inv @ context)
            if ucb > max_ucb:
                max_ucb = ucb
                best_arm = arm
        return best_arm
    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
    @classmethod
    def load(cls, model_dir=Path("../models")):
        with open(model_dir / "linucb_params.json", "r") as f:
            params = json.load(f)
        bandit = cls(n_arms=2, n_features=params["n_features"], alpha=params["alpha"])
        bandit.A = [np.array(A) for A in params["A"]]
        bandit.b = [np.array(b) for b in params["b"]]
        return bandit