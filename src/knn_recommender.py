import joblib
import warnings
from pathlib import Path

class KNNRecommender:
    def __init__(self, model_dir: Path = Path("../models")):
        sim_path = model_dir / "item_similarity.pkl"
        if sim_path.exists():
            self.sim_df = joblib.load(sim_path)
        else:
            warnings.warn("item_similarity.pkl not found.")
            self.sim_df = None

    def recommend(self, user_history: list, top_n: int = 1) -> list:
        if self.sim_df is None or not user_history:
            return []
        last_item = user_history[-1]
        if last_item not in self.sim_df.columns:
            return []
        sims = self.sim_df[last_item].sort_values(ascending=False)
        sims = sims[sims.index != last_item]
        return sims.head(top_n).index.tolist()