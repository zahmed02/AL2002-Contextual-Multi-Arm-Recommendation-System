import pandas as pd
from pathlib import Path

def build_category_graph(data_dir: Path = Path("../data/raw")):
    category_tree = pd.read_csv(data_dir / "category_tree.csv")
    parent_of = {}
    children_of = {}
    for _, row in category_tree.iterrows():
        child = int(row["categoryid"])
        if pd.notna(row["parentid"]):
            parent = int(row["parentid"])
            parent_of[child] = parent
            children_of.setdefault(parent, []).append(child)
    # Also add roots as keys in children_of even if they have no children (for consistency)
    roots = category_tree[pd.isna(category_tree["parentid"])]["categoryid"].tolist()
    for root in roots:
        children_of.setdefault(root, [])
    return parent_of, children_of, roots