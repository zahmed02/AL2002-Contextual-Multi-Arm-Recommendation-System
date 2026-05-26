import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from src.product_graph import build_category_graph
from src.astar_product import a_star_product
from src.clustering import ClusterAssigner
from src.preprocess import get_context_vector
from src.knn_recommender import KNNRecommender
from src.rf_predictor import PurchasePredictor
from src.contextual_bandit import LinUCB

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data/raw"
DATA_PROCESSED = BASE_DIR / "data/processed"
MODELS_DIR = BASE_DIR / "models"

# ------------------------------------------------------------
# Pre-flight check: make sure all large assets are present
# ------------------------------------------------------------
_REQUIRED_FILES = [
    MODELS_DIR / "item_similarity.pkl",
    MODELS_DIR / "kmeans_model.pkl",
    MODELS_DIR / "random_forest.pkl",
    MODELS_DIR / "scaler_context.pkl",
    DATA_PROCESSED / "events_with_sessions.parquet",
    DATA_PROCESSED / "user_sessions.parquet",
    DATA_PROCESSED / "user_clusters.parquet",
]
_missing = [f.relative_to(BASE_DIR) for f in _REQUIRED_FILES if not f.exists()]
if _missing:
    print()
    print("ERROR: The following required files are missing:")
    for _f in _missing:
        print(f"  - {_f}")
    print()
    print("To download pre-trained assets from the GitHub Release, run:")
    print("  python download_assets.py")
    print()
    print("Alternatively, regenerate them by running the Jupyter notebooks")
    print("in notebooks/ (01 -> 04) after placing the raw Kaggle CSVs in data/raw/.")
    print()
    sys.exit(1)


# ------------------------------------------------------------
# Load product graph and dummy heuristic (A*)
# ------------------------------------------------------------
parent_of, children_of, _ = build_category_graph(DATA_RAW)
def dummy_heuristic(a, b):
    return 0   # uniform cost – actual A* still finds shortest path

# ------------------------------------------------------------
# Load models and scalers
# ------------------------------------------------------------
rf_predictor = PurchasePredictor(model_dir=MODELS_DIR)
cluster_assigner = ClusterAssigner(model_dir=MODELS_DIR)
scaler_context = joblib.load(MODELS_DIR / "scaler_context.pkl")
bandit = LinUCB.load(model_dir=MODELS_DIR)

# KNN recommender – we'll extend it to return similarity scores
knn_recommender = KNNRecommender(model_dir=MODELS_DIR)

NUMERIC_FEATURES = ['num_views', 'num_addtocart', 'unique_items',
                    'categories_viewed', 'duration_min']
CLUSTER_FEATURES = NUMERIC_FEATURES + ['num_transactions']

# ------------------------------------------------------------
# Load existing sessions (features + cluster)
# ------------------------------------------------------------
existing_sessions = pd.read_parquet(DATA_PROCESSED / "user_sessions.parquet",
                                    columns=['visitorid', 'session_id', 'start_time',
                                             *NUMERIC_FEATURES, 'num_transactions'])
try:
    cluster_df = pd.read_parquet(DATA_PROCESSED / "user_clusters.parquet")
except:
    cluster_df = pd.read_csv(DATA_PROCESSED / "user_clusters.csv")
existing_sessions = existing_sessions.merge(cluster_df, on=['visitorid', 'session_id'], how='left')
existing_sessions['cluster'] = existing_sessions['cluster'].fillna(0).astype(int)
existing_sessions['visitorid'] = existing_sessions['visitorid'].astype(int)
existing_sessions['session_id'] = existing_sessions['session_id'].astype(int)

# ------------------------------------------------------------
# Build session → last item map (for KNN)
# ------------------------------------------------------------
events_df = pd.read_parquet(DATA_PROCESSED / "events_with_sessions.parquet",
                            columns=['visitorid', 'session_id', 'itemid', 'timestamp'])
events_df['visitorid'] = events_df['visitorid'].astype(int)
events_df['session_id'] = events_df['session_id'].astype(int)
valid_sessions = existing_sessions[['visitorid', 'session_id']].drop_duplicates()
events_df = events_df.merge(valid_sessions, on=['visitorid', 'session_id'], how='inner')
last_items = events_df.sort_values('timestamp').groupby(['visitorid', 'session_id']).last().reset_index()
session_last_item = dict(zip(zip(last_items['visitorid'], last_items['session_id']), last_items['itemid']))
print(f"Loaded {len(session_last_item)} sessions with last item.")

# ------------------------------------------------------------
# Build global popularity (for new sessions)
# ------------------------------------------------------------
item_popularity = events_df['itemid'].value_counts()
most_popular_item = int(item_popularity.index[0]) if not item_popularity.empty else 1000

# ------------------------------------------------------------
# Helper: bandit context from session row
# ------------------------------------------------------------
def build_bandit_context(session_row):
    return get_context_vector(session_row, scaler_context, NUMERIC_FEATURES)

# ------------------------------------------------------------
def get_knn_recommendation(last_item):
    """Return (item_id, similarity_score) or (None, 0) if fails."""
    recs = knn_recommender.recommend([last_item], top_n=1)
    if not recs:
        return None, 0.0
    rec_item = recs[0]
    # Get similarity from the precomputed matrix
    if rec_item in knn_recommender.sim_df.columns and last_item in knn_recommender.sim_df.index:
        sim = knn_recommender.sim_df.loc[last_item, rec_item]
        return rec_item, float(sim)
    return rec_item, 0.0

# ------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------
@app.route('/api/pathfinder', methods=['POST'])
def api_pathfinder():
    data = request.get_json()
    start = int(data['start'])
    goal = int(data['goal'])
    path, expanded = a_star_product(start, goal, parent_of, children_of, dummy_heuristic)
    if path is None:
        return jsonify({'error': 'No path found'}), 404
    return jsonify({
        'path': path,
        'nodes_explored': expanded,
        'path_length': len(path) - 1,
        'cost': len(path) - 1
    })

@app.route('/api/predictor', methods=['POST'])
def api_predictor():
    data = request.get_json()
    try:
        features = {k: float(data[k]) for k in NUMERIC_FEATURES}
        cluster = int(float(data.get('cluster', 0)))
    except Exception:
        return jsonify({'error': 'Invalid input'}), 400
    session_row = pd.Series({**features, 'cluster': cluster})
    prob = rf_predictor.predict_probability(session_row)
    risk = "Low" if prob < 0.5 else "High"
    return jsonify({
        'probability': round(prob * 100, 1),
        'risk_label': risk
    })

@app.route('/api/bandit/recommend', methods=['POST'])
def api_bandit():
    data = request.get_json()
    use_existing = data.get('use_existing', True)

    if not use_existing:
        # New session – user provides features, no history
        try:
            features = {k: float(data[k]) for k in NUMERIC_FEATURES}
            # Assign cluster (needs num_transactions=0)
            cluster_features = features.copy()
            cluster_features['num_transactions'] = 0
            cluster_val = cluster_assigner.assign_cluster(pd.Series(cluster_features))
            session_row = pd.Series({**features, 'cluster': cluster_val})
            context = build_bandit_context(session_row)
            # For new sessions, KNN cannot be used; we force RF arm
            arm = 1
            # Recommend most popular item globally
            recommended_item = most_popular_item
            # Confidence = RF purchase probability
            confidence = rf_predictor.predict_probability(session_row)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        # Existing session – use stored data
        try:
            visitor_id = int(data.get('visitor_id', 0))
            session_id = int(data.get('session_id', 0))
        except:
            return jsonify({'error': 'Invalid visitor/session ID (must be integers)'}), 400

        row = existing_sessions[
            (existing_sessions['visitorid'] == visitor_id) &
            (existing_sessions['session_id'] == session_id)
        ]
        if row.empty:
            return jsonify({'error': f'Session not found: ({visitor_id},{session_id})'}), 404
        row = row.iloc[0]
        context = build_bandit_context(row)

        force_arm = data.get('force_arm')
        if force_arm == 'knn':
            arm = 0
        elif force_arm == 'rf':
            arm = 1
        else:
            arm = bandit.select_arm(context)

        if arm == 0:  # KNN
            last_item = session_last_item.get((visitor_id, session_id))
            if last_item is None:
                return jsonify({'error': 'No interaction history for KNN'}), 400
            rec_item, sim_score = get_knn_recommendation(last_item)
            if rec_item is None:
                return jsonify({'error': 'No similar item found'}), 400
            recommended_item = rec_item
            confidence = sim_score
        else:  # Random Forest
            # Most frequent item in the session (from events)
            session_items = events_df[
                (events_df['visitorid'] == visitor_id) &
                (events_df['session_id'] == session_id)
            ]['itemid']
            if session_items.empty:
                return jsonify({'error': 'No items in this session'}), 400
            # Most viewed item in the session
            recommended_item = int(session_items.value_counts().index[0])
            confidence = rf_predictor.predict_probability(row)

    return jsonify({
        'arm': 'KNN' if arm == 0 else 'Random Forest',
        'recommended_item': recommended_item,
        'confidence': round(confidence, 3)
    })

# ------------------------------------------------------------
# HTML template — improved UX for non-technical users
# ------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Insights Engine — RetailRocket Recommendation System</title>
    <meta name="description" content="Interactive dashboard for the Contextual Multi-Arm Recommendation System: A* pathfinding, purchase prediction, and bandit-based recommendations.">
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0b1326; color: #dae2fd; }
        .glass-panel { background: rgba(30,41,59,0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); border-radius: 1rem; }
        .active-nav { background-color: rgba(78,222,163,0.12); border-left: 3px solid #4edea3; }
        .btn-primary { background: #4d8eff; color: #fff; font-weight: 600; transition: all 0.2s; border-radius: 0.75rem; }
        .btn-primary:hover:not(:disabled) { filter: brightness(1.12); box-shadow: 0 0 18px rgba(77,142,255,0.4); }
        .btn-primary:disabled { opacity: 0.55; cursor: not-allowed; }
        .input-dark { background: #060e20; border: 1px solid rgba(140,144,159,0.35); border-radius: 0.65rem; padding: 0.55rem 1rem; color: #dae2fd; width: 100%; }
        .input-dark:focus { outline: none; border-color: #4d8eff; box-shadow: 0 0 0 1px #4d8eff; }
        select.input-dark option { background: #0b1326; }
        .hint { font-size: 0.71rem; color: #6b7280; margin-top: 3px; line-height: 1.4; }
        .inline-error { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.35); border-radius: 0.5rem; padding: 0.6rem 1rem; color: #fca5a5; font-size: 0.84rem; margin-top: 0.5rem; display: none; }
        .info-card { background: rgba(77,142,255,0.06); border: 1px solid rgba(77,142,255,0.18); border-radius: 0.75rem; padding: 0.85rem 1.1rem; font-size: 0.82rem; color: #93b4e8; line-height: 1.6; }
        .stat-box { background: #060e20; border-radius: 0.75rem; padding: 0.9rem 1rem; }
        .tag { display: inline-block; font-size: 0.67rem; font-weight: 700; padding: 2px 9px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.06em; }
        .tag-green { background: rgba(78,222,163,0.15); color: #4edea3; }
        .tag-blue  { background: rgba(77,142,255,0.15); color: #4d8eff; }
        .tag-amber { background: rgba(251,191,36,0.15); color: #fbbf24; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .spinner { display: inline-block; width: 15px; height: 15px; border: 2px solid rgba(255,255,255,0.25); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
        .mode-btn { flex: 1; padding: 0.4rem; border-radius: 0.5rem; font-size: 0.85rem; font-weight: 500; transition: all 0.2s; cursor: pointer; border: none; }
        .mode-btn-active { background: #4d8eff; color: #fff; }
        .mode-btn-inactive { background: transparent; color: #6b7280; }
        .path-step { display: flex; align-items: center; gap: 10px; padding: 6px 4px; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .path-step:last-child { border-bottom: none; }
    </style>
</head>
<body>

<!-- ═══ SIDEBAR ═══════════════════════════════════════════════ -->
<aside class="fixed left-0 top-0 h-full w-64 flex flex-col z-50 p-5" style="background:#060e20; border-right: 1px solid rgba(255,255,255,0.07);">
    <div class="mb-8">
        <h1 class="text-lg font-bold" style="color:#4d8eff">AI / ML Dashboard</h1>
        <p class="text-xs mt-0.5" style="color:#4b5563">RetailRocket Recommendation System</p>
    </div>
    <nav class="flex-1 space-y-1">
        <a href="#" onclick="showSection('pathfinder');return false;" id="nav-pathfinder"
           class="flex items-center gap-3 px-3 py-3 rounded-lg active-nav" style="color:#dae2fd; text-decoration:none;">
            <span class="material-symbols-outlined text-[20px]" style="color:#4d8eff">route</span>
            <div>
                <div class="text-sm font-medium">Product Pathfinder</div>
                <div class="text-xs" style="color:#4b5563">A* category search</div>
            </div>
        </a>
        <a href="#" onclick="showSection('predictor');return false;" id="nav-predictor"
           class="flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-white/5" style="color:#dae2fd; text-decoration:none;">
            <span class="material-symbols-outlined text-[20px]" style="color:#4edea3">query_stats</span>
            <div>
                <div class="text-sm font-medium">Purchase Predictor</div>
                <div class="text-xs" style="color:#4b5563">Random Forest model</div>
            </div>
        </a>
        <a href="#" onclick="showSection('bandit');return false;" id="nav-bandit"
           class="flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-white/5" style="color:#dae2fd; text-decoration:none;">
            <span class="material-symbols-outlined text-[20px]" style="color:#fbbf24">smart_toy</span>
            <div>
                <div class="text-sm font-medium">Bandit Recommender</div>
                <div class="text-xs" style="color:#4b5563">KNN vs Random Forest</div>
            </div>
        </a>
    </nav>
    <div class="pt-4 border-t text-xs" style="border-color:rgba(255,255,255,0.07); color:#4b5563;">
        <p class="mb-1 font-medium" style="color:#6b7280">AL2002 — AI Lab</p>
        <p>Dataset: RetailRocket E-commerce</p>
        <p class="mt-1">1.76 M sessions · 2.75 M events</p>
    </div>
</aside>

<!-- ═══ MAIN CONTENT ══════════════════════════════════════════ -->
<main class="ml-64 p-8 overflow-y-auto h-screen">
<div class="max-w-5xl mx-auto space-y-8">

<!-- ─────────────────────────────────────────────────────────── -->
<!-- PATHFINDER                                                   -->
<!-- ─────────────────────────────────────────────────────────── -->
<div id="section-pathfinder" class="space-y-5">
    <div>
        <div class="flex items-center gap-3 mb-1">
            <h2 class="text-3xl font-bold">Product Pathfinder</h2>
            <span class="tag tag-blue">A* Search</span>
        </div>
        <p style="color:#6b7280">Finds the shortest navigation path between two product categories in the RetailRocket catalog — like GPS, but for an e-commerce category tree.</p>
    </div>

    <div class="info-card">
        <span class="material-symbols-outlined text-[15px] align-middle mr-1">info</span>
        The catalog is a forest of multiple disjoint category trees. The pre-filled pair <strong>(1000 → 2)</strong> is connected — hit <em>Find Shortest Path</em> to see it. If you try categories in disconnected trees (like <strong>1000 → 1542</strong>), it will correctly report <em>No path found</em>.
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="glass-panel p-6 space-y-4">
            <div>
                <label class="block text-sm font-medium mb-1">Start Category ID</label>
                <input type="number" id="start_cat" class="input-dark" value="1000">
                <p class="hint">A numeric product category from the RetailRocket catalog</p>
            </div>
            <div>
                <label class="block text-sm font-medium mb-1">Goal Category ID</label>
                <input type="number" id="goal_cat" class="input-dark" value="2">
                <p class="hint">Must be different from the start ID</p>
            </div>
            <div class="space-y-1">
                <span class="text-xs font-semibold uppercase tracking-wider" style="color:#6b7280">Quick Examples:</span>
                <div class="flex flex-wrap gap-2">
                    <button type="button" onclick="setPathfinderPreset(1000, 2)" class="px-2.5 py-1 text-xs rounded border border-white/10 hover:border-[#4d8eff]/50 hover:bg-[#4d8eff]/10 transition-colors" style="background:#060e20; color:#dae2fd">Standard Leap (1000 → 2)</button>
                    <button type="button" onclick="setPathfinderPreset(1000, 92)" class="px-2.5 py-1 text-xs rounded border border-white/10 hover:border-[#4d8eff]/50 hover:bg-[#4d8eff]/10 transition-colors" style="background:#060e20; color:#dae2fd">Short Step (1000 → 92)</button>
                    <button type="button" onclick="setPathfinderPreset(1000, 1542)" class="px-2.5 py-1 text-xs rounded border border-white/10 hover:border-[#ef4444]/50 hover:bg-[#ef4444]/10 transition-colors" style="background:#060e20; color:#dae2fd">Disjoint Trees (1000 → 1542)</button>
                </div>
            </div>
            <div id="pathfinder-error" class="inline-error"></div>
            <button id="btn-pathfinder" onclick="runPathfinder()" class="btn-primary w-full py-3 flex items-center justify-center gap-2">
                <span class="material-symbols-outlined text-[18px]">search</span> Find Shortest Path
            </button>
        </div>

        <div class="glass-panel p-6 space-y-4">
            <h3 class="font-semibold flex items-center gap-2">
                <span class="material-symbols-outlined text-[18px]" style="color:#4d8eff">timeline</span> Result
            </h3>
            <div id="pathfinder-result" class="text-sm" style="color:#6b7280">Enter two category IDs and click <em>Find Shortest Path</em>.</div>
            <div id="path-list" class="font-mono text-sm rounded-lg p-3" style="background:#060e20; min-height:40px;"></div>
        </div>
    </div>
</div>

<!-- ─────────────────────────────────────────────────────────── -->
<!-- PREDICTOR                                                    -->
<!-- ─────────────────────────────────────────────────────────── -->
<div id="section-predictor" class="space-y-5 hidden">
    <div>
        <div class="flex items-center gap-3 mb-1">
            <h2 class="text-3xl font-bold">Purchase Predictor</h2>
            <span class="tag tag-green">Random Forest</span>
        </div>
        <p style="color:#6b7280">Estimates how likely a shopping session is to end in a purchase. Describe what the visitor did during their session and the model returns a probability.</p>
    </div>

    <div class="info-card">
        <span class="material-symbols-outlined text-[15px] align-middle mr-1">info</span>
        The fields below represent a typical visitor's behaviour. The pre-filled values are a realistic example — click <em>Predict</em> to see the result, then try changing the numbers to see how they affect the score.
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="glass-panel p-6 space-y-4">
            <div class="space-y-1">
                <span class="text-xs font-semibold uppercase tracking-wider" style="color:#6b7280">Quick Personas:</span>
                <div class="flex flex-wrap gap-2">
                    <button type="button" onclick="setPredictorPersona(2, 0, 1, 1, 1.5, 0)" class="px-2.5 py-1 text-xs rounded border border-white/10 hover:border-[#4edea3]/50 hover:bg-[#4edea3]/10 transition-colors" style="background:#060e20; color:#dae2fd">Casual Browser</button>
                    <button type="button" onclick="setPredictorPersona(8, 1, 3, 2, 6.2, 1)" class="px-2.5 py-1 text-xs rounded border border-white/10 hover:border-[#4edea3]/50 hover:bg-[#4edea3]/10 transition-colors" style="background:#060e20; color:#dae2fd">Active Shopper</button>
                    <button type="button" onclick="setPredictorPersona(25, 5, 10, 4, 28.4, 3)" class="px-2.5 py-1 text-xs rounded border border-white/10 hover:border-[#4edea3]/50 hover:bg-[#4edea3]/10 transition-colors" style="background:#060e20; color:#dae2fd">High-Intent Buyer</button>
                </div>
            </div>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium mb-1">Page Views</label>
                    <input type="number" id="views" class="input-dark" value="12" min="0">
                    <p class="hint">Total product pages viewed</p>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">Add-to-Cart</label>
                    <input type="number" id="addtocart" class="input-dark" value="2" min="0">
                    <p class="hint">Times an item was added to cart</p>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">Unique Items</label>
                    <input type="number" id="unique_items" class="input-dark" value="4" min="0">
                    <p class="hint">Distinct products browsed</p>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">Categories Browsed</label>
                    <input type="number" id="categories" class="input-dark" value="2" min="0">
                    <p class="hint">Distinct sections of the store visited</p>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">Session Duration</label>
                    <input type="number" step="0.1" id="duration" class="input-dark" value="8.5" min="0">
                    <p class="hint">Total time spent shopping (minutes)</p>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">Shopper Type</label>
                    <select id="cluster" class="input-dark">
                        <option value="0">Casual Browser</option>
                        <option value="1" selected>Active Shopper</option>
                        <option value="2">Power User</option>
                        <option value="3">High-Intent Buyer</option>
                    </select>
                    <p class="hint">Best-guess visitor segment</p>
                </div>
            </div>
            <div id="predictor-error" class="inline-error"></div>
            <button id="btn-predictor" onclick="runPredictor()" class="btn-primary w-full py-3">
                Predict Purchase Likelihood
            </button>
        </div>

        <div class="glass-panel p-6 space-y-5 flex flex-col justify-center text-center">
            <div>
                <p class="text-xs uppercase tracking-widest mb-2" style="color:#6b7280">Purchase Probability</p>
                <div class="text-6xl font-bold" id="prob-value" style="color:#4edea3">--%</div>
            </div>
            <div class="h-3 w-full rounded-full overflow-hidden" style="background:#1e293b">
                <div id="prob-bar" class="h-full rounded-full transition-all duration-700" style="width:0%; background: linear-gradient(90deg,#4d8eff,#4edea3);"></div>
            </div>
            <div>
                <p class="text-sm mb-1" style="color:#6b7280">Risk Label</p>
                <span id="risk-label" class="text-xl font-bold">--</span>
            </div>
            <p class="text-xs" style="color:#4b5563">
                <strong style="color:#4edea3">Low risk</strong> = session unlikely to convert.<br>
                <strong style="color:#fbbf24">High risk</strong> = session is likely to end in a purchase.
            </p>
        </div>
    </div>
</div>

<!-- ─────────────────────────────────────────────────────────── -->
<!-- BANDIT RECOMMENDER                                          -->
<!-- ─────────────────────────────────────────────────────────── -->
<div id="section-bandit" class="space-y-5 hidden">
    <div>
        <div class="flex items-center gap-3 mb-1">
            <h2 class="text-3xl font-bold">Bandit Recommender</h2>
            <span class="tag tag-amber">LinUCB</span>
        </div>
        <p style="color:#6b7280">A self-learning recommendation engine that automatically picks the best strategy — collaborative filtering (KNN) or frequency-based (Random Forest) — based on who's shopping.</p>
    </div>

    <div class="info-card">
        <span class="material-symbols-outlined text-[15px] align-middle mr-1">info</span>
        <strong>Existing session:</strong> look up a real visitor from the dataset. Try Visitor <strong>0</strong>, Session <strong>0</strong> as a starting point, or Visitor <strong>6</strong> with Sessions 0, 1, or 2.<br>
        <strong>New session:</strong> simulate any visitor by entering their shopping behaviour manually.
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="glass-panel p-6 space-y-4">
            <!-- Mode toggle -->
            <div class="flex gap-1 p-1 rounded-xl" style="background:#060e20;">
                <button id="mode-existing-btn" onclick="setMode('existing')" class="mode-btn mode-btn-active">
                    Existing Session
                </button>
                <button id="mode-new-btn" onclick="setMode('new')" class="mode-btn mode-btn-inactive">
                    New Session
                </button>
            </div>

            <!-- Quick presets container -->
            <div id="bandit-presets-container" class="space-y-1">
                <span class="text-xs font-semibold uppercase tracking-wider" style="color:#6b7280">Quick-Select Visitor Samples:</span>
                <div class="flex flex-wrap gap-2">
                    <button type="button" onclick="setBanditVisitor(0, 0)" class="px-2.5 py-1 text-xs rounded border border-white/10 hover:border-[#fbbf24]/50 hover:bg-[#fbbf24]/10 transition-colors" style="background:#060e20; color:#dae2fd">Visitor 0 (Session 0)</button>
                    <button type="button" onclick="setBanditVisitor(6, 0)" class="px-2.5 py-1 text-xs rounded border border-white/10 hover:border-[#fbbf24]/50 hover:bg-[#fbbf24]/10 transition-colors" style="background:#060e20; color:#dae2fd">Visitor 6 (Session 0)</button>
                    <button type="button" onclick="setBanditVisitor(6, 2)" class="px-2.5 py-1 text-xs rounded border border-white/10 hover:border-[#fbbf24]/50 hover:bg-[#fbbf24]/10 transition-colors" style="background:#060e20; color:#dae2fd">Visitor 6 (Session 2)</button>
                </div>
            </div>

            <!-- Existing session fields -->
            <div id="existing-fields" class="space-y-3">
                <div>
                    <label class="block text-sm font-medium mb-1">Visitor ID</label>
                    <input type="number" id="visitor_id" class="input-dark" value="0" min="0">
                    <p class="hint">Numeric ID from the RetailRocket dataset — e.g. 0, 1, 2, 6, 7 …</p>
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">Session ID</label>
                    <input type="number" id="session_id" class="input-dark" value="0" min="0">
                    <p class="hint">Session number for this visitor — most have 0; visitor 6 has sessions 0, 1 and 2</p>
                </div>
            </div>

            <!-- New session fields -->
            <div id="new-fields" class="hidden space-y-3">
                <p class="text-xs" style="color:#6b7280">Describe the visitor's behaviour during their shopping session.</p>
                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="block text-sm font-medium mb-1">Page Views</label>
                        <input type="number" id="new_views" class="input-dark" value="10" min="0">
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-1">Add-to-Cart</label>
                        <input type="number" id="new_addtocart" class="input-dark" value="1" min="0">
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-1">Unique Items</label>
                        <input type="number" id="new_unique" class="input-dark" value="3" min="0">
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-1">Categories</label>
                        <input type="number" id="new_categories" class="input-dark" value="2" min="0">
                    </div>
                    <div class="col-span-2">
                        <label class="block text-sm font-medium mb-1">Duration (minutes)</label>
                        <input type="number" id="new_duration" class="input-dark" value="5.0" min="0" step="0.1">
                    </div>
                </div>
            </div>

            <!-- Strategy -->
            <div>
                <label class="block text-sm font-medium mb-1">Recommendation Strategy</label>
                <select id="arm_choice" class="input-dark">
                    <option value="auto">Let the Bandit Decide (recommended)</option>
                    <option value="knn">Force KNN — similarity-based</option>
                    <option value="rf">Force Random Forest — frequency-based</option>
                </select>
                <p class="hint">The bandit learns which arm performs better over time via LinUCB</p>
            </div>

            <div id="bandit-error" class="inline-error"></div>
            <button id="btn-bandit" onclick="runBandit()" class="btn-primary w-full py-3">
                Get Recommendation
            </button>
        </div>

        <!-- Result panel -->
        <div class="glass-panel p-6 space-y-4">
            <h3 class="font-semibold">Recommendation Result</h3>
            <div class="stat-box">
                <p class="text-xs uppercase tracking-widest mb-1" style="color:#6b7280">Strategy Chosen</p>
                <p id="bandit-arm" class="text-xl font-bold" style="color:#4d8eff">--</p>
            </div>
            <div class="stat-box">
                <p class="text-xs uppercase tracking-widest mb-1" style="color:#6b7280">Recommended Item ID</p>
                <p id="rec-item" class="text-3xl font-mono font-bold" style="color:#4edea3">--</p>
            </div>
            <div class="stat-box">
                <p class="text-xs uppercase tracking-widest mb-1" style="color:#6b7280">Confidence Score</p>
                <p id="rec-conf" class="text-xl font-bold" style="color:#fbbf24">--</p>
            </div>
            <p id="bandit-note" class="text-xs" style="color:#4b5563"></p>
        </div>
    </div>
</div>

</div>
</main>

<script>
    // ── Navigation ──────────────────────────────────────────────
    function showSection(section) {
        document.querySelectorAll('[id^="section-"]').forEach(el => el.classList.add('hidden'));
        document.getElementById(`section-${section}`).classList.remove('hidden');
        document.querySelectorAll('[id^="nav-"]').forEach(el => {
            el.classList.remove('active-nav');
            el.classList.add('hover:bg-white/5');
        });
        const nav = document.getElementById(`nav-${section}`);
        nav.classList.add('active-nav');
        nav.classList.remove('hover:bg-white/5');
    }

    // ── Interactive Presets & Personas ──────────────────────────
    function setPathfinderPreset(start, goal) {
        document.getElementById('start_cat').value = start;
        document.getElementById('goal_cat').value = goal;
        runPathfinder();
    }

    function setPredictorPersona(views, cart, unique, cats, duration, cluster) {
        document.getElementById('views').value = views;
        document.getElementById('addtocart').value = cart;
        document.getElementById('unique_items').value = unique;
        document.getElementById('categories').value = cats;
        document.getElementById('duration').value = duration;
        document.getElementById('cluster').value = cluster;
        runPredictor();
    }

    function setBanditVisitor(visitor, session) {
        setMode('existing');
        document.getElementById('visitor_id').value = visitor;
        document.getElementById('session_id').value = session;
        runBandit();
    }

    // ── Bandit mode toggle ──────────────────────────────────────
    function setMode(mode) {
        const isExisting = mode === 'existing';
        document.getElementById('existing-fields').classList.toggle('hidden', !isExisting);
        document.getElementById('new-fields').classList.toggle('hidden', isExisting);
        document.getElementById('bandit-presets-container').classList.toggle('hidden', !isExisting);
        document.getElementById('mode-existing-btn').className = 'mode-btn ' + (isExisting ? 'mode-btn-active' : 'mode-btn-inactive');
        document.getElementById('mode-new-btn').className      = 'mode-btn ' + (!isExisting ? 'mode-btn-active' : 'mode-btn-inactive');
    }

    // ── Shared helpers ──────────────────────────────────────────
    async function callAPI(endpoint, data) {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return res.json();
    }

    function setLoading(btnId, loading, label) {
        const btn = document.getElementById(btnId);
        btn.disabled = loading;
        btn.innerHTML = loading ? '<span class="spinner"></span> Working…' : label;
    }

    function showError(divId, msg) {
        const el = document.getElementById(divId);
        el.textContent = '⚠  ' + msg;
        el.style.display = 'block';
    }

    function clearError(divId) {
        const el = document.getElementById(divId);
        el.style.display = 'none';
        el.textContent = '';
    }

    // ── Pathfinder ──────────────────────────────────────────────
    async function runPathfinder() {
        clearError('pathfinder-error');
        const label = '<span class="material-symbols-outlined text-[18px]">search</span> Find Shortest Path';
        setLoading('btn-pathfinder', true, label);
        const start = parseInt(document.getElementById('start_cat').value);
        const goal  = parseInt(document.getElementById('goal_cat').value);
        const res   = await callAPI('/api/pathfinder', { start, goal });
        setLoading('btn-pathfinder', false, label);
        if (res.error) {
            showError('pathfinder-error', res.error);
            document.getElementById('pathfinder-result').innerHTML = '';
            document.getElementById('path-list').innerHTML = '';
            return;
        }
        document.getElementById('pathfinder-result').innerHTML =
            `<div class="grid grid-cols-3 gap-3 text-center">
                <div class="stat-box"><p class="text-xs" style="color:#6b7280">Hops</p><p class="text-2xl font-bold" style="color:#4d8eff">${res.path_length}</p></div>
                <div class="stat-box"><p class="text-xs" style="color:#6b7280">Cost</p><p class="text-2xl font-bold" style="color:#4edea3">${res.cost}</p></div>
                <div class="stat-box"><p class="text-xs" style="color:#6b7280">Nodes Explored</p><p class="text-2xl font-bold" style="color:#fbbf24">${res.nodes_explored}</p></div>
            </div>`;
        document.getElementById('path-list').innerHTML =
            `<p class="text-xs mb-2" style="color:#6b7280">Category sequence:</p>` +
            res.path.map((id, i) =>
                `<div class="path-step">
                    <span style="color:#4b5563; min-width:20px; font-size:0.75rem">${i + 1}.</span>
                    <span style="color:#4edea3">CAT_${id}</span>
                    ${i === 0 ? '<span class="tag tag-blue ml-auto">Start</span>' : ''}
                    ${i === res.path.length - 1 ? '<span class="tag tag-green ml-auto">Goal ✓</span>' : ''}
                </div>`
            ).join('');
    }

    // ── Predictor ───────────────────────────────────────────────
    async function runPredictor() {
        clearError('predictor-error');
        setLoading('btn-predictor', true, 'Predict Purchase Likelihood');
        const payload = {
            num_views:         parseFloat(document.getElementById('views').value),
            num_addtocart:     parseFloat(document.getElementById('addtocart').value),
            unique_items:      parseFloat(document.getElementById('unique_items').value),
            categories_viewed: parseFloat(document.getElementById('categories').value),
            duration_min:      parseFloat(document.getElementById('duration').value),
            cluster:           parseFloat(document.getElementById('cluster').value)
        };
        const res = await callAPI('/api/predictor', payload);
        setLoading('btn-predictor', false, 'Predict Purchase Likelihood');
        if (res.error) { showError('predictor-error', res.error); return; }
        document.getElementById('prob-value').textContent = `${res.probability}%`;
        document.getElementById('prob-bar').style.width   = `${res.probability}%`;
        const riskEl = document.getElementById('risk-label');
        riskEl.textContent = res.risk_label + ' Risk';
        riskEl.style.color = res.risk_label === 'High' ? '#fbbf24' : '#4edea3';
    }

    // ── Bandit ──────────────────────────────────────────────────
    async function runBandit() {
        clearError('bandit-error');
        setLoading('btn-bandit', true, 'Get Recommendation');
        const isExisting = !document.getElementById('existing-fields').classList.contains('hidden');
        const forceArm   = document.getElementById('arm_choice').value;
        let payload = { use_existing: isExisting, force_arm: forceArm === 'auto' ? null : forceArm };
        if (isExisting) {
            payload.visitor_id = parseInt(document.getElementById('visitor_id').value);
            payload.session_id = parseInt(document.getElementById('session_id').value);
        } else {
            payload.num_views         = parseFloat(document.getElementById('new_views').value);
            payload.num_addtocart     = parseFloat(document.getElementById('new_addtocart').value);
            payload.unique_items      = parseFloat(document.getElementById('new_unique').value);
            payload.categories_viewed = parseFloat(document.getElementById('new_categories').value);
            payload.duration_min      = parseFloat(document.getElementById('new_duration').value);
        }
        const res = await callAPI('/api/bandit/recommend', payload);
        setLoading('btn-bandit', false, 'Get Recommendation');
        if (res.error) { showError('bandit-error', res.error); return; }
        document.getElementById('bandit-arm').textContent = res.arm;
        document.getElementById('rec-item').textContent   = res.recommended_item;
        document.getElementById('rec-conf').textContent   = (res.confidence * 100).toFixed(1) + '%';
        document.getElementById('bandit-note').textContent = isExisting
            ? `The bandit selected the ${res.arm} arm based on this visitor's context vector and LinUCB confidence bounds.`
            : 'New session — the globally most-popular item is recommended; confidence = RF purchase probability.';
    }

    showSection('pathfinder');
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)