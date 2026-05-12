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
session_last_item = {
    (row['visitorid'], row['session_id']): row['itemid']
    for _, row in last_items.iterrows()
}
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
# HTML template (same clean UI, no hard‑coded examples)
# ------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Insights Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0b1326; color: #dae2fd; }
        .glass-panel { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); border-radius: 1rem; }
        .active-nav { background-color: rgba(78, 222, 163, 0.1); border-right: 2px solid #4edea3; }
        .btn-primary { background: #4d8eff; color: #002e6a; font-weight: bold; transition: all 0.2s; }
        .btn-primary:hover { filter: brightness(1.1); box-shadow: 0 0 12px rgba(77,142,255,0.3); }
        .input-dark { background: #060e20; border: 1px solid rgba(140,144,159,0.5); border-radius: 0.75rem; padding: 0.5rem 1rem; color: #dae2fd; }
        .input-dark:focus { outline: none; border-color: #4d8eff; box-shadow: 0 0 0 1px #4d8eff; }
    </style>
    <script>
        async function callAPI(endpoint, data) {
            const res = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
            return res.json();
        }
    </script>
</head>
<body>
<aside class="fixed left-0 top-0 h-full w-64 bg-surface-container-low border-r border-white/10 backdrop-blur-lg flex flex-col z-50 p-6">
    <div class="mb-8"><h1 class="text-2xl font-bold text-primary">AI/ML Dashboard</h1><p class="text-xs text-on-surface-variant uppercase">Enterprise Intelligence</p></div>
    <nav class="flex-1 space-y-2">
        <a href="#" onclick="showSection('pathfinder');return false;" id="nav-pathfinder" class="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-container-high active-nav"><span class="material-symbols-outlined">route</span><span>Product Pathfinder</span></a>
        <a href="#" onclick="showSection('predictor');return false;" id="nav-predictor" class="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-container-high"><span class="material-symbols-outlined">query_stats</span><span>Purchase Predictor</span></a>
        <a href="#" onclick="showSection('bandit');return false;" id="nav-bandit" class="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-container-high"><span class="material-symbols-outlined">smart_toy</span><span>Bandit Recommender</span></a>
    </nav>
    <div class="mt-auto pt-4 border-t border-white/10"><button class="w-full py-2 bg-primary text-on-primary rounded-lg font-bold">Run New Model</button></div>
</aside>
<main class="ml-64 p-8 overflow-y-auto h-screen"><div class="max-w-6xl mx-auto space-y-8">
    <!-- Pathfinder -->
    <div id="section-pathfinder" class="space-y-6"><div><h2 class="text-3xl font-bold">Product Pathfinder</h2><p class="text-on-surface-variant">Optimal category path using A* heuristic search</p></div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6"><div class="glass-panel p-6 space-y-4"><label class="block text-sm uppercase">Start Category (ID)</label><input type="number" id="start_cat" class="input-dark w-full" value="1000"><label class="block text-sm uppercase">Goal Category (ID)</label><input type="number" id="goal_cat" class="input-dark w-full" value="1542"><button onclick="runPathfinder()" class="btn-primary w-full py-3 rounded-lg flex items-center justify-center gap-2"><span class="material-symbols-outlined">search</span> Find Path</button></div>
    <div class="glass-panel p-6 space-y-4"><h3 class="font-bold flex items-center gap-2"><span class="material-symbols-outlined text-secondary">timeline</span> Result</h3><div id="pathfinder-result"><p>--</p></div><div id="path-list" class="bg-surface-container-lowest p-3 rounded-lg font-mono text-sm"></div></div></div></div>
    <!-- Predictor -->
    <div id="section-predictor" class="space-y-6 hidden"><div><h2 class="text-3xl font-bold">Purchase Predictor</h2><p class="text-on-surface-variant">Random Forest conversion probability</p></div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6"><div class="glass-panel p-6 space-y-4"><div class="grid grid-cols-2 gap-4"><div><label>Views</label><input type="number" id="views" class="input-dark w-full" value="12"></div><div><label>Add-to-carts</label><input type="number" id="addtocart" class="input-dark w-full" value="2"></div><div><label>Unique items</label><input type="number" id="unique_items" class="input-dark w-full" value="4"></div><div><label>Categories viewed</label><input type="number" id="categories" class="input-dark w-full" value="2"></div><div><label>Duration (min)</label><input type="number" step="0.1" id="duration" class="input-dark w-full" value="8.5"></div><div><label>Cluster (0–3)</label><input type="number" id="cluster" class="input-dark w-full" value="1"></div></div><button onclick="runPredictor()" class="btn-primary w-full py-3 rounded-lg">Predict</button></div>
    <div class="glass-panel p-6 space-y-4 text-center"><span class="text-sm text-on-surface-variant">PURCHASE PROBABILITY</span><div class="text-5xl font-bold text-secondary" id="prob-value">--%</div><div class="h-2 w-full bg-surface-container-high rounded-full overflow-hidden"><div id="prob-bar" class="h-full bg-secondary" style="width:0%"></div></div><div><span class="text-sm">Risk label: </span><span id="risk-label" class="font-bold">--</span></div></div></div></div>
    <!-- Bandit -->
    <div id="section-bandit" class="space-y-6 hidden"><div><h2 class="text-3xl font-bold">Bandit Recommender</h2><p class="text-on-surface-variant">Contextual multi‑armed bandit – KNN vs Random Forest</p></div>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6"><div class="glass-panel p-6 space-y-4"><div class="flex gap-4"><label class="inline-flex items-center gap-2"><input type="radio" name="session_mode" value="existing" checked> Use existing session</label><label class="inline-flex items-center gap-2"><input type="radio" name="session_mode" value="new"> Create new session</label></div>
    <div id="existing-fields"><label class="block text-sm">Visitor ID</label><input type="number" id="visitor_id" class="input-dark w-full" value="0"><label class="block text-sm mt-3">Session ID</label><input type="number" id="session_id" class="input-dark w-full" value="0"></div>
    <div id="new-fields" class="hidden space-y-3"><p class="text-sm text-secondary">Enter session behavior</p><div class="grid grid-cols-2 gap-3"><div><label>Views</label><input type="number" id="new_views" class="input-dark w-full"></div><div><label>Add-to-carts</label><input type="number" id="new_addtocart" class="input-dark w-full"></div><div><label>Unique items</label><input type="number" id="new_unique" class="input-dark w-full"></div><div><label>Categories</label><input type="number" id="new_categories" class="input-dark w-full"></div><div><label>Duration (min)</label><input type="number" id="new_duration" class="input-dark w-full"></div></div></div>
    <div class="mt-4"><label class="block text-sm">Arm selection</label><select id="arm_choice" class="input-dark w-full"><option value="auto">Bandit chooses</option><option value="knn">Force KNN</option><option value="rf">Force Random Forest</option></select></div>
    <button onclick="runBandit()" class="btn-primary w-full py-3 rounded-lg mt-2">Get Recommendation</button></div>
    <div class="glass-panel p-6 space-y-4"><div class="flex justify-between"><span class="text-sm">Selected arm</span><span id="bandit-arm" class="text-secondary font-bold">--</span></div><div><span class="text-sm">Recommended item ID</span><div id="rec-item" class="text-2xl font-mono font-bold">--</div></div><div><span class="text-sm">Confidence</span> <span id="rec-conf">--</span></div><div id="bandit-note" class="text-xs text-on-surface-variant mt-2"></div></div></div></div>
</div></main>
<script>
    function showSection(section) {
        document.querySelectorAll('[id^="section-"]').forEach(el => el.classList.add('hidden'));
        document.getElementById(`section-${section}`).classList.remove('hidden');
        document.querySelectorAll('[id^="nav-"]').forEach(el => el.classList.remove('active-nav'));
        document.getElementById(`nav-${section}`).classList.add('active-nav');
    }
    const radioExisting = document.querySelector('input[value="existing"]');
    const radioNew = document.querySelector('input[value="new"]');
    function toggleSessionMode() { const isExisting = radioExisting.checked; document.getElementById('existing-fields').style.display = isExisting ? 'block' : 'none'; document.getElementById('new-fields').classList.toggle('hidden', isExisting); }
    radioExisting.addEventListener('change', toggleSessionMode); radioNew.addEventListener('change', toggleSessionMode); toggleSessionMode();
    async function runPathfinder() {
        const start = parseInt(document.getElementById('start_cat').value), goal = parseInt(document.getElementById('goal_cat').value);
        const res = await callAPI('/api/pathfinder', { start, goal });
        if (res.error) { document.getElementById('pathfinder-result').innerHTML = `<p class="text-error">Error: ${res.error}</p>`; return; }
        document.getElementById('pathfinder-result').innerHTML = `<p>Nodes explored: ${res.nodes_explored}</p><p>Path length: ${res.path_length}</p><p>Cost: ${res.cost}</p>`;
        document.getElementById('path-list').innerHTML = `<div class="space-y-1">${res.path.map((id,i)=>`<div>${i+1}. CAT_${id}</div>`).join('')}</div>`;
    }
    async function runPredictor() {
        const payload = {
            num_views: parseFloat(document.getElementById('views').value),
            num_addtocart: parseFloat(document.getElementById('addtocart').value),
            unique_items: parseFloat(document.getElementById('unique_items').value),
            categories_viewed: parseFloat(document.getElementById('categories').value),
            duration_min: parseFloat(document.getElementById('duration').value),
            cluster: parseFloat(document.getElementById('cluster').value)
        };
        const res = await callAPI('/api/predictor', payload);
        if (res.error) { alert(res.error); return; }
        document.getElementById('prob-value').innerHTML = `${res.probability}%`;
        document.getElementById('prob-bar').style.width = `${res.probability}%`;
        document.getElementById('risk-label').innerHTML = res.risk_label;
    }
    async function runBandit() {
        const useExisting = document.querySelector('input[name="session_mode"]:checked').value === 'existing';
        const forceArm = document.getElementById('arm_choice').value;
        let payload = { use_existing: useExisting, force_arm: forceArm === 'auto' ? null : forceArm };
        if (useExisting) {
            payload.visitor_id = parseInt(document.getElementById('visitor_id').value);
            payload.session_id = parseInt(document.getElementById('session_id').value);
        } else {
            payload.num_views = parseFloat(document.getElementById('new_views').value);
            payload.num_addtocart = parseFloat(document.getElementById('new_addtocart').value);
            payload.unique_items = parseFloat(document.getElementById('new_unique').value);
            payload.categories_viewed = parseFloat(document.getElementById('new_categories').value);
            payload.duration_min = parseFloat(document.getElementById('new_duration').value);
        }
        const res = await callAPI('/api/bandit/recommend', payload);
        if (res.error) { alert(res.error); return; }
        document.getElementById('bandit-arm').innerHTML = res.arm;
        document.getElementById('rec-item').innerHTML = res.recommended_item;
        document.getElementById('rec-conf').innerHTML = (res.confidence * 100).toFixed(1) + '%';
        document.getElementById('bandit-note').innerHTML = useExisting ? 'Based on session history and bandit decision' : 'New session – using most‑popular item and RF probability.';
    }
    showSection('pathfinder');
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)