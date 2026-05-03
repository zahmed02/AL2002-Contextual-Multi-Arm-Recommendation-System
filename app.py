import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import base64

# Add src to path so we can import custom modules
sys.path.append("src")

from product_graph import build_category_graph
from astar_product import a_star_product
from clustering import ClusterAssigner
from rf_predictor import PurchasePredictor
from knn_recommender import KNNRecommender
from contextual_bandit import LinUCB
from preprocess import get_context_vector

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="E‑commerce Intelligence System",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------------
# Load custom CSS
# -------------------------------------------------------------------
def load_css():
    css_file = Path("assets/styles.css")
    if css_file.exists():
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("CSS file not found. Using default theme.")

load_css()

# -------------------------------------------------------------------
# Cache data loading (unchanged)
# -------------------------------------------------------------------
@st.cache_resource
def load_all():
    RAW_DIR = Path("data/raw")
    PROC_DIR = Path("data/processed")
    MODEL_DIR = Path("models")

    parent_of, children_of, roots = build_category_graph(RAW_DIR)

    try:
        knn = KNNRecommender(MODEL_DIR)
    except Exception as e:
        st.warning(f"KNN recommender not available: {e}")
        knn = None

    try:
        cluster_assigner = ClusterAssigner(MODEL_DIR)
    except Exception as e:
        st.error(f"Clustering model not loaded: {e}")
        cluster_assigner = None

    try:
        predictor = PurchasePredictor(MODEL_DIR)
    except Exception as e:
        st.error(f"Random Forest model not loaded: {e}")
        predictor = None

    feature_importance = None
    if predictor is not None:
        try:
            import json
            with open(MODEL_DIR / "feature_importance.json", "r") as f:
                feature_importance = json.load(f)
        except:
            pass

    try:
        bandit = LinUCB.load(MODEL_DIR)
    except Exception as e:
        st.error(f"Bandit parameters not loaded: {e}")
        bandit = None

    try:
        scaler_context = joblib.load(MODEL_DIR / "scaler_context.pkl")
    except FileNotFoundError:
        st.error("Scaler context not found. Run the notebooks first.")
        scaler_context = None

    numeric_features = ['num_views', 'num_addtocart', 'unique_items',
                        'categories_viewed', 'duration_min']

    try:
        session_df = pd.read_parquet(PROC_DIR / "user_sessions.parquet")
        if 'cluster' not in session_df.columns:
            cluster_df = pd.read_csv(PROC_DIR / "user_clusters.csv")
            session_df = session_df.merge(cluster_df, on=['visitorid', 'session_id'])
    except Exception as e:
        st.error(f"Could not load session data: {e}")
        session_df = pd.DataFrame()

    return {
        "parent_of": parent_of,
        "children_of": children_of,
        "roots": roots,
        "knn": knn,
        "cluster_assigner": cluster_assigner,
        "predictor": predictor,
        "bandit": bandit,
        "scaler_context": scaler_context,
        "numeric_features": numeric_features,
        "session_df": session_df,
        "feature_importance": feature_importance
    }

data = load_all()

# -------------------------------------------------------------------
# Helper functions for visualisations (improved sizing)
# -------------------------------------------------------------------
def get_descendants(node, children_of):
    descendants = set()
    stack = list(children_of.get(node, []))
    while stack:
        child = stack.pop()
        if child in descendants:
            continue
        descendants.add(child)
        stack.extend(children_of.get(child, []))
    return sorted(descendants)

def plot_path_diagram(path):
    if not path:
        return
    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    ax.set_xlim(-0.5, len(path) - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')

    for i, node in enumerate(path):
        box = FancyBboxPatch((i - 0.4, -0.3), 0.8, 0.6,
                              boxstyle="round,pad=0.02",
                              facecolor="#f97316", edgecolor="white", linewidth=1.5)
        ax.add_patch(box)
        ax.text(i, 0, str(node), ha='center', va='center', color='white', fontsize=9, weight='bold')
        if i < len(path) - 1:
            ax.annotate("", xy=(i+0.4, 0), xytext=(i+0.6, 0),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
    ax.set_title("Optimal Category Path", color='white', fontsize=12)
    st.pyplot(fig)
    plt.close(fig)

def plot_probability_gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    theta = np.linspace(0, np.pi, 100)
    r = 1.0
    ax.bar(theta, r, width=0.02, color='#2c4f6e', alpha=0.3)
    fill_theta = np.linspace(0, prob * np.pi, 100)
    ax.fill_between(fill_theta, 0, r, alpha=0.7, color='#f97316')
    ax.set_ylim(0, 1.2)
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels(['0%', '50%', '100%'], color='white')
    ax.set_yticks([])
    ax.set_title("Purchase Probability", color='white', fontsize=10)
    st.pyplot(fig)
    plt.close(fig)

def plot_feature_importance():
    if data["feature_importance"]:
        features = list(data["feature_importance"].keys())
        importances = list(data["feature_importance"].values())
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#121212')
        bars = ax.barh(features, importances, color='#f97316', edgecolor='white')
        ax.set_xlabel("Importance", color='white')
        ax.set_title("Global Feature Importance (Random Forest)", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Feature importance data not available.")

def plot_bandit_ucb(bandit, context):
    n_arms = bandit.n_arms
    ucb_values = []
    theta_vals = []
    for arm in range(n_arms):
        A_inv = np.linalg.inv(bandit.A[arm])
        theta = A_inv @ bandit.b[arm]
        p = np.dot(theta, context)
        ucb = p + bandit.alpha * np.sqrt(context @ A_inv @ context)
        ucb_values.append(ucb)
        theta_vals.append(p)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    arms = ['KNN (arm0)', 'RF (arm1)']
    x = np.arange(len(arms))
    width = 0.35
    bars1 = ax.bar(x - width/2, theta_vals, width, label='Expected reward', color='#3b82f6')
    bars2 = ax.bar(x + width/2, ucb_values, width, label='UCB', color='#f97316', alpha=0.7)
    ax.set_ylabel('Value', color='white')
    ax.set_title('LinUCB Arm Evaluation', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(arms, color='white')
    ax.legend(facecolor='#0f2a3f', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    st.pyplot(fig)
    plt.close(fig)

# -------------------------------------------------------------------
# Fixed Header (similar to PathPulse)
# -------------------------------------------------------------------
def get_base64_emoji():
    # Simple transparent 1x1 pixel base64 for placeholder if needed
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

# Placeholder – you can replace with actual logo if you have one
logo_base64 = get_base64_emoji()

st.markdown(
    f"""
    <div class="custom-header">
        <span class="header-icon">🛒</span>
        <span class="header-title">E‑commerce Intelligence</span>
    </div>
    <div style="text-align: center; margin-bottom: 1.5rem; padding-top: 1rem;">
        <div class="liquid-glass-logo-container" style="margin: 0 auto;">
            <span style="font-size: 3rem;">🛒📊</span>
        </div>
        <p style="font-size: 1rem; color: #d1d5db; margin-top: 1rem;">
            A* search · Purchase prediction · Contextual Bandit orchestration
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 A* Product Search", "📈 Purchase Prediction", "🎯 Contextual Bandit"])

# =========================================================================
# TAB 1 – A* Search
# =========================================================================
with tab1:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 24px;">
            <h2>Optimal Category Pathfinding</h2>
            <p>Find the shortest route between product categories using A* search on the category tree.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1_ctrl, col1_plot = st.columns([1, 1.2])

    with col1_ctrl:
        if not data["parent_of"]:
            st.warning("Category data not loaded. Please check data/raw/category_tree.csv")
        else:
            all_cats = sorted(set(data["parent_of"].keys()) | set(data["children_of"].keys()))
            for r in data["roots"]:
                if r not in all_cats:
                    all_cats.append(r)
            all_cats = sorted(all_cats)

            start_cat = st.selectbox("Start category (root)", data["roots"], key="astar_start")
            descendants = get_descendants(start_cat, data["children_of"])
            if descendants:
                goal_options = [g for g in all_cats if g in descendants]
                if not goal_options:
                    goal_options = all_cats
            else:
                goal_options = all_cats
            goal_cat = st.selectbox("Goal category", goal_options, key="astar_goal")

            if st.button("Find Path", key="astar_btn", use_container_width=True):
                def simple_heuristic(node, goal):
                    return 0
                path, expanded = a_star_product(start_cat, goal_cat,
                                                data["parent_of"], data["children_of"],
                                                simple_heuristic)
                if path:
                    st.success(f"✅ Path found! Expanded {expanded} nodes.")
                    st.markdown(f"**Path:** {' → '.join(map(str, path))}")
                else:
                    st.error("❌ No path found between these categories.")

    with col1_plot:
        if 'path' in locals() and path:
            plot_path_diagram(path)

# =========================================================================
# TAB 2 – Purchase Prediction
# =========================================================================
with tab2:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 24px;">
            <h2>Purchase Probability Engine</h2>
            <p>Enter session metrics to predict the likelihood of a purchase using a trained Random Forest model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data["predictor"] is None:
        st.error("Random Forest model not available. Run notebook 03 first.")
    else:
        col_form, col_gauge = st.columns([1, 0.8])

        with col_form:
            with st.form("prediction_form"):
                colA, colB = st.columns(2)
                with colA:
                    num_views = st.number_input("Number of views", min_value=0, value=5)
                    num_addtocart = st.number_input("Add‑to‑carts", min_value=0, value=1)
                    unique_items = st.number_input("Unique items viewed", min_value=0, value=3)
                with colB:
                    categories_viewed = st.number_input("Categories explored", min_value=0, value=2)
                    duration_min = st.number_input("Session duration (minutes)", min_value=0.0, value=8.0)
                cluster = st.selectbox("User cluster (0–3)", [0,1,2,3])
                submitted = st.form_submit_button("Predict", use_container_width=True)

        if submitted:
            session_row = pd.Series({
                'num_views': num_views,
                'num_addtocart': num_addtocart,
                'unique_items': unique_items,
                'categories_viewed': categories_viewed,
                'duration_min': duration_min,
                'cluster': cluster
            })
            prob = data["predictor"].predict_probability(session_row)

            with col_gauge:
                plot_probability_gauge(prob)

            # Metric & risk message
            col_metric, _ = st.columns([1, 2])
            with col_metric:
                st.metric("Purchase Probability", f"{prob*100:.2f}%")
                if prob > 0.5:
                    st.warning("⚠️ High purchase likelihood – consider retargeting.")
                else:
                    st.info("✅ Low risk – general recommendations are safe.")

            # Feature importance (full width)
            st.markdown("<div class='glass-card' style='margin-top: 24px;'><h3>Model Insights</h3></div>", unsafe_allow_html=True)
            plot_feature_importance()

# =========================================================================
# TAB 3 – Contextual Bandit
# =========================================================================
with tab3:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 24px;">
            <h2>Live Recommendation Orchestration</h2>
            <p>LinUCB chooses between KNN (item similarity) and Random Forest (purchase likelihood) based on context.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data["bandit"] is None or data["predictor"] is None:
        st.error("Bandit or Random Forest model missing. Please run notebook 04.")
    else:
        @st.cache_resource
        def load_events():
            return pd.read_parquet("data/processed/events_with_sessions.parquet",
                                   columns=['visitorid', 'session_id', 'timestamp', 'event', 'itemid'])

        events_df = load_events()

        def get_user_items(user_id, session_id):
            session_events = events_df[
                (events_df['visitorid'] == user_id) & 
                (events_df['session_id'] == session_id)
            ].sort_values('timestamp')
            if session_events.empty:
                return []
            return session_events['itemid'].tolist()

        input_type = st.radio("Choose input method", ["Use existing session ID", "Create new session"], horizontal=True)

        if input_type == "Use existing session ID":
            if data["session_df"].empty:
                st.error("No session data loaded.")
            else:
                col_ctrl, col_bandit_plot = st.columns([1, 1.2])
                with col_ctrl:
                    user_id = st.number_input("Visitor ID", min_value=0, step=1, value=0)
                    session_id = st.number_input("Session ID", min_value=0, step=1, value=0)
                    if st.button("Get Recommendation", key="bandit_existing", use_container_width=True):
                        session = data["session_df"][(data["session_df"]["visitorid"] == user_id) &
                                                     (data["session_df"]["session_id"] == session_id)]
                        if session.empty:
                            st.error("Session not found. Please use a valid (visitorid, session_id) pair.")
                        else:
                            row = session.iloc[0]
                            context = get_context_vector(row, data["scaler_context"], data["numeric_features"])
                            arm = data["bandit"].select_arm(context)

                            with col_bandit_plot:
                                plot_bandit_ucb(data["bandit"], context)

                            user_items = get_user_items(user_id, session_id)

                            if arm == 0:
                                st.info("🎯 **Arm 0 selected: KNN**")
                                if data["knn"] is None:
                                    st.warning("KNN recommender not available.")
                                elif not user_items:
                                    st.warning("No items found for this session. Cannot recommend using KNN.")
                                else:
                                    recommended_items = data["knn"].recommend(user_items, top_n=3)
                                    if recommended_items:
                                        st.write("**Recommended items (similar to your last click):**")
                                        for rec in recommended_items:
                                            st.code(f"Item ID: {rec}", language="text")
                                    else:
                                        st.write("No similar items found.")
                            else:
                                st.info("📊 **Arm 1 selected: Random Forest**")
                                prob = data["predictor"].predict_probability(row)
                                st.metric("Predicted purchase probability", f"{prob*100:.2f}%")

        else:  # Create new session
            with st.form("new_session_form"):
                col1, col2 = st.columns(2)
                with col1:
                    nv = st.number_input("Views", 0, 100, 5)
                    nac = st.number_input("Add‑to‑carts", 0, 50, 1)
                    uitems = st.number_input("Unique items", 0, 50, 3)
                with col2:
                    cats = st.number_input("Categories viewed", 0, 50, 2)
                    dur = st.number_input("Duration (min)", 0.0, 300.0, 8.0)
                cluster = st.selectbox("Cluster", [0,1,2,3])
                submitted = st.form_submit_button("Recommend", use_container_width=True)

            if submitted:
                row = pd.Series({
                    'num_views': nv,
                    'num_addtocart': nac,
                    'unique_items': uitems,
                    'categories_viewed': cats,
                    'duration_min': dur,
                    'cluster': cluster
                })
                context = get_context_vector(row, data["scaler_context"], data["numeric_features"])
                arm = data["bandit"].select_arm(context)

                # Show UCB plot even for new sessions
                plot_bandit_ucb(data["bandit"], context)

                if arm == 0:
                    st.warning("KNN requires an existing session with item history. Cannot recommend for a new session.")
                    st.info("Try using a real session ID from the examples above.")
                else:
                    st.success("🧠 Bandit chooses **Random Forest**")
                    prob = data["predictor"].predict_probability(row)
                    st.metric("Purchase probability", f"{prob*100:.2f}%")

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown(
    """
    <hr style="margin-top: 48px;" />
    <p style="text-align: center; font-size: 0.75rem; color: #d1d5db; font-weight: 500;">
        E‑commerce Intelligence System — A* · Random Forest · LinUCB
    </p>
    """,
    unsafe_allow_html=True,
)