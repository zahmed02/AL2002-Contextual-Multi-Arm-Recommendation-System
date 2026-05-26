# Contextual Multi‑Arm Recommendation System

## Course: AL2002 Artificial Intelligence Lab
## Instructor: Sir Abdullah Shaikh

## Overview

This project implements an intelligent e‑commerce recommendation system that combines:

- **A* Search**: optimal path finding through a product category tree.
- **Random Forest Classifier**: predicts the probability that a user will purchase during a session.
- **Contextual Bandit (LinUCB)**: dynamically selects the best recommendation strategy (KNN cooperative filtering vs. Random Forest) to maximise user engagement.

The system is deployed as an interactive **Flask** web application with a modern Tailwind CSS interface.

## Features

- **Product Category Pathfinder**: A* search to find the shortest path from a start category to a goal category in the product tree.
- **Purchase Predictor**: Enter session behavioural features and receive a purchase probability (and risk label) from a trained Random Forest model.
- **Bandit Recommender**: Live recommendation engine that chooses between:
  - **KNN arm**: recommends items similar to the user’s last interaction.
  - **Random Forest arm**: recommends the most frequent item in the session and uses the RF purchase probability as confidence.
- The bandit learns from user feedback (purchase or not) and updates its internal LinUCB matrices.

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/zahmed02/AL2002-Contextual-Multi-Arm-Recommendation-System.git
cd AL2002-Contextual-Multi-Arm-Recommendation-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download pre-trained assets
Large model weights and processed datasets are hosted on the [GitHub Releases page](https://github.com/zahmed02/AL2002-Contextual-Multi-Arm-Recommendation-System/releases/tag/v1.0.0) (~340 MB total). Download them with the included script:
```bash
python download_assets.py
```
Files that already exist are skipped automatically. Use `--force` to re-download everything.

### 4. Launch the Flask application
```bash
python app.py
```
The app will be available at **http://localhost:5000**.

> If you forget to run step 3, `app.py` will detect the missing files and
> print exactly which ones are absent along with a reminder to run
> `python download_assets.py`.

---

### (Optional) Re-generating models from scratch
If you want to modify the data pipeline or retrain the models:
- Download the [Kaggle RetailRocket E‑commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).
- Place the four CSV files into `data/raw/`:
  `category_tree.csv`, `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`.
- Run the Jupyter notebooks in `notebooks/` in numerical order (01 → 04).
- Restart `python app.py` — it will pick up the freshly generated files automatically.

## Project Structure

```
.
├── app.py                        # Main Flask application (Tailwind UI)
├── download_assets.py            # Downloads pre-trained assets from GitHub Release
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/                      # Original Kaggle CSVs (git-ignored, ~1 GB — optional)
│   └── processed/                # Pre-processed parquet/CSV files (via download_assets.py)
│
├── models/                       # Trained model weights (via download_assets.py)
│   ├── scaler.pkl                # Committed directly (< 1 KB)
│   ├── scaler_context.pkl        # Committed directly (< 1 KB)
│   ├── linucb_params.json        # Committed directly
│   ├── rf_metrics.json           # Committed directly
│   ├── feature_importance.json   # Committed directly
│   ├── item_similarity.pkl       # Downloaded (~191 MB)
│   ├── kmeans_model.pkl          # Downloaded (~7 MB)
│   └── random_forest.pkl         # Downloaded (~3 MB)
│
├── notebooks/                    # Jupyter notebooks (EDA, clustering, RF, bandit)
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Clustering_KMeans.ipynb
│   ├── 03_RandomForest_PurchasePrediction.ipynb
│   ├── 04_ContextualBandit_Orchestration.ipynb
│   └── 05_Testing_Values.ipynb
│
├── outputs/                      # Figures and test examples (CSV outputs)
│   ├── figures/                  # Generated plots (PNG)
│   └── *.csv                     # Example input files
│
├── src/                          # Reusable Python modules
│   ├── astar_product.py          # A* search on category tree
│   ├── clustering.py             # K‑Means cluster assignment
│   ├── contextual_bandit.py      # LinUCB bandit implementation
│   ├── knn_recommender.py        # Item‑based KNN recommender
│   ├── preprocess.py             # Feature scaling & context vector
│   ├── product_graph.py          # Build parent/children dictionaries
│   └── rf_predictor.py           # Random Forest purchase predictor
```

## Mechanics behind the features

### A* Search (Pathfinder)
- Builds a graph from `category_tree.csv` (parent‑child relationships).
- Given a start category ID and a goal category ID, the algorithm expands both children and parents (bidirectional movement) to find the shortest path.
- The current heuristic is zero, making the search equivalent to Dijkstra: it finds the minimum number of edges.

### Random Forest Purchase Predictor
- Trained on session features: `num_views`, `num_addtocart`, `unique_items`, `categories_viewed`, `duration_min` and user cluster labels (0–3).
- Outputs a purchase probability between 0 and 1.

### Contextual Bandit (LinUCB)
Two recommendation arms:

| Arm | Strategy | Details |
|-----|----------|---------|
| 0   | **KNN**  | Recommends the item most similar (cosine similarity) to the user’s last clicked item, based on a pre‑computed item‑item similarity matrix (weighted by event type: view=1, add‑to‑cart=3, transaction=5). |
| 1   | **Random Forest** | Recommends the most frequently viewed item in the session; confidence is the RF purchase probability. |

- The bandit maintains linear models (A matrices and b vectors) for each arm.
- Context vector = 5 scaled numeric features + the integer cluster label (0–3).
- For each request, LinUCB selects the arm with the highest upper confidence bound (UCB).
- After the user makes a purchase (or not), the reward (1 or 0) is used to update the corresponding arm’s model.

## Performance Metrics

| Model                | Accuracy | ROC‑AUC | Precision (class 1) | Recall (class 1) | F1 (class 1) |
|----------------------|----------|---------|---------------------|------------------|---------------|
| Random Forest        | 0.9855   | 0.9967  | 0.3498              | 0.9513           | 0.5115        |

- The Random Forest achieves **very high recall** (95.1%): excellent at identifying purchase sessions, at the cost of moderate precision (35%).
- The ROC‑AUC of 0.9967 indicates outstanding ranking ability.

## Tech Stack

- **Python** 3.9+
- **Data handling:** Pandas, NumPy
- **Machine learning:** Scikit‑learn (Random Forest, K‑Means, StandardScaler)
- **Bandit algorithm:** LinUCB (custom implementation)
- **Web framework:** Flask
- **Frontend:** HTML, Tailwind CSS (CDN), JavaScript
- **Visualisation:** Matplotlib, Seaborn
