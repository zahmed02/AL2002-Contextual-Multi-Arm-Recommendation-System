#!/usr/bin/env python3
"""
download_assets.py
------------------
Downloads pre-trained model weights and pre-processed datasets from the
GitHub Release so you don't need to run the Jupyter notebooks locally.

Usage:
    python download_assets.py              # download everything that is missing
    python download_assets.py --force      # re-download even if files exist already

After running this once, simply start the app with:
    python app.py
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Release configuration — update RELEASE_TAG if a new release is published
# ---------------------------------------------------------------------------
REPO        = "zahmed02/AL2002-Contextual-Multi-Arm-Recommendation-System"
RELEASE_TAG = "v1.0.0"
RELEASE_BASE = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

# (relative_path_from_project_root, approximate_size_in_bytes)
ASSETS = [
    # Trained model weights
    ("models/item_similarity.pkl",                  200_080_747),  # 191 MB — KNN similarity matrix
    ("models/kmeans_model.pkl",                       7_047_879),  #   7 MB — K-Means clustering
    ("models/random_forest.pkl",                      3_072_041),  #   3 MB — Random Forest classifier
    # Pre-processed datasets
    ("data/processed/events_with_sessions.parquet",  65_471_602),  #  62 MB
    ("data/processed/user_sessions.parquet",         37_496_583),  #  36 MB
    ("data/processed/user_clusters.parquet",          7_489_084),  #   7 MB
    ("data/processed/user_clusters.csv",             21_555_552),  #  21 MB — fallback CSV
    ("data/processed/item_category_mapping.csv",      5_072_162),  #   5 MB
]

BASE_DIR = Path(__file__).parent


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    mb_done    = downloaded / 1_048_576
    if total_size > 0:
        pct    = min(100, downloaded * 100 // total_size)
        mb_tot = total_size / 1_048_576
        filled = pct // 2
        bar    = "#" * filled + "-" * (50 - filled)
        print(f"\r    [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_tot:.1f} MB", end="", flush=True)
    else:
        print(f"\r    Downloaded {mb_done:.1f} MB", end="", flush=True)


def download_file(rel_path: str, size_bytes: int, force: bool = False) -> bool:
    dest = BASE_DIR / rel_path
    if dest.exists() and not force:
        print(f"  [OK] Already exists - skipping:  {rel_path}")
        return True

    # File name on the Release page is just the basename (flat structure)
    filename = Path(rel_path).name
    url      = f"{RELEASE_BASE}/{filename}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    size_mb = size_bytes / 1_048_576
    print(f"  [DOWNLOAD] {rel_path}  (~{size_mb:.0f} MB)")
    try:
        urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after the progress bar
        return True
    except URLError as exc:
        print(f"\n  [ERROR] Download failed: {exc}")
        if dest.exists():
            dest.unlink()   # remove partial file
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pre-trained assets for the recommendation system."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Asset Downloader")
    print(f"  Release : {RELEASE_BASE}")
    print("=" * 60)
    print()

    all_ok = True
    for rel_path, size in ASSETS:
        ok = download_file(rel_path, size, force=args.force)
        if not ok:
            all_ok = False
        print()

    if all_ok:
        print("[SUCCESS] All assets ready. Run the app with:  python app.py")
    else:
        print("[WARNING] Some downloads failed. Check your internet connection or")
        print(f"    visit the release page manually:\n    https://github.com/{REPO}/releases/tag/{RELEASE_TAG}")
        sys.exit(1)


if __name__ == "__main__":
    main()
