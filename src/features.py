"""
Phase 3 — Feature Engineering
Reads data/clean.csv and writes data/features.csv.

Engineered numeric features
    rank_diff      = Rank_1 - Rank_2
    rank_ratio     = Rank_1 / Rank_2
    log_rank_diff  = log1p(Rank_1) - log1p(Rank_2)

One-hot encoded (drop_first=True)
    Surface, Round, Series, Court, Best of

Reference columns kept but never used as model features
    Date, Player_1, Player_2
"""

import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
INPUT = ROOT / "data" / "clean.csv"
OUTPUT = ROOT / "data" / "features.csv"

# Columns that identify a match but must never enter the feature matrix
REFERENCE_COLS = ["Date", "Player_1", "Player_2"]

# Categorical columns to one-hot encode
CATEGORICAL_COLS = ["Surface", "Round", "Series", "Court", "Best of"]


def build_features(
    input_path: pathlib.Path = INPUT,
    output_path: pathlib.Path = OUTPUT,
) -> pd.DataFrame:
    """Engineer features from clean data and save to features.csv.
    Returns the feature DataFrame."""

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    df = pd.read_csv(input_path, parse_dates=["Date"])

    # ------------------------------------------------------------------
    # 2. Numeric rank features
    # ------------------------------------------------------------------
    df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
    df["rank_ratio"] = df["Rank_1"] / df["Rank_2"]
    df["log_rank_diff"] = np.log1p(df["Rank_1"]) - np.log1p(df["Rank_2"])

    # ------------------------------------------------------------------
    # 3. One-hot encode categorical columns
    # ------------------------------------------------------------------
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # ------------------------------------------------------------------
    # 4. Column ordering: reference cols | target | feature cols
    # ------------------------------------------------------------------
    non_feature = REFERENCE_COLS + ["player_1_wins"]
    feature_cols = [c for c in df.columns if c not in non_feature]
    df = df[non_feature + feature_cols]

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    string_cols = [c for c in df.columns if df[c].dtype == object and c not in REFERENCE_COLS]
    print(f"Shape          : {df.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"String cols left (excl. reference): {string_cols or 'none'}")
    print(f"Columns        : {df.columns.tolist()}")
    print(f"Output         : {output_path}")

    return df


if __name__ == "__main__":
    build_features()
