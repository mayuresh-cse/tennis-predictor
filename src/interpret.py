"""
Phase 5 — Model Interpretation
Produces two feature-importance charts from the trained Random Forest.

Outputs
    outputs/feature_importance.png       — built-in (impurity) importances
    outputs/permutation_importance.png   — permutation importances (val set, n_repeats=10)
"""

import pathlib
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "rf_model.joblib"
COLS_PATH  = ROOT / "models" / "feature_columns.txt"
FEATURES   = ROOT / "data" / "features.csv"
OUT_DIR    = ROOT / "outputs"

TOP_N = 20   # how many features to show in each chart


def _horizontal_bar(values, labels, title, xlabel, out_path, xerr=None):
    """Save a clean horizontal bar chart to *out_path*."""
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.38)))
    y = range(len(labels))
    ax.barh(y, values, xerr=xerr, align="center", color="#3b82d4",
            ecolor="#57606a", capsize=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()          # highest importance at top
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved : {out_path}")


def interpret(
    model_path: pathlib.Path = MODEL_PATH,
    cols_path:  pathlib.Path = COLS_PATH,
    features_path: pathlib.Path = FEATURES,
    out_dir:    pathlib.Path = OUT_DIR,
) -> None:
    """Run both importance methods and save charts."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model + feature list
    # ------------------------------------------------------------------
    clf = joblib.load(model_path)
    feature_cols = cols_path.read_text().splitlines()
    print(f"Model loaded   : {model_path}")
    print(f"Feature cols   : {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 2. Reconstruct 2025 validation split
    # ------------------------------------------------------------------
    df = pd.read_csv(features_path, parse_dates=["Date"])
    val = df[df["Date"].dt.year == 2025]
    X_val = val[feature_cols]
    y_val = val["player_1_wins"]
    print(f"Validation rows: {len(X_val):,}")

    # ------------------------------------------------------------------
    # 3. Built-in (impurity) importances
    # ------------------------------------------------------------------
    imp_series = (
        pd.Series(clf.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(TOP_N)
    )

    print(f"\n--- Built-in importances (top {TOP_N}) ---")
    print(imp_series.to_string())

    _horizontal_bar(
        values=imp_series.values,
        labels=imp_series.index.tolist(),
        title=f"Random Forest — Built-in Feature Importance (top {TOP_N})",
        xlabel="Mean decrease in impurity",
        out_path=out_dir / "feature_importance.png",
    )

    # ------------------------------------------------------------------
    # 4. Permutation importances
    # ------------------------------------------------------------------
    print(f"\nComputing permutation importances (n_repeats=10) ...")
    perm = permutation_importance(
        clf, X_val, y_val, n_repeats=10, random_state=42, n_jobs=1
    )

    perm_series = (
        pd.DataFrame({
            "mean": perm.importances_mean,
            "std":  perm.importances_std,
        }, index=feature_cols)
        .sort_values("mean", ascending=False)
        .head(TOP_N)
    )

    print(f"\n--- Permutation importances (top {TOP_N}) ---")
    print(perm_series.to_string())

    _horizontal_bar(
        values=perm_series["mean"].values,
        labels=perm_series.index.tolist(),
        title=f"Random Forest — Permutation Importance (top {TOP_N}, val 2025)",
        xlabel="Mean accuracy decrease",
        out_path=out_dir / "permutation_importance.png",
        xerr=perm_series["std"].values,
    )

    print("\nDone.")


if __name__ == "__main__":
    interpret()
