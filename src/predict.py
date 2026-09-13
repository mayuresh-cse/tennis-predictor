"""
Phase 6 — Live Prediction Challenge
Runs the trained model against the held-out 2026 matches and compares
predictions to actual outcomes.

Outputs
    outputs/holdout_predictions.csv  — per-match predictions with correctness flag
"""

import pathlib
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = pathlib.Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "rf_model.joblib"
COLS_PATH  = ROOT / "models" / "feature_columns.txt"
FEATURES   = ROOT / "data" / "features.csv"
OUT_DIR    = ROOT / "outputs"
PREDS_OUT  = OUT_DIR / "holdout_predictions.csv"


def predict(
    model_path: pathlib.Path = MODEL_PATH,
    cols_path:  pathlib.Path = COLS_PATH,
    features_path: pathlib.Path = FEATURES,
    out_path:   pathlib.Path = PREDS_OUT,
) -> pd.DataFrame:
    """Run holdout predictions and save results.  Returns the results DataFrame."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model + feature list
    # ------------------------------------------------------------------
    clf = joblib.load(model_path)
    feature_cols = cols_path.read_text().splitlines()
    print(f"Model loaded  : {model_path}")
    print(f"Feature cols  : {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 2. Load features.csv and filter to holdout split (2026+)
    # ------------------------------------------------------------------
    df = pd.read_csv(features_path, parse_dates=["Date"])
    holdout = df[df["Date"].dt.year >= 2026].copy()
    print(f"Holdout rows  : {len(holdout):,}  (year >= 2026)")

    X_holdout = holdout[feature_cols]
    y_holdout = holdout["player_1_wins"]

    # ------------------------------------------------------------------
    # 3. Predict
    # ------------------------------------------------------------------
    proba = clf.predict_proba(X_holdout)[:, 1]   # P(player_1_wins)

    # ------------------------------------------------------------------
    # 4. Derive predicted and actual winner labels
    # ------------------------------------------------------------------
    predicted_winner = holdout["Player_1"].where(proba >= 0.5, holdout["Player_2"])
    actual_winner    = holdout["Player_1"].where(y_holdout == 1, holdout["Player_2"])
    correct          = predicted_winner == actual_winner

    # ------------------------------------------------------------------
    # 5. Assemble output DataFrame
    # ------------------------------------------------------------------
    results = pd.DataFrame({
        "Date":               holdout["Date"].dt.date,
        "Player_1":           holdout["Player_1"].values,
        "Player_2":           holdout["Player_2"].values,
        "prob_player_1_wins": proba.round(4),
        "predicted_winner":   predicted_winner.values,
        "actual_winner":      actual_winner.values,
        "correct":            correct.values,
    })

    # ------------------------------------------------------------------
    # 6. Metrics
    # ------------------------------------------------------------------
    acc = accuracy_score(y_holdout, (proba >= 0.5).astype(int))
    roc = roc_auc_score(y_holdout, proba)

    print("\n--- Holdout metrics (2026) ---")
    print(f"Accuracy : {acc:.4f}  ({correct.sum():,} / {len(correct):,} correct)")
    print(f"ROC-AUC  : {roc:.4f}")

    # ------------------------------------------------------------------
    # 7. Save CSV
    # ------------------------------------------------------------------
    results.to_csv(out_path, index=False)
    print(f"\nSaved     : {out_path}")

    # ------------------------------------------------------------------
    # 8. Sample output
    # ------------------------------------------------------------------
    print("\n--- Sample predictions (10 rows) ---")
    print(results.head(10).to_string(index=False))

    return results


if __name__ == "__main__":
    predict()
