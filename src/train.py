"""
Phase 4 — Model Building
Trains a Random Forest on the 2000-2024 data, evaluates on 2025,
and serialises the model + feature list for reuse.

Outputs
    models/rf_model.joblib       — fitted RandomForestClassifier
    models/feature_columns.txt   — one feature name per line
"""

import pathlib
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
INPUT = ROOT / "data" / "features.csv"
MODEL_OUT = ROOT / "models" / "rf_model.joblib"
COLS_OUT = ROOT / "models" / "feature_columns.txt"

# Columns present in features.csv that are NOT model inputs
NON_FEATURE_COLS = ["Date", "Player_1", "Player_2", "player_1_wins"]


def train(
    input_path: pathlib.Path = INPUT,
    model_out: pathlib.Path = MODEL_OUT,
    cols_out: pathlib.Path = COLS_OUT,
) -> RandomForestClassifier:
    """Train, evaluate and save the model. Returns the fitted classifier."""

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    df = pd.read_csv(input_path, parse_dates=["Date"])

    # ------------------------------------------------------------------
    # 2. Define feature columns
    # ------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # ------------------------------------------------------------------
    # 3. Chronological split
    # ------------------------------------------------------------------
    train_mask = df["Date"].dt.year <= 2024
    val_mask = df["Date"].dt.year == 2025
    # holdout (2026+) is set aside and not used in this phase

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "player_1_wins"]

    X_val = df.loc[val_mask, feature_cols]
    y_val = df.loc[val_mask, "player_1_wins"]

    print(f"Train rows : {len(X_train):,}  (years 2000-2024)")
    print(f"Val rows   : {len(X_val):,}  (year 2025)")

    # ------------------------------------------------------------------
    # 4. Fit
    # ------------------------------------------------------------------
    print("\nFitting RandomForestClassifier(n_estimators=200, random_state=42) ...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Done.")

    # ------------------------------------------------------------------
    # 5. Evaluate on validation set
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_proba)

    print("\n--- Validation metrics (2025) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=["Player2 wins", "Player1 wins"]))

    # ------------------------------------------------------------------
    # 6. Save model
    # ------------------------------------------------------------------
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"Model saved : {model_out}")

    # ------------------------------------------------------------------
    # 7. Save feature column list
    # ------------------------------------------------------------------
    cols_out.parent.mkdir(parents=True, exist_ok=True)
    cols_out.write_text("\n".join(feature_cols))
    print(f"Feature cols: {cols_out}  ({len(feature_cols)} columns)")

    return clf


if __name__ == "__main__":
    train()
