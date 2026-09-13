# Phase 4 Implementation Log — Model Building

**Date:** 2025-07-29  
**Phase:** 4 of 8  
**Status:** Complete

---

## Objectives

Train a `RandomForestClassifier` on historical ATP match data (2000–2024),
evaluate it on the held-out 2025 validation year, and serialise the model
and feature column list for reuse by Phases 5, 6, and 7.

---

## Pre-flight Inspection

Confirmed split sizes from `data/features.csv` before training:

| Split | Years | Rows |
|---|---|---|
| Train | 2000–2024 | 64,167 |
| Validation | 2025 | 2,487 |
| Holdout | 2026+ | 1,955 |
| **Total** | | **68,609** |

Feature columns confirmed: **24** (matches Phase 3 output exactly).

---

## Steps Executed

### 1. `src/train.py` written

Single `train()` function — runnable as a script and importable by other modules.

| Step | Action |
|---|---|
| 1 | Load `data/features.csv` with `parse_dates=["Date"]` |
| 2 | Define `feature_cols` = all columns except `Date`, `Player_1`, `Player_2`, `player_1_wins` |
| 3 | Chronological split: train `year <= 2024`, val `year == 2025` |
| 4 | Fit `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)` on train |
| 5 | Evaluate on val: `accuracy_score`, `classification_report`, `roc_auc_score` |
| 6 | Save model to `models/rf_model.joblib` via `joblib.dump` |
| 7 | Write feature column list to `models/feature_columns.txt` (one per line) |

---

## Validation Metrics (2025 hold-out year)

| Metric | Value |
|---|---|
| **Accuracy** | **61.1%** |
| **ROC-AUC** | **0.6568** |

### Classification Report

|  | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Player 2 wins | 0.61 | 0.64 | 0.62 | 1,244 |
| Player 1 wins | 0.62 | 0.58 | 0.60 | 1,243 |
| **Macro avg** | 0.61 | 0.61 | 0.61 | 2,487 |

### Interpretation

- **61.1% accuracy** is meaningfully above the 50% random baseline — the model
  learns real signal from rankings and match context without any betting odds or
  post-match information.
- **ROC-AUC 0.657** confirms the model's probability estimates have discriminative
  power beyond a coin-flip classifier.
- **Balanced precision/recall** across both classes confirms no systematic bias
  toward predicting one player over the other.

---

## Output Verification

| Check | Result |
|---|---|
| `models/rf_model.joblib` exists | Yes |
| Model type | `RandomForestClassifier` |
| `n_estimators` | 200 |
| `n_features_in_` | 24 (matches feature matrix exactly) |
| `classes_` | `[0, 1]` |
| `models/feature_columns.txt` exists | Yes |
| Lines in feature_columns.txt | 24 |
| Model file size | 559.9 MB |

---

## Deviations from Plan

None. All steps executed exactly as specified.

---

## Files Created / Modified

| File | Action | Purpose |
|---|---|---|
| `src/train.py` | Created | Model training module |
| `models/rf_model.joblib` | Created | Serialised Random Forest (559.9 MB) |
| `models/feature_columns.txt` | Created | 24 feature names, one per line |
| `phase_4_implementation_log.md` | Created | This log |

---

## Next Phases

Phases 5, 6, and 7 can all begin now (they depend only on Phase 4 outputs):

- **Phase 5** — Model Interpretation (`src/interpret.py`)
- **Phase 6** — Live Prediction Challenge (`src/predict.py`)
- **Phase 7** — Streamlit Web UI (`app.py`)
