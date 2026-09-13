# Phase 6 Implementation Log — Live Prediction Challenge

**Date:** 2025-07-29  
**Phase:** 6 of 8  
**Status:** Complete

---

## Objectives

Simulate a "live" prediction scenario by running the trained model against the
held-out 2026 matches and comparing predictions to actual outcomes.

---

## Steps Executed

### 1. `src/predict.py` written

Single `predict()` function — runnable as a script and importable.

| Step | Action |
|---|---|
| 1 | Load `models/rf_model.joblib` and `models/feature_columns.txt` |
| 2 | Load `data/features.csv`; filter to `year >= 2026` holdout split (1,955 rows) |
| 3 | Run `model.predict_proba` on holdout feature matrix |
| 4 | Derive `predicted_winner`: `Player_1` if `prob >= 0.5`, else `Player_2` |
| 5 | Derive `actual_winner` from `player_1_wins` column |
| 6 | Compute `correct = (predicted_winner == actual_winner)` |
| 7 | Save 7-column CSV to `outputs/holdout_predictions.csv` |
| 8 | Print holdout accuracy, ROC-AUC, and 10-row sample |

---

## Holdout Metrics (2026)

| Metric | Value |
|---|---|
| **Accuracy** | **0.6113** (1,195 / 1,955 correct) |
| **ROC-AUC** | **0.6456** |
| Holdout rows | 1,955 |
| Date range | 2026-01-04 → 2026-08-29 |
| Target balance | 978 Player1 wins / 977 Player2 wins (near-perfect 50/50) |

**Takeaway:** 61.1% accuracy on truly unseen 2026 matches, comfortably above
the 50% random baseline and matching the 2025 validation accuracy seen in
Phase 4 — confirming that the model generalises well across years.

---

## Sample Predictions

```
      Date         Player_1     Player_2  prob_p1  predicted_winner  actual_winner  correct
2026-01-04        Tiafoe F.     Vukic A.   0.7400         Tiafoe F.      Tiafoe F.     True
2026-01-04 Ugo Carabelli C.      Tien L.   0.3224           Tien L.        Tien L.     True
2026-01-05      Altmaier D. Majchrzak K.   0.3250      Majchrzak K.   Majchrzak K.     True
2026-01-05     Duckworth J. Michelsen A.   0.3813      Michelsen A.   Michelsen A.     True
2026-01-05         Halys Q.   Popyrin A.   0.3641        Popyrin A.       Halys Q.    False
2026-01-05         Korda S.  Vacherot V.   0.3522       Vacherot V.       Korda S.    False
2026-01-05       Lehecka J.    Machac T.   0.7700        Lehecka J.     Lehecka J.     True
2026-01-05      Medvedev D. Fucsovics M.   0.5843       Medvedev D.    Medvedev D.     True
2026-01-05        Sweeny D.    Opelka R.   0.4600         Opelka R.      Opelka R.     True
2026-01-05        Borges N.   Dzumhur D.   0.3000        Dzumhur D.      Borges N.    False
```

---

## Output Verification

| File | Exists | Shape | Null count |
|---|---|---|---|
| `outputs/holdout_predictions.csv` | Yes | 1,955 rows × 7 columns | 0 nulls |

**Columns:** `Date`, `Player_1`, `Player_2`, `prob_player_1_wins`, `predicted_winner`, `actual_winner`, `correct`

---

## Deviations from Plan

None — implementation matches the plan exactly.

---

## Files Created / Modified

| File | Action | Purpose |
|---|---|---|
| `src/predict.py` | Created | Holdout prediction module |
| `outputs/holdout_predictions.csv` | Created | Per-match predictions with correctness flag |
| `phase_6_implementation_log.md` | Created | This log |

---

## Next Phase

**Phase 7 — Streamlit Web UI** (`app.py`)  
Inputs: `data/clean.csv`, `models/rf_model.joblib`, `models/feature_columns.txt`  
Run command: `streamlit run app.py`
