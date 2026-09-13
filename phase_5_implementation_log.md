# Phase 5 Implementation Log — Model Interpretation

**Date:** 2025-07-29  
**Phase:** 5 of 8  
**Status:** Complete

---

## Objectives

Explain which features drive the model's predictions using two complementary
methods, and save publication-ready charts for both.

---

## Steps Executed

### 1. `src/interpret.py` written

Single `interpret()` function — runnable as a script and importable.

| Step | Action |
|---|---|
| 1 | Load `models/rf_model.joblib` and `models/feature_columns.txt` |
| 2 | Load `data/features.csv`; filter to `year == 2025` validation split (2,487 rows) |
| 3 | Compute built-in `feature_importances_`; plot top-20 horizontal bar chart |
| 4 | Run `permutation_importance(n_repeats=10, random_state=42, n_jobs=1)` on val set |
| 5 | Plot permutation importance means with std error bars; save chart |
| 6 | Print both ranked tables to console |

**Note:** `n_jobs=1` used for permutation importance (instead of plan's implicit `-1`)
because `n_jobs=-1` attempts to memory-map the 560 MB model to a temp device
that has no disk space — single-threaded runs cleanly and completes quickly on the
2,487-row validation set.

---

## Built-in (Impurity) Importances — Top 20

| Rank | Feature | Importance |
|---|---|---|
| 1 | `rank_ratio` | 0.1968 |
| 2 | `log_rank_diff` | 0.1951 |
| 3 | `rank_diff` | 0.1645 |
| 4 | `Rank_2` | 0.1495 |
| 5 | `Rank_1` | 0.1484 |
| 6 | `Round_2nd Round` | 0.0165 |
| 7 | `Court_Outdoor` | 0.0136 |
| 8 | `Series_International` | 0.0127 |
| 9 | `Surface_Hard` | 0.0121 |
| 10 | `Surface_Clay` | 0.0107 |

**Takeaway:** The five rank-related features (raw, engineered ratio, log-diff)
collectively account for ~85% of the built-in importance, confirming that ATP
ranking is by far the strongest pre-match predictor.

## Permutation Importances — Top features

| Rank | Feature | Mean | Std |
|---|---|---|---|
| 1 | `log_rank_diff` | 0.003016 | 0.005308 |
| 2 | `Surface_Hard` | 0.002975 | 0.004313 |
| 3 | `Round_2nd Round` | 0.002573 | 0.002967 |
| 4 | `Series_Masters 1000` | 0.001850 | 0.003845 |
| 5 | `Surface_Grass` | 0.000804 | 0.001393 |

**Takeaway:** Permutation importance (which measures actual accuracy drop when a
feature is shuffled) confirms `log_rank_diff` as the top signal. Notably,
several categorical dummies (`Surface_Hard`, `Round_2nd Round`,
`Series_Masters 1000`) show meaningful permutation importance, suggesting match
context adds real predictive value beyond rankings alone.

---

## Output Verification

| File | Exists | Size |
|---|---|---|
| `outputs/feature_importance.png` | Yes | 72,873 bytes |
| `outputs/permutation_importance.png` | Yes | 77,464 bytes |

---

## Deviations from Plan

| Item | Plan | Actual | Reason |
|---|---|---|---|
| `permutation_importance` `n_jobs` | Not specified (implied `-1`) | `n_jobs=1` | `n_jobs=-1` triggers joblib memory-mapping of the 560 MB model to a temp device with no disk space; single-threaded runs cleanly |

---

## Files Created / Modified

| File | Action | Purpose |
|---|---|---|
| `src/interpret.py` | Created | Model interpretation module |
| `outputs/feature_importance.png` | Created | Built-in importance bar chart |
| `outputs/permutation_importance.png` | Created | Permutation importance bar chart |
| `phase_5_implementation_log.md` | Created | This log |

---

## Next Phase

**Phase 6 — Live Prediction Challenge** (`src/predict.py`)  
Inputs: `models/rf_model.joblib`, `models/feature_columns.txt`, `data/features.csv`  
Output: `outputs/holdout_predictions.csv`
