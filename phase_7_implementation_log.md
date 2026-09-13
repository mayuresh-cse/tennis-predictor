# Phase 7 Implementation Log — Streamlit Web UI

**Date:** 2025-07-29  
**Phase:** 7 of 8  
**Status:** Complete

---

## Objectives

Provide a browser UI where a user selects two ATP players and optional match
context, then sees the predicted win probability for each player.

---

## Steps Executed

### 1. `app.py` written

| Step | Action |
|---|---|
| 1 | `@st.cache_data` — load `data/clean.csv` once for player list + ranking lookup |
| 2 | `@st.cache_resource` — load `models/rf_model.joblib` once |
| 3 | `@st.cache_data` — load `models/feature_columns.txt` once |
| 4 | Build `player_ranking_lookup` dict: most-recent rank per player across both Player_1 and Player_2 columns |
| 5 | Two-column player selector (`st.selectbox`) with auto-populated latest-ranking `st.number_input` |
| 6 | Three-column match context: Surface, Court, Round, Series, Best of |
| 7 | `build_feature_row()` helper: constructs a single-row DataFrame matching the exact feature matrix the model was trained on |
| 8 | On "Predict" click: call `model.predict_proba`, display `st.metric` + `st.progress` bars per player, announce predicted winner |
| 9 | "Match details used" expander for transparency |
| 10 | "How it works" expander explaining model, features, and accuracy |
| 11 | `TESTING=true` environment guard so `import app` works in pytest without a Streamlit server |

---

## Feature Row Construction

`build_feature_row()` replicates `features.py`'s one-hot encoding at prediction
time without re-running pandas `get_dummies` on the full dataset:

1. Compute numeric rank features (`rank_diff`, `rank_ratio`, `log_rank_diff`).
2. Initialise all dummy columns from `feature_columns.txt` to `False`.
3. Set the active dummy (e.g. `Surface_Clay`) to `True` based on selected context.
4. Return a single-row DataFrame in the exact column order the model expects.

**Key design note:** `drop_first=True` means the baseline category for each
variable is absent from the feature list — e.g. `Surface_Carpet` (alphabetical
first) is the implicit baseline; selecting "Carpet" leaves all Surface dummies
`False`, which is correct.

---

## Validation (smoke test)

| Check | Result |
|---|---|
| `app.py` imports cleanly under `TESTING=true` | ✅ |
| `build_feature_row` shape | `(1, 24)` ✅ |
| Rank feature values correct | ✅ |
| All active dummies set to `True` | ✅ |
| `model.predict_proba` round-trip | ✅ (`proba[1]=0.6963` for rank-10 vs rank-50 on Clay/QF/GS) |
| Players in ranking lookup | 1,816 ✅ |

---

## UI Layout

```
🎾 ATP Tennis Match Predictor
─────────────────────────────
[Player 1 col]          [Player 2 col]
 selectbox               selectbox
 number_input (rank)     number_input (rank)
─────────────────────────────
Match Context
[Surface | Court]  [Round]  [Series | Best of]
─────────────────────────────
[ 🎾 Predict  (full-width primary button) ]
─────────────────────────────
Prediction (after click)
[Player 1 metric + progress]  [Player 2 metric + progress]
🏆 Predicted winner: <name>
▼ Match details used
─────────────────────────────
▼ ℹ️ How it works
```

---

## Deviations from Plan

| Item | Plan | Actual | Reason |
|---|---|---|---|
| `TESTING` guard | Mentioned as optional | Implemented | Required for pytest `import app` smoke test in Phase 8 |

---

## Files Created / Modified

| File | Action | Purpose |
|---|---|---|
| `app.py` | Created | Streamlit web UI |
| `phase_7_implementation_log.md` | Created | This log |

---

## Run Command

```bash
streamlit run app.py
```

---

## Next Phase

**Phase 8 — Testing & Validation** (`tests/test_pipeline.py`)  
Run with: `pytest tests/ -v`
