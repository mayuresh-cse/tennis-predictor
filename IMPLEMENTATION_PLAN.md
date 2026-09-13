# ATP Tennis Match Predictor — Implementation Plan

## Top-Level Overview

Build a machine-learning web application that predicts the win probability for a
match between two ATP players. The model is trained on historical match data
(`atp_tennis.csv`, 68 636 rows, 2000–2026) using only pre-match contextual
features (rankings, surface, round, series, court type, format). Betting odds
and post-match columns are excluded to prevent data leakage. The Streamlit web
UI lets a user choose two players from a dropdown and immediately see predicted
win probabilities.

---

## Dataset Summary

| Column | Type | Role |
|---|---|---|
| `Tournament` | string | **Dropped** — replaced by `Series` |
| `Date` | date string | Chronological split anchor |
| `Series` | string | Tournament tier — kept as feature |
| `Court` | string | Indoor / Outdoor — kept as feature |
| `Surface` | string | Hard / Clay / Grass / Carpet — kept as feature |
| `Round` | string | Match round — kept as feature |
| `Best of` | integer | 3 or 5 — kept as feature |
| `Player_1` | string | **UI only** — never a model feature |
| `Player_2` | string | **UI only** — never a model feature |
| `Winner` | string | Used only to derive target, then dropped |
| `Rank_1` | integer | Player 1 ATP ranking — kept as feature |
| `Rank_2` | integer | Player 2 ATP ranking — kept as feature |
| `Pts_1` | integer | **Dropped** — sentinel `-1` in early years |
| `Pts_2` | integer | **Dropped** — sentinel `-1` in early years |
| `Odd_1` | float | **Dropped** — data leakage (betting odds) |
| `Odd_2` | float | **Dropped** — data leakage (betting odds) |
| `Score` | string | **Dropped** — post-match information |

**Target variable:** `player_1_wins = 1 if Winner == Player_1 else 0`

**Chronological splits:**

| Split | Years | Purpose |
|---|---|---|
| Train | 2000 – 2024 | Model fitting |
| Validation | 2025 | Hyperparameter / threshold tuning |
| Holdout | 2026 | Live prediction challenge |

---

## Recommended Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Wide library support, workshop standard |
| Data handling | pandas 2.x | Tabular data manipulation |
| ML | scikit-learn 1.4+ | Random Forest, metrics, permutation importance |
| Model serialisation | joblib (bundled with scikit-learn) | Save/load trained model |
| Model interpretation | scikit-learn built-in + `sklearn.inspection.permutation_importance` | No extra dependency |
| Visualisation | matplotlib + seaborn | Feature importance bar charts |
| Web UI | **Streamlit** | Minimal boilerplate, runs with `streamlit run app.py`, ideal for workshops |
| UI alternative considered | Gradio — comparable simplicity but less flexible layout control |
| Dependency management | `venv` + `requirements.txt` | Beginner-friendly, no extra tooling |
| Testing | `pytest` (lightweight smoke tests only) | Verify pipeline outputs |

---

## Project Structure

```
tennis-predictor/
├── atp_tennis.csv            # Source dataset (already present)
├── data/
│   ├── clean.csv             # Output of Phase 2 (prepare.py)
│   └── features.csv          # Output of Phase 3 (features.py)
├── models/
│   └── rf_model.joblib       # Serialised Random Forest (Phase 4)
├── outputs/
│   ├── feature_importance.png
│   ├── permutation_importance.png
│   └── holdout_predictions.csv
├── src/
│   ├── prepare.py            # Phase 2 — data cleaning
│   ├── features.py           # Phase 3 — feature engineering
│   ├── train.py              # Phase 4 — model training + metrics
│   ├── interpret.py          # Phase 5 — feature importance plots
│   └── predict.py            # Phase 6 — holdout prediction challenge
├── tests/
│   └── test_pipeline.py      # Phase 8 — smoke tests
├── app.py                    # Phase 7 — Streamlit web UI
├── requirements.txt
└── README.md
```

---

## Assumptions & Decisions

- **Historical data only.** There is no real-time ATP feed. The "live prediction
  challenge" uses the held-out 2026 rows from the same CSV, treating them as
  unseen matches.
- **Pts_1 / Pts_2 excluded.** Sentinel value `-1` appears across a large block
  of early-era rows; `Rank_1` / `Rank_2` are fully populated and sufficient.
- **Tournament dropped.** `Series` already encodes the tier
  (Grand Slam, Masters 1000, International, etc.).
- **Betting odds excluded.** `Odd_1` / `Odd_2` encode post-market information
  and would artificially inflate model accuracy.
- **Player names never enter the feature matrix.** They are used only to
  populate the Streamlit dropdowns and to look up a player's latest ranking
  from the dataset.
- **Random state fixed at 42** throughout for reproducibility.
- **Class balance.** Because `Player_1` is not systematically the favourite,
  the target is expected to be roughly 50/50 — no resampling needed.

---

## Risks & Trade-offs

### Risk 1 — Data Leakage
**Problem:** `Odd_1`, `Odd_2`, and `Score` are derived from or reflect the
match outcome. Including them would produce unrealistically high accuracy.
**Mitigation:** Drop all three columns unconditionally in `prepare.py` before
any feature is constructed.

### Risk 2 — Random vs. Chronological Train/Test Split
**Problem:** A random split allows future matches to appear in the training set,
leaking temporal information and inflating test accuracy.
**Mitigation:** All splits are strictly time-based using the `Date` column:
train <= 2024, validation = 2025, holdout = 2026.

### Risk 3 — High-Cardinality Categorical Columns
**Problem:** `Tournament` has hundreds of unique values; one-hot encoding it
would create a very wide, sparse matrix and risk overfitting.
**Mitigation:** Drop `Tournament`; use `Series` (approximately 6 unique values) instead.
`Surface`, `Round`, `Court`, and `Series` are all low-cardinality and safe to
one-hot encode.

---

## Phase 1 — Project Setup

**Intent:** Create the scaffolding every subsequent phase depends on — folder
structure, virtual environment, pinned dependencies, and the dataset in place.

**Expected Outcomes:**
- All directories (`data/`, `models/`, `outputs/`, `src/`, `tests/`) exist.
- `requirements.txt` is present with pinned library versions.
- `README.md` contains a one-page project description and quickstart.
- Running `pip install -r requirements.txt` in a fresh venv succeeds.

**Todo List:**
- [ ] Create directory structure: `data/`, `models/`, `outputs/`, `src/`, `tests/`
- [ ] Write `requirements.txt` with: `pandas`, `scikit-learn`, `matplotlib`,
      `seaborn`, `streamlit`, `joblib`, `pytest`
- [ ] Write `README.md` with project description, dataset note, and
      `streamlit run app.py` quickstart
- [ ] Add empty `__init__.py` files to `src/` and `tests/` so they are
      importable packages
- [ ] Confirm `atp_tennis.csv` is accessible at the workspace root

**Status:** `[ ] pending`

---

## Phase 2 — Module 1: Data Preparation

**File:** `src/prepare.py`

**Intent:** Produce a clean, leakage-free CSV that every downstream module can
load without repeating cleaning logic.

**Expected Outcomes:**
- `data/clean.csv` written with exactly these columns:
  `Date`, `Series`, `Court`, `Surface`, `Round`, `Best of`,
  `Player_1`, `Player_2`, `Rank_1`, `Rank_2`, `player_1_wins`
- No rows with missing `Rank_1` or `Rank_2`.
- No sentinel `-1` values remain in any kept column.
- A printed summary confirms row count and date range.

**Todo List:**
- [ ] Load `atp_tennis.csv` with `pandas.read_csv`, parse `Date` as datetime
- [ ] Drop columns: `Tournament`, `Pts_1`, `Pts_2`, `Odd_1`, `Odd_2`, `Score`
- [ ] Derive target: `player_1_wins = (Winner == Player_1).astype(int)`
- [ ] Drop `Winner` column after target is derived
- [ ] Drop rows where `Rank_1` or `Rank_2` is null or <= 0
- [ ] Reset index and write to `data/clean.csv`
- [ ] Print: total rows, date range, target class balance

**Relevant Context:**
- Source file: `atp_tennis.csv` (workspace root)
- Output: `data/clean.csv`

**Status:** `[ ] pending`

---

## Phase 3 — Module 2: Feature Engineering

**File:** `src/features.py`

**Intent:** Transform the clean data into a numeric feature matrix that the
Random Forest can consume, with no player-name columns and no leakage.

**Expected Outcomes:**
- `data/features.csv` written containing all engineered numeric/encoded columns
  plus `player_1_wins`, `Date`, `Player_1`, `Player_2`.
- `Date`, `Player_1`, `Player_2` are present for reference but never passed to
  the model.
- All categorical columns are one-hot encoded (no string columns remain except
  the three reference columns above).

**Todo List:**
- [ ] Load `data/clean.csv`
- [ ] Engineer numeric features:
  - `rank_diff = Rank_1 - Rank_2` (positive means Player 1 is lower-ranked / weaker)
  - `rank_ratio = Rank_1 / Rank_2`
  - `log_rank_diff = log1p(Rank_1) - log1p(Rank_2)`
- [ ] One-hot encode: `Surface`, `Round`, `Series`, `Court`, `Best of`
      using `pandas.get_dummies` with `drop_first=True`
- [ ] Drop original categorical source columns after encoding
- [ ] Keep `Date`, `Player_1`, `Player_2` as non-feature reference columns
- [ ] Write to `data/features.csv`
- [ ] Print: final column list, shape

**Relevant Context:**
- Input: `data/clean.csv`
- Output: `data/features.csv`

**Status:** `[ ] pending`

---

## Phase 4 — Module 3: Model Building

**File:** `src/train.py`

**Intent:** Train a Random Forest classifier using a chronological split,
evaluate it on the validation set, and serialise the model for reuse.

**Expected Outcomes:**
- `models/rf_model.joblib` saved to disk.
- Console output showing: accuracy, precision, recall, F1, ROC-AUC on the 2025
  validation split.
- A `feature_columns.txt` file listing the exact feature names the model was
  trained on (needed by `app.py` and `predict.py`).

**Todo List:**
- [ ] Load `data/features.csv`, parse `Date` as datetime
- [ ] Define `FEATURE_COLS` as all columns except `Date`, `Player_1`,
      `Player_2`, `player_1_wins`
- [ ] Split chronologically:
  - Train: `Date.dt.year <= 2024`
  - Validation: `Date.dt.year == 2025`
  - Holdout: `Date.dt.year >= 2026` (set aside, not used here)
- [ ] Fit `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)`
      on train set
- [ ] Evaluate on validation set: `accuracy_score`, `classification_report`,
      `roc_auc_score`
- [ ] Save model with `joblib.dump` to `models/rf_model.joblib`
- [ ] Write `FEATURE_COLS` list to `models/feature_columns.txt` (one per line)
- [ ] Print all metrics to console

**Relevant Context:**
- Input: `data/features.csv`
- Outputs: `models/rf_model.joblib`, `models/feature_columns.txt`

**Status:** `[ ] pending`

---

## Phase 5 — Module 4: Model Interpretation

**File:** `src/interpret.py`

**Intent:** Explain which features drive the model's predictions using two
complementary methods: the Random Forest's built-in impurity importance and
sklearn's permutation importance on the validation set.

**Expected Outcomes:**
- `outputs/feature_importance.png` — bar chart of top-20 built-in importances.
- `outputs/permutation_importance.png` — bar chart of top-20 permutation
  importances (mean +/- std).
- Console printout of both ranked lists.

**Todo List:**
- [ ] Load `models/rf_model.joblib` and `models/feature_columns.txt`
- [ ] Load `data/features.csv`; reconstruct the 2025 validation split
- [ ] Plot built-in `feature_importances_` as a horizontal bar chart; save to
      `outputs/feature_importance.png`
- [ ] Run `sklearn.inspection.permutation_importance` on the validation set
      (`n_repeats=10, random_state=42`)
- [ ] Plot permutation importance means with error bars (std); save to
      `outputs/permutation_importance.png`
- [ ] Print both ranked tables to console

**Relevant Context:**
- Inputs: `models/rf_model.joblib`, `models/feature_columns.txt`,
  `data/features.csv`
- Outputs: `outputs/feature_importance.png`,
  `outputs/permutation_importance.png`

**Status:** `[ ] pending`

---

## Phase 6 — Module 5: Live Prediction Challenge

**File:** `src/predict.py`

**Intent:** Simulate a "live" prediction scenario by running the trained model
against the held-out 2026 matches and comparing predictions to actual outcomes.

**Expected Outcomes:**
- `outputs/holdout_predictions.csv` containing: `Date`, `Player_1`, `Player_2`,
  `prob_player_1_wins`, `predicted_winner`, `actual_winner`, `correct`.
- Console output of holdout accuracy, ROC-AUC, and a sample of 10 rows.

**Todo List:**
- [ ] Load `models/rf_model.joblib` and `models/feature_columns.txt`
- [ ] Load `data/features.csv`; filter to `Date.dt.year >= 2026`
- [ ] Run `model.predict_proba` on holdout feature matrix
- [ ] Derive `predicted_winner`:
      if `prob_player_1_wins >= 0.5` then `Player_1` else `Player_2`
- [ ] Derive `actual_winner` from `player_1_wins` column
- [ ] Compute `correct = (predicted_winner == actual_winner)`
- [ ] Save results to `outputs/holdout_predictions.csv`
- [ ] Print: holdout accuracy, ROC-AUC, sample rows

**Relevant Context:**
- Inputs: `models/rf_model.joblib`, `models/feature_columns.txt`,
  `data/features.csv`
- Output: `outputs/holdout_predictions.csv`

**Status:** `[ ] pending`

---

## Phase 7 — Module 6: Streamlit Web UI

**File:** `app.py`

**Intent:** Provide a beginner-friendly browser interface where a user selects
two players and an optional match context, then sees the predicted win
probability for each player.

**Expected Outcomes:**
- `streamlit run app.py` launches a browser UI with no errors.
- Two player dropdowns populated from all unique player names in the dataset.
- A "Predict" button triggers `model.predict_proba` and displays:
  - Win probability for Player 1 (progress bar / metric widget).
  - Win probability for Player 2.
  - Which player is predicted to win.
- Optional context inputs: Surface, Round, Series, Court, Best of (with
  sensible defaults).
- Player names are never passed to the model; only their latest ranking is
  looked up from the dataset.

**Todo List:**
- [ ] Load `data/clean.csv` once (cached with `@st.cache_data`) to build
      player name list and latest-ranking lookup
- [ ] Load `models/rf_model.joblib` and `models/feature_columns.txt`
      (cached with `@st.cache_resource`)
- [ ] Sidebar or main panel: two `st.selectbox` widgets for Player 1 / Player 2
- [ ] Context inputs: `st.selectbox` for Surface, Round, Series, Court;
      `st.radio` for Best of (3 or 5)
- [ ] On button click: look up `Rank_1` and `Rank_2` as the player's most
      recent ranking from clean data; build feature row matching
      `feature_columns.txt`; call `model.predict_proba`
- [ ] Display results using `st.metric`, `st.progress`, and a winner callout
- [ ] Add a brief "How it works" expander explaining the model and features

**Relevant Context:**
- Inputs: `data/clean.csv`, `models/rf_model.joblib`,
  `models/feature_columns.txt`
- Run command: `streamlit run app.py`

**Status:** `[ ] pending`

---

## Phase 8 — Testing & Validation

**File:** `tests/test_pipeline.py`

**Intent:** Confirm that the full pipeline runs end-to-end and that each
module's output meets minimum quality expectations.

**Expected Outcomes:**
- All tests pass with `pytest tests/` from the workspace root.
- No new warnings or failures introduced.

**Todo List:**
- [ ] Test `prepare.py`: assert `data/clean.csv` exists, has no null `Rank_1`/
      `Rank_2`, target is binary (0/1 only), columns list is correct
- [ ] Test `features.py`: assert `data/features.csv` contains expected
      engineered columns (`rank_diff`, `rank_ratio`, `log_rank_diff`), no
      string columns except reference columns
- [ ] Test `train.py`: assert `models/rf_model.joblib` exists and is loadable;
      assert `models/feature_columns.txt` is non-empty
- [ ] Test `predict.py`: assert `outputs/holdout_predictions.csv` exists;
      assert holdout accuracy > 0.55 (better than random)
- [ ] Test `app.py` imports cleanly (no `streamlit run` required — just
      `import app` with `TESTING=true` guard if needed)

**Relevant Context:**
- All outputs from Phases 2–7 must exist before running the full test suite.
- Run with: `pytest tests/ -v`

**Status:** `[ ] pending`

---

## Implementation Order

```
Phase 1  Setup
    └── Phase 2  prepare.py
            └── Phase 3  features.py
                    └── Phase 4  train.py
                            ├── Phase 5  interpret.py
                            ├── Phase 6  predict.py
                            └── Phase 7  app.py
                                    └── Phase 8  tests
```

Each phase depends on the outputs of the phase immediately above it.
Phases 5, 6, and 7 can all begin once Phase 4 is complete.
