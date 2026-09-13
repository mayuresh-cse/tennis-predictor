# ATP Tennis Match Predictor

A machine-learning web application that predicts the win probability for a match
between any two ATP players, built on historical match data (2000–2026).

---

## Project Overview

The model is trained on **68 636 ATP match records** from `atp_tennis.csv` using
only pre-match contextual features — rankings, surface, round, tournament series,
court type, and match format. Betting odds and post-match columns are excluded
to prevent data leakage.

A **Streamlit** web UI lets a user pick two players from a dropdown and instantly
see predicted win probabilities.

---

## Dataset

| Field | Description |
|---|---|
| Source | `atp_tennis.csv` (workspace root) |
| Rows | 68 636 |
| Date range | 2000 – 2026 |
| Target | `player_1_wins` — 1 if Player 1 won, 0 otherwise |

**Chronological splits:**

| Split | Years | Purpose |
|---|---|---|
| Train | 2000–2024 | Model fitting |
| Validation | 2025 | Hyperparameter / threshold tuning |
| Holdout | 2026 | Live prediction challenge |

---

## Tech Stack

| Layer | Library |
|---|---|
| Data handling | pandas 2.2 |
| Machine learning | scikit-learn 1.5 |
| Serialisation | joblib 1.4 |
| Visualisation | matplotlib 3.9, seaborn 0.13 |
| Web UI | Streamlit 1.37 |
| Testing | pytest 8.3 |

---

## Project Structure

```
tennis-predictor/
├── atp_tennis.csv            # Source dataset
├── data/
│   ├── clean.csv             # Phase 2 output
│   └── features.csv          # Phase 3 output
├── models/
│   ├── rf_model.joblib       # Trained Random Forest (Phase 4)
│   └── feature_columns.txt   # Feature column list (Phase 4)
├── outputs/
│   ├── feature_importance.png
│   ├── permutation_importance.png
│   └── holdout_predictions.csv
├── src/
│   ├── prepare.py            # Phase 2 — data cleaning
│   ├── features.py           # Phase 3 — feature engineering
│   ├── train.py              # Phase 4 — model training
│   ├── interpret.py          # Phase 5 — feature importance
│   └── predict.py            # Phase 6 — holdout prediction
├── tests/
│   └── test_pipeline.py      # Phase 8 — smoke tests
├── app.py                    # Phase 7 — Streamlit web UI
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python src/prepare.py
python src/features.py
python src/train.py
python src/interpret.py
python src/predict.py

# 4. Launch the web UI
streamlit run app.py

# 5. Run tests
pytest tests/ -v
```

---

## Key Design Decisions

- **No data leakage** — betting odds (`Odd_1`, `Odd_2`) and `Score` are dropped before any feature is built.
- **Chronological splits** — all train/val/test boundaries are time-based, never random.
- **Player names never enter the model** — only their latest ATP ranking is used as a numeric feature.
- **Random state fixed at 42** throughout for full reproducibility.
