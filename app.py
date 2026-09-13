"""
Phase 7 — Streamlit Web UI
ATP Tennis Match Win Probability Predictor

Run with:
    streamlit run app.py
"""

import pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = pathlib.Path(__file__).parent
CLEAN_CSV  = ROOT / "data" / "clean.csv"
MODEL_PATH = ROOT / "models" / "rf_model.joblib"
COLS_PATH  = ROOT / "models" / "feature_columns.txt"

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV, parse_dates=["Date"])


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_cols() -> list[str]:
    return COLS_PATH.read_text().splitlines()


@st.cache_data
def build_player_ranking_lookup(df: pd.DataFrame) -> dict[str, int]:
    """Return {player_name: latest_rank} using the most recent appearance."""
    df_sorted = df.sort_values("Date")
    # Take the most recent rank from Player_1 appearances
    p1 = df_sorted.groupby("Player_1")["Rank_1"].last().rename("Rank")
    # Take the most recent rank from Player_2 appearances
    p2 = df_sorted.groupby("Player_2")["Rank_2"].last().rename("Rank")
    # Merge: prefer the most-recent record across both columns
    combined = pd.concat([p1, p2])
    # For players in both, keep the most recent (p2 concat after p1; last wins)
    latest = combined.groupby(level=0).last()
    return latest.to_dict()


@st.cache_data
def get_sorted_players(df: pd.DataFrame) -> list[str]:
    players = set(df["Player_1"].unique()) | set(df["Player_2"].unique())
    return sorted(players)


# ---------------------------------------------------------------------------
# Feature row builder
# ---------------------------------------------------------------------------

def build_feature_row(
    rank_1: int,
    rank_2: int,
    surface: str,
    round_: str,
    series: str,
    court: str,
    best_of: int,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame matching the exact feature matrix the
    model was trained on.  Uses the same one-hot encoding scheme as features.py
    (get_dummies with drop_first=True on a single row against known categories).
    """
    # ------------------------------------------------------------------
    # 1. Numeric rank features (mirroring features.py)
    # ------------------------------------------------------------------
    rank_diff     = rank_1 - rank_2
    rank_ratio    = rank_1 / rank_2
    log_rank_diff = np.log1p(rank_1) - np.log1p(rank_2)

    # ------------------------------------------------------------------
    # 2. Build a dict of all possible one-hot columns (False by default)
    # ------------------------------------------------------------------
    # We reconstruct every dummy column the model knows about by scanning
    # feature_cols for the pattern <category>_<value>.
    row: dict = {
        "Rank_1":        rank_1,
        "Rank_2":        rank_2,
        "rank_diff":     rank_diff,
        "rank_ratio":    rank_ratio,
        "log_rank_diff": log_rank_diff,
    }

    # Initialise all dummy columns to False
    dummy_cols = [c for c in feature_cols if c not in row]
    for col in dummy_cols:
        row[col] = False

    # ------------------------------------------------------------------
    # 3. Set the active dummies to True
    # ------------------------------------------------------------------
    # Mapping: context value  →  column name in features.csv
    context_map = {
        f"Surface_{surface}": True,
        f"Round_{round_}":    True,
        f"Series_{series}":   True,
        f"Court_{court}":     True,
        f"Best of_{best_of}": True,
    }
    for col, val in context_map.items():
        if col in row:   # only set if the model actually has this column
            row[col] = val

    # ------------------------------------------------------------------
    # 4. Return as single-row DataFrame in correct column order
    # ------------------------------------------------------------------
    return pd.DataFrame([row])[feature_cols]


# ---------------------------------------------------------------------------
# Known category values (from prepare.py analysis)
# ---------------------------------------------------------------------------
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
ROUNDS   = ["1st Round", "2nd Round", "3rd Round", "4th Round",
            "Quarterfinals", "Round Robin", "Semifinals", "The Final"]
SERIES   = ["Masters 1000", "Grand Slam", "ATP500", "Masters",
            "International", "International Gold", "Masters Cup", "ATP250"]
COURTS   = ["Outdoor", "Indoor"]

# ---------------------------------------------------------------------------
# Guard: skip all Streamlit UI calls when running under tests
# ---------------------------------------------------------------------------
import os as _os
_TESTING = _os.environ.get("TESTING", "").lower() in ("1", "true", "yes")

if not _TESTING:
    # -----------------------------------------------------------------------
    # Page config
    # -----------------------------------------------------------------------
    st.set_page_config(
        page_title="ATP Tennis Predictor",
        page_icon="🎾",
        layout="centered",
    )

    # -----------------------------------------------------------------------
    # Load resources
    # -----------------------------------------------------------------------
    df_clean      = load_clean_data()
    model         = load_model()
    feature_cols  = load_feature_cols()
    player_lookup = build_player_ranking_lookup(df_clean)
    all_players   = get_sorted_players(df_clean)

# ---------------------------------------------------------------------------
# UI  (only executed when Streamlit actually runs the app)
# ---------------------------------------------------------------------------
if not _TESTING:
    st.title("🎾 ATP Tennis Match Predictor")
    st.markdown(
        "Select two players and match context to get predicted win probabilities "
        "from a Random Forest model trained on ATP data (2000–2024)."
    )

    st.divider()

    # ── Player selection ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Player 1")
        player_1 = st.selectbox("Select Player 1", all_players, index=all_players.index("Federer R.") if "Federer R." in all_players else 0, key="p1")
        rank_1_default = int(player_lookup.get(player_1, 100))
        rank_1 = st.number_input("ATP Ranking (Player 1)", min_value=1, max_value=2000, value=rank_1_default, key="r1")

    with col2:
        st.subheader("Player 2")
        p2_default_idx = all_players.index("Nadal R.") if "Nadal R." in all_players else min(1, len(all_players) - 1)
        player_2 = st.selectbox("Select Player 2", all_players, index=p2_default_idx, key="p2")
        rank_2_default = int(player_lookup.get(player_2, 100))
        rank_2 = st.number_input("ATP Ranking (Player 2)", min_value=1, max_value=2000, value=rank_2_default, key="r2")

    st.divider()

    # ── Match context ──────────────────────────────────────────────────────────
    st.subheader("Match Context")

    ctx1, ctx2, ctx3 = st.columns(3)
    with ctx1:
        surface = st.selectbox("Surface", SURFACES, index=0)
        court   = st.selectbox("Court", COURTS, index=0)
    with ctx2:
        round_  = st.selectbox("Round", ROUNDS, index=0)
    with ctx3:
        series  = st.selectbox("Series / Tournament tier", SERIES, index=0)
        best_of = st.radio("Best of", [3, 5], index=0, horizontal=True)

    st.divider()

    # ── Predict ────────────────────────────────────────────────────────────────
    predict_clicked = st.button("🎾 Predict", type="primary", use_container_width=True)

    if predict_clicked:
        if player_1 == player_2:
            st.warning("Please select two different players.")
        else:
            X = build_feature_row(
                rank_1=rank_1,
                rank_2=rank_2,
                surface=surface,
                round_=round_,
                series=series,
                court=court,
                best_of=best_of,
                feature_cols=feature_cols,
            )

            proba = model.predict_proba(X)[0]
            p1_win = float(proba[1])
            p2_win = float(proba[0])
            winner = player_1 if p1_win >= 0.5 else player_2

            st.subheader("Prediction")

            res1, res2 = st.columns(2)
            with res1:
                st.metric(label=f"{player_1}", value=f"{p1_win:.1%}")
                st.progress(p1_win)
            with res2:
                st.metric(label=f"{player_2}", value=f"{p2_win:.1%}")
                st.progress(p2_win)

            st.success(f"🏆 Predicted winner: **{winner}**")

            with st.expander("Match details used"):
                st.write({
                    "Player 1": player_1, "Rank 1": rank_1,
                    "Player 2": player_2, "Rank 2": rank_2,
                    "Surface": surface, "Round": round_,
                    "Series": series, "Court": court, "Best of": best_of,
                })

    # ── How it works ────────────────────────────────────────────────────────────
    with st.expander("ℹ️ How it works"):
        st.markdown("""
**Model:** Random Forest (200 trees, trained on 2000–2024 ATP match data)

**Features used:**
- ATP rankings of both players (`Rank_1`, `Rank_2`)
- Engineered ranking features: `rank_diff`, `rank_ratio`, `log_rank_diff`
- Match context: Surface, Round, Series/Tournament tier, Court type, Best of

**What's intentionally excluded:**
- Player names (never enter the model — only used to look up rankings)
- Betting odds (post-market information — data leakage)
- Match scores (post-match information — data leakage)

**Validation (2025):** ~61% accuracy, ROC-AUC ~0.65
**Holdout (2026):** ~61% accuracy, ROC-AUC ~0.65

ATP ranking is by far the strongest predictor, accounting for ~85% of the
model's feature importance. Match context (surface, round, series) adds a
meaningful but smaller signal.
""")
