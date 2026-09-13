"""
Phase 8 — Smoke tests for the full ATP Tennis Predictor pipeline.

Covers:
    prepare.py  →  data/clean.csv
    features.py →  data/features.csv
    train.py    →  models/rf_model.joblib + models/feature_columns.txt
    predict.py  →  outputs/holdout_predictions.csv
    app.py      →  importable under TESTING=true (no Streamlit server needed)

Run with:
    pytest tests/ -v
"""

import os
import sys
import pathlib
import types

import joblib
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of how pytest is invoked
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def clean_csv() -> pd.DataFrame:
    path = ROOT / "data" / "clean.csv"
    assert path.exists(), f"data/clean.csv not found — run src/prepare.py first"
    return pd.read_csv(path, parse_dates=["Date"])


@pytest.fixture(scope="session")
def features_csv() -> pd.DataFrame:
    path = ROOT / "data" / "features.csv"
    assert path.exists(), f"data/features.csv not found — run src/features.py first"
    return pd.read_csv(path, parse_dates=["Date"])


@pytest.fixture(scope="session")
def feature_cols() -> list:
    path = ROOT / "models" / "feature_columns.txt"
    assert path.exists(), "models/feature_columns.txt not found — run src/train.py first"
    cols = path.read_text().splitlines()
    assert len(cols) > 0, "feature_columns.txt is empty"
    return cols


@pytest.fixture(scope="session")
def rf_model():
    path = ROOT / "models" / "rf_model.joblib"
    assert path.exists(), "models/rf_model.joblib not found — run src/train.py first"
    return joblib.load(path)


@pytest.fixture(scope="session")
def holdout_preds() -> pd.DataFrame:
    path = ROOT / "outputs" / "holdout_predictions.csv"
    assert path.exists(), "outputs/holdout_predictions.csv not found — run src/predict.py first"
    return pd.read_csv(path)


# ===========================================================================
# Phase 2 — data/clean.csv
# ===========================================================================

class TestPrepare:
    EXPECTED_COLS = {
        "Date", "Series", "Court", "Surface", "Round", "Best of",
        "Player_1", "Player_2", "Rank_1", "Rank_2", "player_1_wins",
    }

    def test_file_exists(self):
        assert (ROOT / "data" / "clean.csv").exists()

    def test_expected_columns(self, clean_csv):
        assert self.EXPECTED_COLS == set(clean_csv.columns), (
            f"Column mismatch.\n"
            f"  Missing : {self.EXPECTED_COLS - set(clean_csv.columns)}\n"
            f"  Extra   : {set(clean_csv.columns) - self.EXPECTED_COLS}"
        )

    def test_no_null_rank_1(self, clean_csv):
        assert clean_csv["Rank_1"].isnull().sum() == 0, "Rank_1 has null values"

    def test_no_null_rank_2(self, clean_csv):
        assert clean_csv["Rank_2"].isnull().sum() == 0, "Rank_2 has null values"

    def test_no_nonpositive_rank_1(self, clean_csv):
        assert (clean_csv["Rank_1"] > 0).all(), "Rank_1 contains values <= 0"

    def test_no_nonpositive_rank_2(self, clean_csv):
        assert (clean_csv["Rank_2"] > 0).all(), "Rank_2 contains values <= 0"

    def test_target_is_binary(self, clean_csv):
        unique_vals = set(clean_csv["player_1_wins"].unique())
        assert unique_vals == {0, 1}, f"player_1_wins has unexpected values: {unique_vals}"

    def test_row_count(self, clean_csv):
        assert len(clean_csv) > 50_000, "Expected >50k rows in clean.csv"

    def test_leakage_columns_absent(self, clean_csv):
        leakage = {"Odd_1", "Odd_2", "Score", "Tournament", "Pts_1", "Pts_2", "Winner"}
        present = leakage & set(clean_csv.columns)
        assert not present, f"Leakage columns still present: {present}"


# ===========================================================================
# Phase 3 — data/features.csv
# ===========================================================================

class TestFeatures:
    ENGINEERED = {"rank_diff", "rank_ratio", "log_rank_diff"}
    REFERENCE  = {"Date", "Player_1", "Player_2"}

    def test_file_exists(self):
        assert (ROOT / "data" / "features.csv").exists()

    def test_engineered_columns_present(self, features_csv):
        missing = self.ENGINEERED - set(features_csv.columns)
        assert not missing, f"Engineered columns missing: {missing}"

    def test_no_string_cols_except_reference(self, features_csv):
        string_cols = {
            c for c in features_csv.columns
            if features_csv[c].dtype == object and c not in self.REFERENCE
        }
        assert not string_cols, f"Unexpected string columns: {string_cols}"

    def test_reference_cols_present(self, features_csv):
        missing = self.REFERENCE - set(features_csv.columns)
        assert not missing, f"Reference columns missing: {missing}"

    def test_target_present(self, features_csv):
        assert "player_1_wins" in features_csv.columns

    def test_rank_diff_values(self, features_csv):
        expected = features_csv["Rank_1"] - features_csv["Rank_2"]
        diff = (features_csv["rank_diff"] - expected).abs().max()
        assert diff < 1e-9, "rank_diff values are incorrect"


# ===========================================================================
# Phase 4 — models/rf_model.joblib + feature_columns.txt
# ===========================================================================

class TestTrain:
    def test_model_file_exists(self):
        assert (ROOT / "models" / "rf_model.joblib").exists()

    def test_feature_cols_file_exists(self):
        assert (ROOT / "models" / "feature_columns.txt").exists()

    def test_model_loadable(self, rf_model):
        assert rf_model is not None

    def test_model_has_predict_proba(self, rf_model):
        assert hasattr(rf_model, "predict_proba")

    def test_feature_cols_non_empty(self, feature_cols):
        assert len(feature_cols) > 0

    def test_feature_cols_no_reference_cols(self, feature_cols):
        forbidden = {"Date", "Player_1", "Player_2", "player_1_wins"}
        present = forbidden & set(feature_cols)
        assert not present, f"Non-feature columns found in feature_columns.txt: {present}"

    def test_model_predicts_on_val_set(self, rf_model, features_csv, feature_cols):
        val = features_csv[features_csv["Date"].dt.year == 2025]
        assert len(val) > 0, "No 2025 rows found for validation"
        proba = rf_model.predict_proba(val[feature_cols])
        assert proba.shape == (len(val), 2)
        # Probabilities must sum to 1 per row
        assert (abs(proba.sum(axis=1) - 1.0) < 1e-6).all()


# ===========================================================================
# Phase 6 — outputs/holdout_predictions.csv
# ===========================================================================

class TestPredict:
    EXPECTED_COLS = {
        "Date", "Player_1", "Player_2", "prob_player_1_wins",
        "predicted_winner", "actual_winner", "correct",
    }

    def test_file_exists(self):
        assert (ROOT / "outputs" / "holdout_predictions.csv").exists()

    def test_expected_columns(self, holdout_preds):
        assert self.EXPECTED_COLS == set(holdout_preds.columns), (
            f"Column mismatch: {set(holdout_preds.columns)}"
        )

    def test_no_nulls(self, holdout_preds):
        null_counts = holdout_preds.isnull().sum()
        assert null_counts.sum() == 0, f"Nulls found:\n{null_counts[null_counts > 0]}"

    def test_prob_in_unit_interval(self, holdout_preds):
        p = holdout_preds["prob_player_1_wins"]
        assert p.between(0.0, 1.0).all(), "prob_player_1_wins out of [0,1]"

    def test_holdout_accuracy_above_baseline(self, holdout_preds):
        acc = holdout_preds["correct"].mean()
        assert acc > 0.55, (
            f"Holdout accuracy {acc:.4f} is not better than the 0.55 threshold"
        )

    def test_row_count(self, holdout_preds):
        assert len(holdout_preds) > 100, "Expected >100 holdout rows"


# ===========================================================================
# Phase 7 — app.py importable under TESTING=true
# ===========================================================================

class TestApp:
    def test_app_imports_cleanly(self):
        """app.py must import without a Streamlit server when TESTING=true."""
        # Ensure any cached import is cleared so the env var takes effect
        if "app" in sys.modules:
            del sys.modules["app"]

        os.environ["TESTING"] = "true"

        # Stub streamlit if not already stubbed
        if "streamlit" not in sys.modules or not isinstance(
            sys.modules["streamlit"], types.ModuleType
        ):
            st_stub = types.ModuleType("streamlit")
            st_stub.cache_data     = lambda f=None, **kw: (f if f else lambda fn: fn)
            st_stub.cache_resource = lambda f=None, **kw: (f if f else lambda fn: fn)
            sys.modules["streamlit"] = st_stub

        import app  # noqa: F401
        assert hasattr(app, "build_feature_row"), "app.build_feature_row not found"
        assert hasattr(app, "load_feature_cols"), "app.load_feature_cols not found"

    def test_build_feature_row(self):
        """build_feature_row must produce correct shape and values."""
        import numpy as np

        if "app" not in sys.modules:
            os.environ["TESTING"] = "true"
            import app

        import app as _app
        cols = _app.load_feature_cols()
        row  = _app.build_feature_row(10, 50, "Clay", "Quarterfinals",
                                      "Grand Slam", "Outdoor", 5, cols)

        assert isinstance(row, pd.DataFrame)
        assert list(row.columns) == cols
        assert row.shape == (1, len(cols))
        assert row["rank_diff"].iloc[0] == 10 - 50
        assert abs(row["rank_ratio"].iloc[0] - 10 / 50) < 1e-9
        assert abs(
            row["log_rank_diff"].iloc[0]
            - (np.log1p(10) - np.log1p(50))
        ) < 1e-9
