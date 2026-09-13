"""
Phase 2 — Data Preparation
Cleans atp_tennis.csv and writes data/clean.csv.

Columns kept:
    Date, Series, Court, Surface, Round, Best of,
    Player_1, Player_2, Rank_1, Rank_2, player_1_wins

Columns dropped (leakage / sentinel / redundant):
    Tournament, Pts_1, Pts_2, Odd_1, Odd_2, Score, Winner
"""

import pathlib
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
SOURCE = ROOT / "atp_tennis.csv"
OUTPUT = ROOT / "data" / "clean.csv"

COLS_TO_DROP = ["Tournament", "Pts_1", "Pts_2", "Odd_1", "Odd_2", "Score"]


def prepare(source: pathlib.Path = SOURCE, output: pathlib.Path = OUTPUT) -> pd.DataFrame:
    """Load, clean, and save the dataset.  Returns the cleaned DataFrame."""

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    df = pd.read_csv(source, parse_dates=["Date"])

    # ------------------------------------------------------------------
    # 2. Drop leakage / redundant columns
    # ------------------------------------------------------------------
    df = df.drop(columns=COLS_TO_DROP)

    # ------------------------------------------------------------------
    # 3. Derive target variable
    # ------------------------------------------------------------------
    df["player_1_wins"] = (df["Winner"] == df["Player_1"]).astype(int)
    df = df.drop(columns=["Winner"])

    # ------------------------------------------------------------------
    # 4. Remove rows with invalid rankings (null or sentinel <= 0)
    # ------------------------------------------------------------------
    valid_rank = (df["Rank_1"] > 0) & (df["Rank_2"] > 0)
    n_dropped = (~valid_rank).sum()
    df = df[valid_rank].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    balance = df["player_1_wins"].mean()
    print(f"Rows written : {len(df):,}  (dropped {n_dropped} invalid-rank rows)")
    print(f"Date range   : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Class balance: player_1_wins=1 -> {balance:.1%}  |  player_1_wins=0 -> {1 - balance:.1%}")
    print(f"Output       : {output}")

    return df


if __name__ == "__main__":
    prepare()
