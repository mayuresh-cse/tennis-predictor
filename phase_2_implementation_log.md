# Phase 2 Implementation Log — Data Preparation

**Date:** 2025-07-29  
**Phase:** 2 of 8  
**Status:** Complete

---

## Objectives

Produce a clean, leakage-free CSV (`data/clean.csv`) that every downstream module
can load without repeating cleaning logic.

---

## Steps Executed

### 1. Pre-flight inspection

Before writing any code, inspected the raw CSV to confirm the exact state of the data:

```
Total rows   : 68,635
Rank_1 range : -1 to 3390  (14 rows with value -1)
Rank_2 range : -1 to 4915  (12 rows with value -1)
Rank nulls   : 0 in either column
Date range   : 2000-01-03 to 2026-08-29
Target balance: Winner == Player_1 exactly 50.0%
```

Key finding: `Rank_1`/`Rank_2` have no null values but do contain the sentinel value
`-1` in 26 rows (14 + 12, with some overlap). These are dropped by the `> 0` filter.

### 2. `src/prepare.py` written

The module implements a single `prepare()` function with these steps:

| Step | Action |
|---|---|
| 1 | `pd.read_csv` with `parse_dates=["Date"]` |
| 2 | Drop `Tournament`, `Pts_1`, `Pts_2`, `Odd_1`, `Odd_2`, `Score` |
| 3 | Derive `player_1_wins = (Winner == Player_1).astype(int)` |
| 4 | Drop `Winner` column |
| 5 | Filter to `Rank_1 > 0` and `Rank_2 > 0`; reset index |
| 6 | Write to `data/clean.csv` |
| 7 | Print summary (rows, date range, class balance) |

The file is runnable standalone (`python src/prepare.py`) and also importable by
other modules via `from src.prepare import prepare`.

### 3. Execution output

```
Rows written : 68,609  (dropped 26 invalid-rank rows)
Date range   : 2000-01-03 to 2026-08-29
Class balance: player_1_wins=1 -> 50.0%  |  player_1_wins=0 -> 50.0%
Output       : D:\tennis-predictor\data\clean.csv
```

---

## Output Verification

All acceptance criteria from the implementation plan were confirmed:

| Check | Result |
|---|---|
| `data/clean.csv` exists | Yes |
| Exactly 11 columns | Yes — `Date, Series, Court, Surface, Round, Best of, Player_1, Player_2, Rank_1, Rank_2, player_1_wins` |
| No null values in any column | 0 nulls across all 11 columns |
| No `Rank_1` or `Rank_2` <= 0 | 0 rows |
| `player_1_wins` is binary (0/1 only) | Unique values: `[0, 1]` |
| Target is balanced | 50.0% class-1, 50.0% class-0 |
| Row count | 68,609 (26 invalid-rank rows dropped from 68,635) |
| Date range | 2000-01-03 to 2026-08-29 |

---

## Deviations from Plan

None. All steps executed exactly as specified.

---

## Minor Fix Applied

The `print` statement originally used a Unicode arrow character (`->`) which caused
a `UnicodeEncodeError` on the Windows cp1252 console. Replaced with ASCII `->`.
This has no effect on the output CSV or any downstream module.

---

## Files Created / Modified

| File | Action | Purpose |
|---|---|---|
| `src/prepare.py` | Created | Data cleaning module |
| `data/clean.csv` | Created | Cleaned dataset (68,609 rows, 11 columns) |
| `phase_2_implementation_log.md` | Created | This log |

---

## Next Phase

**Phase 3 — Feature Engineering** (`src/features.py`)  
Input: `data/clean.csv`  
Output: `data/features.csv`
