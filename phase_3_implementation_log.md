# Phase 3 Implementation Log — Feature Engineering

**Date:** 2025-07-29  
**Phase:** 3 of 8  
**Status:** Complete

---

## Objectives

Transform `data/clean.csv` into a numeric feature matrix (`data/features.csv`)
that the Random Forest can consume directly, with:
- Three engineered rank-comparison features
- All categorical columns one-hot encoded
- Reference columns (`Date`, `Player_1`, `Player_2`) preserved but not in the feature set

---

## Pre-flight Inspection

Inspected `data/clean.csv` to confirm cardinalities before encoding:

| Column | Unique values | Dummies produced (drop_first) |
|---|---|---|
| `Surface` | 4 (Carpet, Clay, Grass, Hard) | 3 |
| `Round` | 8 (1st–4th Round, QF, RR, SF, Final) | 7 |
| `Series` | 8 (ATP250, ATP500, Grand Slam, International, International Gold, Masters, Masters 1000, Masters Cup) | 7 |
| `Court` | 2 (Indoor, Outdoor) | 1 |
| `Best of` | 2 (3, 5) | 1 |

Total dummy columns: **19**. Plus `Rank_1`, `Rank_2`, `rank_diff`, `rank_ratio`, `log_rank_diff` = **24 feature columns**.

---

## Steps Executed

### 1. `src/features.py` written

Single `build_features()` function — both callable by downstream modules and runnable as a script.

| Step | Action |
|---|---|
| 1 | Load `data/clean.csv` with `parse_dates=["Date"]` |
| 2 | Engineer `rank_diff`, `rank_ratio`, `log_rank_diff` from `Rank_1` / `Rank_2` |
| 3 | `pd.get_dummies` on `Surface`, `Round`, `Series`, `Court`, `Best of` with `drop_first=True` |
| 4 | Order columns: reference cols → target → feature cols |
| 5 | Write to `data/features.csv` (index-free) |
| 6 | Print shape, feature count, leftover string check, full column list |

### 2. Execution output

```
Shape          : (68609, 28)
Feature columns: 24
String cols left (excl. reference): none
Columns        : ['Date', 'Player_1', 'Player_2', 'player_1_wins',
                  'Rank_1', 'Rank_2', 'rank_diff', 'rank_ratio', 'log_rank_diff',
                  'Surface_Clay', 'Surface_Grass', 'Surface_Hard',
                  'Round_2nd Round', ..., 'Round_The Final',
                  'Series_ATP500', ..., 'Series_Masters Cup',
                  'Court_Outdoor', 'Best of_5']
Output         : D:\tennis-predictor\data\features.csv
```

---

## Output Verification

| Check | Result |
|---|---|
| `data/features.csv` exists | Yes |
| Shape | (68,609, 28) |
| Total nulls | 0 |
| `Date`, `Player_1`, `Player_2` present | Yes |
| `rank_diff`, `rank_ratio`, `log_rank_diff` present | Yes |
| Engineered rank features compute correctly (row 0) | `rank_diff=47`, `rank_ratio=1.8103`, `log_rank_diff=0.5859` |
| No string columns except reference cols | Confirmed — `string_cols = []` |
| All dummy columns present | Yes (`Surface_Clay`, `Round_Semifinals`, `Series_Grand Slam`, `Court_Outdoor`, `Best of_5` spot-checked) |
| Original categorical cols dropped | Yes (all 5 gone) |
| Feature column count | 24 |

---

## Column Map

| Column | Type | Role |
|---|---|---|
| `Date` | datetime | Reference only |
| `Player_1` | string | Reference only |
| `Player_2` | string | Reference only |
| `player_1_wins` | int (0/1) | Target |
| `Rank_1`, `Rank_2` | int | Raw rank features |
| `rank_diff` | float | Rank_1 - Rank_2 |
| `rank_ratio` | float | Rank_1 / Rank_2 |
| `log_rank_diff` | float | log1p(Rank_1) - log1p(Rank_2) |
| `Surface_Clay/Grass/Hard` | bool | One-hot (ref = Carpet) |
| `Round_2nd/3rd/4th Round/QF/RR/SF/Final` | bool | One-hot (ref = 1st Round) |
| `Series_ATP500/.../Masters Cup` | bool | One-hot (ref = ATP250) |
| `Court_Outdoor` | bool | One-hot (ref = Indoor) |
| `Best of_5` | bool | One-hot (ref = Best of 3) |

---

## Deviations from Plan

None. All steps executed exactly as specified.

---

## Files Created / Modified

| File | Action | Purpose |
|---|---|---|
| `src/features.py` | Created | Feature engineering module |
| `data/features.csv` | Created | Feature matrix (68,609 rows, 28 columns) |
| `phase_3_implementation_log.md` | Created | This log |

---

## Next Phase

**Phase 4 — Model Building** (`src/train.py`)  
Input: `data/features.csv`  
Outputs: `models/rf_model.joblib`, `models/feature_columns.txt`
