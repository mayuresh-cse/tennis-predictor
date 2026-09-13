# Phase 1 Implementation Log — Project Setup

**Date:** 2025-07-29  
**Phase:** 1 of 8  
**Status:** ✅ Complete

---

## Objectives

Create the project scaffolding that all subsequent phases depend on:
- Directory structure
- Dependency file (`requirements.txt`)
- Project documentation (`README.md`)
- Package init files (`src/__init__.py`, `tests/__init__.py`)
- Confirm dataset accessibility

---

## Steps Executed

### 1. Workspace Pre-check

Verified existing workspace contents before creating anything:

```
.venv\          ← existing virtual environment (Python 3.13.7)
atp_tennis.csv  ← source dataset confirmed present (17 columns)
IMPLEMENTATION_PLAN.md
```

### 2. Directory Structure Created

```
data/       ← output of prepare.py and features.py
models/     ← output of train.py
outputs/    ← output of interpret.py and predict.py
src/        ← Python module files
tests/      ← pytest test suite
```

Command used: `New-Item -ItemType Directory -Force -Path data, models, outputs, src, tests`

### 3. requirements.txt

**Decision logged:** Initial plan called for pinned versions (`pandas==2.2.2`, etc.).
However, Python 3.13.7 is present in the `.venv` and `pandas==2.2.2` requires a
Meson/C compiler build on this platform — no pre-built wheel available.

**Resolution:** Switched to minimum-version constraints (`>=`) so pip resolves the
latest available wheels for Python 3.13. Final installed versions:

| Package | Installed Version |
|---|---|
| pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| streamlit | 1.63.0 |
| joblib | 1.6.0 |
| pytest | 9.1.1 (already present) |
| numpy | 2.5.3 |

All packages installed successfully with `pip install -r requirements.txt`.

### 4. README.md

Written with:
- Project overview and dataset summary
- Tech stack table
- Full project structure tree
- Quickstart instructions (venv, pip install, pipeline run, streamlit, pytest)
- Key design decisions (no leakage, chronological splits, player names excluded)

### 5. Package Init Files

- `src/__init__.py` — marks `src/` as a Python package
- `tests/__init__.py` — marks `tests/` as a Python package

### 6. Dataset Verification

```python
import pandas as pd
df = pd.read_csv('atp_tennis.csv', nrows=3)
# Columns: ['Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round',
#            'Best of', 'Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2',
#            'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'Score']
```

All 17 expected columns present. Dataset ready for Phase 2.

---

## Deviations from Plan

| Item | Plan | Actual | Reason |
|---|---|---|---|
| `requirements.txt` version pinning | Exact pins (`==`) | Minimum bounds (`>=`) | Python 3.13.7 has no pre-built wheels for `pandas==2.2.2`; Meson build would fail. Latest compatible wheels resolve cleanly. |

---

## Verification Checklist

- [x] `data/` directory exists
- [x] `models/` directory exists
- [x] `outputs/` directory exists
- [x] `src/` directory exists
- [x] `tests/` directory exists
- [x] `requirements.txt` present and installs successfully
- [x] `README.md` present with project description and quickstart
- [x] `src/__init__.py` present
- [x] `tests/__init__.py` present
- [x] `atp_tennis.csv` accessible at workspace root (17 columns confirmed)
- [x] All 6 libraries import without error in `.venv`

---

## Files Created

| File | Purpose |
|---|---|
| `data/` | Output directory for cleaned and feature-engineered CSVs |
| `models/` | Output directory for serialised model and feature list |
| `outputs/` | Output directory for plots and prediction results |
| `src/__init__.py` | Makes src/ a Python package |
| `tests/__init__.py` | Makes tests/ a Python package |
| `requirements.txt` | Pinned dependency list |
| `README.md` | Project documentation and quickstart |

---

## Next Phase

**Phase 2 — Data Preparation** (`src/prepare.py`)  
Input: `atp_tennis.csv`  
Output: `data/clean.csv`
