# Education Imputation Pipeline — RDC Deployment Guide

XGBoost-based imputation of LEHD `educ_c` for the ~80-85% of workers with
missing observed education. Three nested models: demographics only,
+ neighborhood (RCF), + NAICS sector (EHF + ECF).

## Required directory structure inside the RDC

The package is the entire `screening/rdc/` folder. Place it under
`/projects/<project_id>/programs/` so that `screening/rdc/helpers.py`
is importable from there:

```
/projects/<project_id>/programs/
└── screening/
    └── rdc/
        ├── __init__.py
        ├── helpers.py          (logging, timing, SAS reader, thread limits)
        ├── env_check.py        (first-login diagnostics)
        ├── submit.py           (PBS job script generator)
        └── educ_impute/
            ├── __init__.py
            ├── config.py       (paths, columns, hyperparameters)
            ├── features.py     (data loading, feature engineering)
            ├── train.py        (CV, training, evaluation)
            ├── predict.py      (full-sample prediction)
            ├── run_all.py      (orchestrator entry point)
            ├── test_model1.py  (standalone debug script)
            └── README.md       (this file)
```

`screening/__init__.py` is NOT required — only the `rdc/` subtree.

## First-login workflow

```bash
# 1. Get a compute node (analytical jobs are not allowed on login nodes)
qsub -IX

# 2. Activate Python (Anaconda is not on PATH by default)
source /apps/anaconda/bin/activate py3cf

# 3. Set PYTHONPATH so the screening.rdc package is importable
cd /projects/<project_id>/programs
export PYTHONPATH=.

# 4. Run environment diagnostics (saves a report for your records)
python screening/rdc/env_check.py --save env_report.txt

# 5. Standalone smoke test on a small subset
python screening/rdc/educ_impute/test_model1.py \
    --icf-path /data/lehd/current/icf_us.sas7bdat \
    --n 100000

# 6. Full pipeline (interactive debugging)
python -m screening.rdc.educ_impute.run_all \
    --project-id <project_id> --debug --steps prep,train

# 7. Production batch job
python -m screening.rdc.submit ml_prod \
    --cmd "python -m screening.rdc.educ_impute.run_all --project-id <project_id>" \
    --submit
```

## Items to verify on first login

Open `educ_impute/config.py` and check every line marked `>>> VERIFY <<<`:

| Line | Constant | What to verify |
|------|----------|----------------|
| `DATA_ROOT` | `/data/lehd/current` | Actual LEHD data path on this RDC |
| `ICF_US_PATH` | `icf_us.sas7bdat` | File name and case |
| `RCF_PATH` | `icf_us_residence_rcf.sas7bdat` | File name and case |
| `EHF_PATH_TEMPLATE` | `ehf_{state}.sas7bdat` | Per-state naming convention |
| `ECF_SEIN_PATH_TEMPLATE` | `ecf_{state}_sein.sas7bdat` | Per-state naming convention |
| `USE_PARQUET` | `False` | Set to `True` if `parquet/` dir exists (much faster) |
| `ALL_STATES` | 51 FIPS codes | Drop any not approved for this project |
| `COL_NAICS` | `mode_naics2017fnl_emp` | NAICS vintage available in your ECF release |
| Column names (`COL_*`) | lowercase | If SAS returns UPPERCASE, the loader handles this — but verify |

## Diagnostic output to expect

`test_model1.py` prints:
- Raw row count from ICF read
- Columns and dtypes (sanity-check column names match config)
- First 3 rows (sanity-check encodings — `dob` should be datetime)
- Education distribution (verify the 1/2/3/4 split looks like LEHD)
- 5-fold CV: RMSE, MAE, R², accuracy, adjacent accuracy
- Per-class precision/recall and confusion matrix

Expected ballpark from prior LEHD evaluations (Census reports 26-47% per-class
accuracy with limited-information imputation): demographics-only XGBoost should
land near or modestly above this baseline.

## Pipeline steps

`run_all.py` orchestrates three steps that can run independently:

| Step | What it does | Output |
|------|-------------|--------|
| `prep` | Read LEHD files, engineer features, save to disk | `prepared_model{1,2,3}.npz`, `educ_c.npy` |
| `train` | 5-fold stratified CV, train final models | `model{1,2,3}.pkl`, `cv_metrics.csv`, `confusion_*.csv`, `model_comparison.csv` |
| `predict` | Apply best model to imputation set | `predictions_model{ml}.csv`, `pred_distribution.csv` |

Use `--steps prep` first to verify data loading without paying for training.

## Activation reminder

The compute nodes do NOT have Anaconda on `PATH` by default. Every interactive
session and every PBS job must start with:

```bash
source /apps/anaconda/bin/activate py3cf
```

`py3cf` is the conda-forge environment with the most complete package set —
verified to include `xgboost`, `numpy`, `scipy`, `pandas`, `scikit-learn`.
If `py3cf` is missing something, try `py3` or list available envs with
`conda env list`.

## Troubleshooting

**`ModuleNotFoundError: No module named 'screening'`**
You forgot `export PYTHONPATH=.` after `cd /projects/<id>/programs`.

**`ModuleNotFoundError: No module named 'xgboost'`**
You're in the wrong conda env. Run `conda env list` and try a different one.

**`TypeError: integer subtraction with datetime`**
The `dob` column is a datetime, not a year integer. The loader handles this
via `dob.dt.year` — if you see this error, check that you're running the
current version of `features.py` / `test_model1.py`.

**SAS file column names look uppercase or wrong**
The loader does `chunk.columns = chunk.columns.str.lower()` automatically.
If columns still don't match, edit the `COL_*` constants in `config.py`.

**Out of memory (exit code 137)**
Request more memory in the PBS submission, or use `--debug` mode (100K PIKs)
or `--states 24` (single state).
