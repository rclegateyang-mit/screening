#!/usr/bin/env python3
"""Standalone Model 1 test: demographics-only education imputation.

Self-contained — no project dependencies. Type this into the RDC to validate:
  - SAS file reads (column names, byte decoding, case)
  - Age filter and education category mapping
  - XGBoost trains and predicts
  - 5-fold CV produces reasonable metrics

Usage (interactive PBS session):
  qsub -IX
  source /apps/anaconda/bin/activate py3cf
  cd /projects/<PROJECT_ID>/programs
  python test_model1.py --icf-path /data/lehd/current/icf_us.sas7bdat --n 100000
"""

import argparse, time, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# ---- Config (>>> VERIFY THESE <<<) ----

ICF_PATH = "/data/lehd/current/icf_us.sas7bdat"  # >>> VERIFY <<<
REF_YEAR = 2014

# Column names — may be UPPERCASE in SAS; we lowercase after read
COL_PIK = "pik"
COL_DOB = "dob"
COL_SEX = "sex"
COL_POB = "pob"
COL_RACE = "race"
COL_ETHNICITY = "ethnicity"
COL_EDUC = "educ_c"
COL_EDUC_IMP = "educ_c_imputed"

EDUC_TO_YEARS = {"1": 10, "2": 12, "3": 14, "4": 16}
EDUC_LABELS = {1: "<HS", 2: "HS", 3: "SomeCol", 4: "BA+"}
BIN_EDGES = [11, 13, 15]  # for np.digitize: <11->1, 11-13->2, 13-15->3, >=15->4

POB_MAP = {
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8,
    "A": 9, "B":10, "C":11, "D":12, "E":13, "F":14, "G":15, "H":16, "I":17,
    "J":18, "K":19, "L":20, "M":21, "N":22, "O":23, "P":24, "Q":25, "R":26,
    "S":27, "T":28, "U":29, "V":30, "W":31, "X":32, "Y":33, "Z":34,
}

XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
           min_child_weight=100, max_bin=255, tree_method="hist",
           random_state=42, n_jobs=1)
ES_FRAC = 0.1


# ---- Data loading ----

def load_icf(path, n=None):
    """Load ICF_US, filter age 25+, encode features. Returns (X, y, educ_c)."""
    t0 = time.time()
    cols = [COL_PIK, COL_DOB, COL_SEX, COL_POB, COL_RACE,
            COL_ETHNICITY, COL_EDUC, COL_EDUC_IMP]

    # Read in chunks to handle large files
    chunks = []
    for chunk in pd.read_sas(path, chunksize=500_000):
        chunk.columns = chunk.columns.str.lower()
        # Decode byte strings
        for c in chunk.select_dtypes("object").columns:
            s = chunk[c].dropna()
            if len(s) > 0 and isinstance(s.iloc[0], bytes):
                chunk[c] = chunk[c].apply(
                    lambda v: v.decode("utf-8", "replace").strip()
                    if isinstance(v, bytes) else v)
        keep = [c for c in cols if c in chunk.columns]
        chunks.append(chunk[keep])
        if n and sum(len(c) for c in chunks) > n * 3:
            break  # read enough raw rows
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Raw rows read: {len(df):,} ({time.time()-t0:.1f}s)")

    # First print what we got to verify column names
    print(f"  Columns: {list(df.columns)}")
    print(f"  dtypes:\n{df.dtypes}")
    print(f"  Sample values (first 3 rows):")
    print(df.head(3).to_string())

    # Age filter — dob may be datetime (from SAS) or numeric year
    dob = df[COL_DOB]
    if hasattr(dob.dtype, "year") or pd.api.types.is_datetime64_any_dtype(dob):
        df["age"] = REF_YEAR - dob.dt.year
    else:
        df["age"] = REF_YEAR - dob.astype(int)
    df = df[df["age"] >= 25].copy()
    print(f"  After age>=25: {len(df):,}")

    # Keep only observed education (educ_c_imputed == '1') for train
    train = df[df[COL_EDUC_IMP] == "1"].copy()
    print(f"  Observed education: {len(train):,} ({len(train)/len(df):.1%})")

    if n and len(train) > n:
        train = train.iloc[:n].copy()
        print(f"  Subsampled to: {len(train):,}")

    # Target
    train["educ_years"] = train[COL_EDUC].map(EDUC_TO_YEARS).astype("float32")
    missing_target = train["educ_years"].isna().sum()
    if missing_target > 0:
        print(f"  WARNING: {missing_target} rows with unmapped educ_c — dropping")
        train = train.dropna(subset=["educ_years"])

    # Features
    X, feat_names, cat_idx = encode_features(train)
    y = train["educ_years"].values
    educ_c = train[COL_EDUC].values

    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Features: {feat_names}")
    print(f"  Categorical indices: {cat_idx}")
    print(f"  NaN rate in X: {np.isnan(X).mean():.4f}")
    print(f"  Education distribution:")
    for cat, label in EDUC_LABELS.items():
        share = (educ_c == str(cat)).mean()
        print(f"    {cat} ({label}): {share:.1%}")
    return X, y, educ_c, feat_names, cat_idx


def encode_features(df):
    """Encode demographics into feature matrix. Returns (X, names, cat_indices)."""
    parts = []
    names = []

    # Age (continuous)
    parts.append(df["age"].values.astype("float32").reshape(-1, 1))
    names.append("age")

    # Sex (binary)
    parts.append((df[COL_SEX] == "M").values.astype("float32").reshape(-1, 1))
    names.append("sex_M")

    # Ethnicity (binary)
    parts.append((df[COL_ETHNICITY] == "H").values.astype("float32").reshape(-1, 1))
    names.append("ethnicity_H")

    # Race (one-hot)
    for code in ["1", "2", "3", "4", "5", "7"]:
        parts.append((df[COL_RACE] == code).values.astype("float32").reshape(-1, 1))
        names.append(f"race_{code}")

    # POB (ordinal, treated as categorical)
    pob = df[COL_POB].map(POB_MAP).astype("float32").values.reshape(-1, 1)
    parts.append(pob)
    names.append("pob")
    cat_idx = [len(names) - 1]  # pob is categorical

    X = np.hstack(parts)
    return X, names, cat_idx


# ---- Evaluation ----

def evaluate(y_true, y_pred, educ_c):
    """Compute and print all metrics."""
    resid = y_pred - y_true
    rmse = np.sqrt(np.mean(resid**2))
    mae = np.mean(np.abs(resid))
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Categorical
    true_cat = np.digitize(y_true, BIN_EDGES) + 1
    pred_cat = np.digitize(y_pred, BIN_EDGES) + 1
    acc = np.mean(true_cat == pred_cat)
    adj_acc = np.mean(np.abs(true_cat - pred_cat) <= 1)

    print(f"    RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    print(f"    Accuracy={acc:.4f}  Adjacent={adj_acc:.4f}")

    # Mean residual by true category
    for cat in ["1", "2", "3", "4"]:
        m = educ_c == cat
        if m.any():
            print(f"    Mean resid cat {cat}: {np.mean(resid[m]):+.3f}")

    # Confusion matrix
    print("    Confusion matrix (rows=true, cols=pred):")
    print("         ", "  ".join(f"P={c}" for c in [1,2,3,4]))
    for tc in [1, 2, 3, 4]:
        row = [np.sum((true_cat == tc) & (pred_cat == pc)) for pc in [1,2,3,4]]
        total = sum(row)
        pcts = [f"{r/total*100:5.1f}" if total > 0 else "  N/A" for r in row]
        print(f"    T={tc}  {'  '.join(pcts)}%")

    # Per-class recall
    for cat in [1, 2, 3, 4]:
        tp = np.sum((true_cat == cat) & (pred_cat == cat))
        fn = np.sum(true_cat == cat) - tp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"    Recall cat {cat} ({EDUC_LABELS[cat]}): {rec:.3f}")

    return {"rmse": rmse, "mae": mae, "r2": r2, "acc": acc, "adj_acc": adj_acc}


# ---- Cross-validation ----

def run_cv(X, y, educ_c, feat_names, cat_idx, n_folds=5):
    """Stratified K-fold CV. Prints per-fold and mean metrics."""
    print(f"\n{'='*60}")
    print(f"  {n_folds}-fold stratified CV (XGBoost)")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, educ_c)):
        t0 = time.time()

        # Early-stopping split from training fold
        rng = np.random.RandomState(42 + fold)
        n_es = max(1, int(len(tr_idx) * ES_FRAC))
        perm = rng.permutation(len(tr_idx))
        es_idx = tr_idx[perm[:n_es]]
        fit_idx = tr_idx[perm[n_es:]]

        model = xgb.XGBRegressor(**XGB)
        model.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[es_idx], y[es_idx])],
                  verbose=False)
        pred = model.predict(X[va_idx]).astype("float32")
        elapsed = time.time() - t0

        best = getattr(model, "best_iteration", model.n_estimators)
        print(f"\n  Fold {fold+1}/{n_folds} "
              f"(train={len(fit_idx):,} es={n_es:,} val={len(va_idx):,} "
              f"{elapsed:.1f}s iters={best})")
        m = evaluate(y[va_idx], pred, educ_c[va_idx])
        all_metrics.append(m)

    # Mean
    print(f"\n{'='*60}")
    print(f"  MEAN across {n_folds} folds:")
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics]
        print(f"    {k}: {np.mean(vals):.4f} (+/- {np.std(vals):.4f})")
    print(f"{'='*60}")


# ---- Main ----

def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    p = argparse.ArgumentParser(description="Model 1 test: demographics-only XGBoost")
    p.add_argument("--icf-path", default=ICF_PATH, help="Path to ICF_US file")
    p.add_argument("--n", type=int, default=100_000,
                   help="Max training-set PIKs to use (default: 100000)")
    p.add_argument("--folds", type=int, default=5, help="CV folds (default: 5)")
    p.add_argument("--output-dir", default=None, help="Save metrics CSV here")
    args = p.parse_args()

    print(f"Model 1 test: demographics-only education imputation (XGBoost)")
    print(f"  ICF path: {args.icf_path}")
    print(f"  Max PIKs: {args.n:,}")
    print(f"  CV folds: {args.folds}")
    print(f"  XGBoost version: {xgb.__version__}")

    X, y, educ_c, feat_names, cat_idx = load_icf(args.icf_path, n=args.n)
    run_cv(X, y, educ_c, feat_names, cat_idx, n_folds=args.folds)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"\n  Training final model on full sample...")
        model = xgb.XGBRegressor(**XGB)
        model.fit(X, y)
        pred = model.predict(X).astype("float32")
        print(f"  Prediction range: [{pred.min():.1f}, {pred.max():.1f}]")
        pred_cat = np.digitize(pred, BIN_EDGES) + 1
        for cat in [1,2,3,4]:
            print(f"  Predicted cat {cat}: {(pred_cat==cat).mean():.1%}")


if __name__ == "__main__":
    main()
