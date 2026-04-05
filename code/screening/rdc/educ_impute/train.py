"""Model training, cross-validation, and evaluation for education imputation.

All functions are stateless. Receives prepared feature matrices from features.py.
Output is plain CSV (internal evaluation, no disclosure rounding).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats as sp_stats
from sklearn.model_selection import StratifiedKFold

from screening.rdc.helpers import log, log_section, time_block

from . import config as C


# ---------------------------------------------------------------------------
# XGBoost model factory + categorical helpers
# ---------------------------------------------------------------------------

def _make_model(params: dict | None = None) -> xgb.XGBRegressor:
    """Create XGBRegressor with config defaults."""
    p = dict(C.XGB_PARAMS)
    if params:
        p.update(params)
    return xgb.XGBRegressor(**p)


# ---------------------------------------------------------------------------
# Train / predict split
# ---------------------------------------------------------------------------

def split_train_predict(icf: pd.DataFrame,
                        logfile: str | None = None,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for training (observed educ) and prediction (imputed educ).

    Training:  educ_c_imputed == '1' (observed education)
    Predict:   educ_c_imputed in {'2', '3'} (Census-imputed)
    """
    imp = icf[C.COL_EDUC_IMPUTED]
    train_mask = (imp == "1").values
    predict_mask = imp.isin(["2", "3"]).values

    n_train = train_mask.sum()
    n_pred = predict_mask.sum()
    n_total = len(icf)
    log(f"  Train (observed): {n_train:,} ({n_train/n_total:.1%})", logfile)
    log(f"  Predict (imputed): {n_pred:,} ({n_pred/n_total:.1%})", logfile)
    return train_mask, predict_mask


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray, y: np.ndarray,
                feature_names: list[str] | None = None,
                cat_indices: list[int] | None = None,
                params: dict | None = None,
                logfile: str | None = None):
    """Train XGBRegressor with early stopping on a held-out validation split."""
    model = _make_model(params)

    # Validation split for early stopping
    n = len(y)
    n_val = max(1, int(n * C.VALIDATION_FRACTION))
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    tr_idx, va_idx = idx[n_val:], idx[:n_val]

    with time_block("Train model", logfile):
        model.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[va_idx], y[va_idx])],
                  verbose=False)

    best = getattr(model, "best_iteration", model.n_estimators)
    log(f"  Best iteration: {best}", logfile)
    return model


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(X: np.ndarray, y: np.ndarray,
                   educ_c: np.ndarray,
                   feature_names: list[str] | None = None,
                   cat_indices: list[int] | None = None,
                   n_folds: int = C.N_FOLDS,
                   params: dict | None = None,
                   logfile: str | None = None) -> dict:
    """Stratified K-fold CV with XGBoost early stopping."""
    log_section(f"{n_folds}-fold CV", logfile)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    oof_pred = np.full(len(y), np.nan, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, educ_c)):
        log(f"  Fold {fold+1}/{n_folds}: train {len(train_idx):,}, val {len(val_idx):,}", logfile)
        model = _make_model(params)

        # Early-stopping split from training fold
        rng = np.random.RandomState(42 + fold)
        n_es = max(1, int(len(train_idx) * C.VALIDATION_FRACTION))
        perm = rng.permutation(len(train_idx))
        es_idx = train_idx[perm[:n_es]]
        fit_idx = train_idx[perm[n_es:]]

        with time_block(f"Fold {fold+1}", logfile):
            model.fit(X[fit_idx], y[fit_idx],
                      eval_set=[(X[es_idx], y[es_idx])],
                      verbose=False)
            pred = model.predict(X[val_idx]).astype(np.float32)

        best = getattr(model, "best_iteration", model.n_estimators)
        log(f"    best_iteration={best}", logfile)

        oof_pred[val_idx] = pred
        cont = compute_continuous_metrics(y[val_idx], pred, educ_c[val_idx])
        cat = compute_categorical_metrics(y[val_idx], pred)
        fold_metrics.append({**cont, **cat, "fold": fold + 1})

    # Mean metrics
    keys = [k for k in fold_metrics[0] if k != "fold"]
    mean_metrics = {k: np.mean([fm[k] for fm in fold_metrics]) for k in keys}

    log(f"  Mean RMSE: {mean_metrics['rmse']:.4f}", logfile)
    log(f"  Mean accuracy: {mean_metrics['accuracy']:.4f}", logfile)
    log(f"  Mean adj accuracy: {mean_metrics['adjacent_accuracy']:.4f}", logfile)

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "oof_pred": oof_pred,
    }


# ---------------------------------------------------------------------------
# Continuous metrics
# ---------------------------------------------------------------------------

def compute_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               educ_c: np.ndarray) -> dict:
    """RMSE, MAE, R-squared, mean residual by true category."""
    residuals = y_pred - y_true
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    metrics = {
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "pred_std": float(np.std(y_pred)),
    }

    # Mean residual by true category
    for cat in ["1", "2", "3", "4"]:
        mask = educ_c == cat
        if mask.any():
            metrics[f"mean_resid_cat{cat}"] = float(np.mean(residuals[mask]))
        else:
            metrics[f"mean_resid_cat{cat}"] = np.nan

    return metrics


# ---------------------------------------------------------------------------
# Categorical metrics
# ---------------------------------------------------------------------------

def _bin_predictions(y_pred: np.ndarray) -> np.ndarray:
    """Bin continuous predictions to categories 1-4 using BIN_EDGES."""
    return np.digitize(y_pred, C.BIN_EDGES[1:-1]) + 1  # 1-indexed


def compute_categorical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion matrix, accuracy, adjacent accuracy, per-class precision/recall."""
    # Convert continuous to bins
    true_cat = _bin_predictions(y_true)
    pred_cat = _bin_predictions(y_pred)
    cats = [1, 2, 3, 4]

    # Confusion matrix
    cm = np.zeros((4, 4), dtype=int)
    for i, tc in enumerate(cats):
        for j, pc in enumerate(cats):
            cm[i, j] = int(np.sum((true_cat == tc) & (pred_cat == pc)))

    # Overall accuracy
    accuracy = float(np.mean(true_cat == pred_cat))

    # Adjacent accuracy (correct or +-1)
    adjacent = float(np.mean(np.abs(true_cat - pred_cat) <= 1))

    metrics = {"accuracy": accuracy, "adjacent_accuracy": adjacent}

    # Per-class precision and recall
    for i, cat in enumerate(cats):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        metrics[f"precision_cat{cat}"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics[f"recall_cat{cat}"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Marginal distribution comparison
# ---------------------------------------------------------------------------

def compute_marginal_comparison(y_pred: np.ndarray,
                                train_educ_c: np.ndarray,
                                logfile: str | None = None) -> pd.DataFrame:
    """Compare distribution of predicted categories vs training distribution."""
    pred_cat = _bin_predictions(y_pred)
    cats = [1, 2, 3, 4]

    train_shares = np.array([(train_educ_c == str(c)).mean() for c in cats])
    pred_shares = np.array([(pred_cat == c).mean() for c in cats])

    df = pd.DataFrame({
        "category": cats,
        "label": [C.EDUC_LABELS[c] for c in cats],
        "train_share": train_shares,
        "pred_share": pred_shares,
        "diff": pred_shares - train_shares,
    })
    return df


# ---------------------------------------------------------------------------
# Model comparison (paired tests across folds)
# ---------------------------------------------------------------------------

def compare_models(cv_results: dict[str, dict],
                   logfile: str | None = None) -> pd.DataFrame:
    """Paired t-test of RMSE and accuracy between consecutive models."""
    names = sorted(cv_results.keys())
    rows = []
    for i in range(1, len(names)):
        a, b = names[i - 1], names[i]
        for metric in ["rmse", "accuracy"]:
            vals_a = [fm[metric] for fm in cv_results[a]["fold_metrics"]]
            vals_b = [fm[metric] for fm in cv_results[b]["fold_metrics"]]
            diffs = np.array(vals_b) - np.array(vals_a)
            mean_diff = float(diffs.mean())
            se_diff = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
            t_stat = mean_diff / se_diff if se_diff > 0 else 0.0
            p_val = float(2 * sp_stats.t.sf(abs(t_stat), df=len(diffs) - 1))
            rows.append({
                "comparison": f"{b} vs {a}",
                "metric": metric,
                "mean_diff": mean_diff,
                "se": se_diff,
                "t_stat": t_stat,
                "p_value": p_val,
            })
            sign = "+" if mean_diff > 0 else ""
            log(f"  {b} vs {a} [{metric}]: {sign}{mean_diff:.4f} (p={p_val:.3f})", logfile)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save evaluation output
# ---------------------------------------------------------------------------

def save_evaluation(cv_results: dict[str, dict],
                    comparison: pd.DataFrame,
                    output_dir: str,
                    y_true: np.ndarray | None = None,
                    educ_c_true: np.ndarray | None = None,
                    logfile: str | None = None) -> None:
    """Save all evaluation results to CSV (internal, no disclosure rounding)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CV metrics per fold
    rows = []
    for name, res in cv_results.items():
        for fm in res["fold_metrics"]:
            rows.append({"model": name, **fm})
        rows.append({"model": name, "fold": "mean", **res["mean_metrics"]})
    pd.DataFrame(rows).to_csv(out / "cv_metrics.csv", index=False)
    log(f"  Saved cv_metrics.csv", logfile)

    # Model comparison
    comparison.to_csv(out / "model_comparison.csv", index=False)
    log(f"  Saved model_comparison.csv", logfile)

    # Per-model OOF confusion matrices and marginal comparison
    if y_true is not None:
        for name, res in cv_results.items():
            oof = res["oof_pred"]
            valid = ~np.isnan(oof)
            if not valid.any():
                continue

            # Confusion matrix from OOF predictions
            true_cat = _bin_predictions(y_true[valid])
            pred_cat = _bin_predictions(oof[valid])
            cats = [1, 2, 3, 4]
            cm_rows = []
            for tc in cats:
                for pc in cats:
                    cm_rows.append({
                        "true_cat": tc, "pred_cat": pc,
                        "count": int(np.sum((true_cat == tc) & (pred_cat == pc))),
                    })
            pd.DataFrame(cm_rows).to_csv(out / f"confusion_{name}.csv", index=False)

            # Marginal comparison
            if educ_c_true is not None:
                marg = compute_marginal_comparison(oof[valid], educ_c_true[valid], logfile)
                marg.to_csv(out / f"marginal_{name}.csv", index=False)

        log(f"  Saved confusion matrices and marginal comparisons", logfile)

    log(f"  Evaluation saved to {out}", logfile)
