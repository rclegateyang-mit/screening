"""Full-sample prediction and output for education imputation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from screening.rdc.helpers import log, log_section, time_block

from . import config as C
from .train import _bin_predictions


def predict_full_sample(model, X: np.ndarray, piks: np.ndarray,
                        logfile: str | None = None) -> pd.DataFrame:
    """Run model.predict on the full prediction set.

    Returns DataFrame with columns: pik, educ_years_pred, educ_c_pred.
    """
    log_section("Full-sample prediction", logfile)
    with time_block("Predict", logfile):
        y_pred = model.predict(X).astype(np.float32)

    cats = _bin_predictions(y_pred)

    result = pd.DataFrame({
        C.COL_PIK: piks,
        "educ_years_pred": y_pred,
        "educ_c_pred": cats,
    })

    for cat in [1, 2, 3, 4]:
        share = (cats == cat).mean()
        log(f"  Category {cat} ({C.EDUC_LABELS[cat]}): {share:.1%}", logfile)

    return result


def save_predictions(predictions: pd.DataFrame, output_path: str,
                     logfile: str | None = None) -> None:
    """Save full prediction set to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    log(f"  Saved predictions: {output_path} ({len(predictions):,} rows)", logfile)


def save_prediction_summary(predictions: pd.DataFrame,
                            icf: pd.DataFrame,
                            output_dir: str,
                            logfile: str | None = None) -> None:
    """Save summary statistics of predictions by demographic subgroup."""
    log_section("Prediction summary", logfile)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Overall category distribution
    cats = predictions["educ_c_pred"]
    overall = pd.DataFrame({
        "category": [1, 2, 3, 4],
        "label": [C.EDUC_LABELS[c] for c in [1, 2, 3, 4]],
        "count": [(cats == c).sum() for c in [1, 2, 3, 4]],
        "share": [(cats == c).mean() for c in [1, 2, 3, 4]],
        "mean_pred_years": [
            predictions.loc[cats == c, "educ_years_pred"].mean() for c in [1, 2, 3, 4]
        ],
    })
    overall.to_csv(out / "pred_distribution.csv", index=False)

    # Summary by sex and race (join back to ICF for demographics)
    piks = predictions[C.COL_PIK]
    demo = icf.loc[icf.index.isin(piks), [C.COL_SEX, C.COL_RACE]].copy()
    demo = demo.loc[~demo.index.duplicated()]
    merged = predictions.set_index(C.COL_PIK).join(demo, how="left")

    for group_col, label in [(C.COL_SEX, "sex"), (C.COL_RACE, "race")]:
        if group_col not in merged.columns:
            continue
        grp = merged.groupby(group_col).agg(
            n=("educ_years_pred", "count"),
            mean_years=("educ_years_pred", "mean"),
            std_years=("educ_years_pred", "std"),
            share_ba=("educ_c_pred", lambda x: (x == 4).mean()),
            share_lths=("educ_c_pred", lambda x: (x == 1).mean()),
        ).reset_index()
        grp.to_csv(out / f"pred_by_{label}.csv", index=False)

    log(f"  Prediction summaries saved to {out}", logfile)
