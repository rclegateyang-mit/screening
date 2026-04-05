"""Orchestrator for education imputation pipeline.

Usage:
    # Debug (interactive PBS session)
    python -m screening.rdc.educ_impute.run_all --project-id XXXX --debug --steps prep,train

    # Production (via submit.py)
    python -m screening.rdc.submit ml_prod \\
        --cmd "python -m screening.rdc.educ_impute.run_all --project-id XXXX" --submit

Steps:
    prep    — Load LEHD files, engineer features, save to disk
    train   — Cross-validate and evaluate all requested models
    predict — Full-sample prediction using best model
    all     — Run prep, train, predict in sequence
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
from pathlib import Path

import numpy as np

from screening.rdc.helpers import log, log_section, log_error, time_block, set_single_threaded

from . import config as C
from . import features as F
from . import train as T
from . import predict as P


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Education imputation pipeline for Census RDC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--project-id", required=True,
                   help="RDC project identifier (for path resolution)")
    p.add_argument("--steps", default="all",
                   help="Comma-separated steps: prep,train,predict,all (default: all)")
    p.add_argument("--models", default="1,2,3",
                   help="Comma-separated model levels to run (default: 1,2,3)")
    p.add_argument("--debug", action="store_true",
                   help="Use small data subset for interactive testing")
    p.add_argument("--ref-year", type=int, default=None,
                   help=f"Override reference year (default: {C.REF_YEAR})")
    p.add_argument("--states", default=None,
                   help="Comma-separated state FIPS codes (default: config)")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory")
    p.add_argument("--logfile", default=None,
                   help="Path for log output file")
    return p.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> dict:
    """Build runtime configuration from args + config.py defaults."""
    cfg = {}
    cfg["project_id"] = args.project_id
    cfg["logfile"] = args.logfile

    # Steps
    if args.steps == "all":
        cfg["steps"] = ["prep", "train", "predict"]
    else:
        cfg["steps"] = [s.strip() for s in args.steps.split(",")]

    # Model levels
    cfg["model_levels"] = [int(m.strip()) for m in args.models.split(",")]

    # Debug mode
    cfg["debug"] = args.debug
    cfg["debug_n"] = C.DEBUG_N_PIKS if args.debug else None

    # States
    if args.states:
        cfg["states"] = [s.strip() for s in args.states.split(",")]
    elif args.debug:
        cfg["states"] = C.DEBUG_STATES
    else:
        cfg["states"] = C.ALL_STATES

    # Reference year
    if args.ref_year:
        C.REF_YEAR = args.ref_year
        C.WINDOW = (args.ref_year - 2, args.ref_year + 2)

    # Output directory
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    else:
        cfg["output_dir"] = C.OUTPUT_DIR.format(project_id=args.project_id)

    return cfg


# ---------------------------------------------------------------------------
# Step: prep
# ---------------------------------------------------------------------------

def run_prep(cfg: dict) -> None:
    """Load LEHD data, engineer features, save to disk."""
    logfile = cfg["logfile"]
    out = cfg["output_dir"]
    max_model = max(cfg["model_levels"])

    # ---- ICF ----
    icf = F.load_icf(debug_n=cfg["debug_n"], logfile=logfile)
    icf, feat_m1, cat_m1 = F.encode_demographics(icf, logfile=logfile)

    # ---- Train/predict split ----
    train_mask, predict_mask = T.split_train_predict(icf, logfile=logfile)

    # Save educ_c for stratified CV (needed by train step for all models)
    Path(out).mkdir(parents=True, exist_ok=True)
    np.save(Path(out) / "educ_c.npy", icf[C.COL_EDUC].values)

    # ---- Model 1: save ----
    if 1 in cfg["model_levels"]:
        X, y, fnames, cat_idx = F.build_feature_matrix(
            icf, feat_m1, cat_m1, model_level=1, logfile=logfile)
        F.save_prepared_data(out, X, y, train_mask, predict_mask,
                             fnames, cat_idx, icf.index.values, 1, logfile)
        del X
        gc.collect()

    # ---- RCF (needed for models 2 and 3) ----
    rcf = None
    tract_agg = None
    if max_model >= 2:
        rcf = F.load_rcf(logfile=logfile)
        tract_agg = F.compute_tract_aggregates(icf, rcf, train_mask, logfile=logfile)

        if 2 in cfg["model_levels"]:
            X, y, fnames, cat_idx = F.build_feature_matrix(
                icf, feat_m1, cat_m1, tract_agg=tract_agg,
                model_level=2, logfile=logfile)
            F.save_prepared_data(out, X, y, train_mask, predict_mask,
                                 fnames, cat_idx, icf.index.values, 2, logfile)
            del X
            gc.collect()

    # ---- Model 3: NAICS features ----
    if max_model >= 3:
        ehf = F.load_ehf_multi_state(cfg["states"], logfile=logfile)
        dom = F.identify_dominant_job(ehf, logfile=logfile)
        del ehf
        gc.collect()

        ecf = F.load_ecf_naics(cfg["states"], logfile=logfile)
        naics_feats = F.compute_naics_features(dom, ecf, logfile=logfile)
        del dom, ecf
        gc.collect()

        # Tract NAICS distribution (load RCF if not already loaded)
        if rcf is None:
            rcf = F.load_rcf(logfile=logfile)
        tract_naics = F.compute_tract_naics_distribution(
            rcf, naics_feats, icf.index[train_mask], logfile=logfile)

        if 3 in cfg["model_levels"]:
            X, y, fnames, cat_idx = F.build_feature_matrix(
                icf, feat_m1, cat_m1, tract_agg=tract_agg,
                naics_feats=naics_feats, tract_naics=tract_naics,
                model_level=3, logfile=logfile)
            F.save_prepared_data(out, X, y, train_mask, predict_mask,
                                 fnames, cat_idx, icf.index.values, 3, logfile)
            del X
            gc.collect()

    log(f"  Prep complete. Output: {out}", logfile)


# ---------------------------------------------------------------------------
# Step: train
# ---------------------------------------------------------------------------

def run_train(cfg: dict) -> dict:
    """Cross-validate and evaluate all requested models. Returns trained models."""
    logfile = cfg["logfile"]
    out = cfg["output_dir"]

    # Load educ_c for stratified splits
    educ_c = np.load(Path(out) / "educ_c.npy", allow_pickle=True)

    cv_results = {}
    models = {}

    for ml in sorted(cfg["model_levels"]):
        log_section(f"Training Model {ml}", logfile)
        data = F.load_prepared_data(out, ml, logfile)

        X_train = data["X"][data["train_mask"]]
        y_train = data["y"][data["train_mask"]]
        ec_train = educ_c[data["train_mask"]]

        # Cross-validate
        cv = T.cross_validate(X_train, y_train, ec_train,
                              feature_names=data["feature_names"],
                              cat_indices=data["cat_indices"],
                              logfile=logfile)
        cv_results[f"model{ml}"] = cv

        # Train final model on full training set
        model = T.train_model(X_train, y_train,
                              feature_names=data["feature_names"],
                              cat_indices=data["cat_indices"],
                              logfile=logfile)
        models[ml] = model

        # Save model
        model_path = Path(out) / f"model{ml}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log(f"  Saved model: {model_path}", logfile)

        del X_train, y_train
        gc.collect()

    # Model comparison
    comparison = T.compare_models(cv_results, logfile)

    # Save evaluation
    # Use model1's data for y_true and educ_c (same across models)
    data = F.load_prepared_data(out, cfg["model_levels"][0], logfile)
    y_true = data["y"][data["train_mask"]]
    ec_true = educ_c[data["train_mask"]]

    T.save_evaluation(cv_results, comparison, out,
                      y_true=y_true, educ_c_true=ec_true,
                      logfile=logfile)

    return models


# ---------------------------------------------------------------------------
# Step: predict
# ---------------------------------------------------------------------------

def run_predict(cfg: dict, models: dict | None = None) -> None:
    """Full-sample prediction using the highest-level model available."""
    logfile = cfg["logfile"]
    out = cfg["output_dir"]

    # Use highest model level
    ml = max(cfg["model_levels"])

    # Load or reuse model
    if models and ml in models:
        model = models[ml]
    else:
        model_path = Path(out) / f"model{ml}.pkl"
        log(f"  Loading model from {model_path}", logfile)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # Load prepared data
    data = F.load_prepared_data(out, ml, logfile)
    X_pred = data["X"][data["predict_mask"]]
    piks_pred = data["piks"][data["predict_mask"]]

    # Predict
    predictions = P.predict_full_sample(model, X_pred, piks_pred, logfile)

    # Save
    P.save_predictions(predictions, str(Path(out) / f"predictions_model{ml}.csv"), logfile)

    # Load ICF for demographic summary (from prepared data PIKs)
    # Re-load ICF minimally for summary
    icf = F.load_icf(debug_n=cfg["debug_n"], logfile=logfile)
    P.save_prediction_summary(predictions, icf, out, logfile)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    set_single_threaded()

    args = parse_args(argv)
    cfg = resolve_config(args)
    logfile = cfg["logfile"]

    log_section("Education Imputation Pipeline", logfile)
    log(f"  Steps: {cfg['steps']}", logfile)
    log(f"  Models: {cfg['model_levels']}", logfile)
    log(f"  States: {len(cfg['states'])} states", logfile)
    log(f"  Debug: {cfg['debug']}", logfile)
    log(f"  Output: {cfg['output_dir']}", logfile)

    models = None

    with time_block("Total pipeline", logfile):
        if "prep" in cfg["steps"]:
            with time_block("Step: prep", logfile):
                run_prep(cfg)

        if "train" in cfg["steps"]:
            with time_block("Step: train", logfile):
                models = run_train(cfg)

        if "predict" in cfg["steps"]:
            with time_block("Step: predict", logfile):
                run_predict(cfg, models)

    log_section("Done", logfile)


if __name__ == "__main__":
    main()
