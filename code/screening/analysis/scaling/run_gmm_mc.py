#!/usr/bin/env python3
"""Monte Carlo bias distribution for GMM moment specifications.

Runs R replications of the full DGP (new firms, equilibrium, workers, MLE, GMM)
to characterize the sampling distribution of estimator bias across specs.

Usage::

    cd code/
    python -m screening.analysis.scaling.run_gmm_mc --smoke          # M=5, 3 reps
    python -m screening.analysis.scaling.run_gmm_mc --n_reps 20      # full run (~24h)
"""

from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = os.environ.get(_var, "1")
os.environ["XLA_FLAGS"] = os.environ.get(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false"
)
os.environ["JAX_ENABLE_X64"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Import the single-run pipeline
import screening.analysis.scaling.run_gmm_moments as gmm_mod
from screening.analysis.scaling.run_gmm_moments import (
    TRUE_PARAMS, TRUE_TG, PROJ_ROOT, SPECS,
    _run, generate_data, run_mle,
    _load_setup_data, _augment_markets, _naive_alpha_sigma_e,
    _run_one_spec,
)

BASE_DIR = Path("/tmp/gmm_mc")

# Default specs: just-identified ones that actually recovered parameters
DEFAULT_SPECS = "C_amenity,CS_amenity,CP_amenity"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo bias distribution for GMM moments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--smoke", action="store_true",
                   help="Quick test: M=5, 3 reps")
    p.add_argument("--base_dir", type=str, default=str(BASE_DIR))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n_reps", type=int, default=20)
    p.add_argument("--M", type=int, default=100)
    p.add_argument("--N", type=int, default=2500)
    p.add_argument("--J", type=int, default=100)
    p.add_argument("--mpi_np", type=int, default=20)
    p.add_argument("--specs", type=str, default=DEFAULT_SPECS,
                   help="Comma-separated spec names to run")
    p.add_argument("--setup", type=str, default="both",
                   choices=["both", "true", "estimated"],
                   help="Which setup(s) to run")
    p.add_argument("--skip_to_rep", type=int, default=0,
                   help="Resume from this rep number (skip earlier reps)")
    return p.parse_args()


def select_specs(spec_names: str) -> list:
    """Filter SPECS list to selected names."""
    names = [s.strip() for s in spec_names.split(",")]
    selected = [s for s in SPECS if s["name"] in names]
    if len(selected) != len(names):
        found = {s["name"] for s in selected}
        missing = set(names) - found
        raise ValueError(f"Unknown specs: {missing}. Available: {[s['name'] for s in SPECS]}")
    return selected


def run_one_rep(
    rep: int,
    seed: int,
    M: int, N: int, J: int,
    mpi_np: int,
    selected_specs: list,
    setups: List[str],
    base_dir: Path,
) -> List[dict]:
    """Run one full replication: data gen + MLE + GMM."""
    rep_dir = base_dir / f"rep_{rep:03d}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Point the gmm_mod module at this rep's directory
    gmm_mod.BASE_DIR = rep_dir

    rep_t0 = time.perf_counter()
    results = []

    # 1a. Generate full DGP
    t0 = time.perf_counter()
    generate_data(M, N, J, seed=seed)
    data_time = time.perf_counter() - t0

    # 1b. MLE (only if estimated setup needed)
    mle_time = 0.0
    run_estimated = "estimated" in setups
    if run_estimated:
        t0 = time.perf_counter()
        run_mle(M, mpi_np=mpi_np)
        mle_time = time.perf_counter() - t0

    # 1c. GMM for each setup
    for setup_name in setups:
        use_true = (setup_name == "true")
        try:
            market_list, meta, tau_fixed, tg_fixed = _load_setup_data(use_true)
        except Exception as e:
            print(f"    Failed to load {setup_name} data: {e}")
            continue

        _augment_markets(market_list)
        tp = meta["true_params"]

        if use_true:
            theta0 = np.array([tp["alpha"], tp["sigma_e"]], dtype=np.float64)
        else:
            theta0 = _naive_alpha_sigma_e(market_list, tg_fixed)

        for spec in selected_specs:
            try:
                res = _run_one_spec(spec, market_list, tau_fixed, tg_fixed, theta0)
                res["rep"] = rep
                res["seed"] = seed
                res["setup"] = setup_name
                res["M"] = M
                res["J_total"] = meta["total_J"]
                res["alpha_err"] = res["alpha_hat"] - TRUE_PARAMS["alpha"]
                res["sigma_e_err"] = res["sigma_e_hat"] - TRUE_PARAMS["sigma_e"]
                results.append(res)
            except Exception as e:
                print(f"    {setup_name}/{spec['name']} FAILED: {e}")
                results.append({
                    "rep": rep, "seed": seed, "setup": setup_name,
                    "spec_name": spec["name"], "spec_label": spec["label"],
                    "K": 0, "df": 0,
                    "alpha_hat": np.nan, "sigma_e_hat": np.nan,
                    "alpha_err": np.nan, "sigma_e_err": np.nan,
                    "alpha_init": float(theta0[0]), "sigma_e_init": float(theta0[1]),
                    "Q_final": np.nan, "J_stat": None, "p_value": None,
                    "converged": False, "n_iter": 0, "n_fev": 0,
                    "time_s": 0, "M": M, "J_total": meta.get("total_J", 0),
                })

    rep_time = time.perf_counter() - rep_t0

    # Print summary line
    for r in results:
        tag = f"{r['setup']}/{r['spec_name']}"
        cvg = "Y" if r.get("converged") else "N"
        print(f"    {tag:25s}  a={r['alpha_hat']:.4f}({r['alpha_err']:+.4f})  "
              f"se={r['sigma_e_hat']:.4f}({r['sigma_e_err']:+.4f})  "
              f"cvg={cvg}  {r['time_s']:.0f}s")

    print(f"  Rep {rep} total: {rep_time:.0f}s "
          f"(data={data_time:.0f}s, mle={mle_time:.0f}s)")

    return results


def summarize(all_results: List[dict], base_dir: Path):
    """Compute and print bias distribution summary."""
    df = pd.DataFrame(all_results)
    df = df.dropna(subset=["alpha_hat"])

    if len(df) == 0:
        print("\nNo valid results to summarize.")
        return

    n_reps = df["rep"].nunique()

    print(f"\n{'='*70}")
    print(f"Monte Carlo Summary ({n_reps} replications)")
    print(f"{'='*70}")
    print(f"True: alpha={TRUE_PARAMS['alpha']}, sigma_e={TRUE_PARAMS['sigma_e']}")

    # Summary table
    hdr = ("| Setup | Spec | param | mean | bias | std | RMSE "
           "| p5 | median | p95 |")
    sep = "|" + "|".join(["---"] * 10) + "|"
    print(f"\n{hdr}")
    print(sep)

    summary_rows = []
    for setup in ["true", "estimated"]:
        for spec_name in df["spec_name"].unique():
            sub = df[(df["setup"] == setup) & (df["spec_name"] == spec_name)]
            if len(sub) == 0:
                continue

            for param, true_val in [("alpha", TRUE_PARAMS["alpha"]),
                                     ("sigma_e", TRUE_PARAMS["sigma_e"])]:
                hat = sub[f"{param}_hat"]
                err = hat - true_val
                bias = err.mean()
                std = err.std()
                rmse = np.sqrt((err**2).mean())
                p5, med, p95 = hat.quantile([0.05, 0.5, 0.95])

                cells = [
                    setup, spec_name, param,
                    f"{hat.mean():.4f}", f"{bias:+.4f}", f"{std:.4f}",
                    f"{rmse:.4f}", f"{p5:.4f}", f"{med:.4f}", f"{p95:.4f}",
                ]
                print("| " + " | ".join(cells) + " |")

                summary_rows.append({
                    "setup": setup, "spec": spec_name, "param": param,
                    "mean": float(hat.mean()), "bias": float(bias),
                    "std": float(std), "rmse": float(rmse),
                    "p5": float(p5), "median": float(med), "p95": float(p95),
                    "n_reps": len(sub),
                })

    # Save per-rep CSV
    csv_path = base_dir / "mc_results.csv"
    # Clean NaN/None for CSV
    df_out = df[["rep", "seed", "setup", "spec_name",
                 "alpha_hat", "sigma_e_hat", "alpha_err", "sigma_e_err",
                 "Q_final", "converged", "n_iter", "time_s"]].copy()
    df_out.to_csv(csv_path, index=False)
    print(f"\nPer-rep results: {csv_path}")

    # Save summary JSON
    json_path = base_dir / "mc_summary.json"
    summary = {
        "true_params": {
            "alpha": TRUE_PARAMS["alpha"],
            "sigma_e": TRUE_PARAMS["sigma_e"],
        },
        "n_reps": n_reps,
        "summary": summary_rows,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {json_path}")


def main():
    args = parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    M = 5 if args.smoke else args.M
    N = args.N
    J = args.J
    n_reps = 3 if args.smoke else args.n_reps

    selected_specs = select_specs(args.specs)
    spec_names = [s["name"] for s in selected_specs]

    if args.setup == "both":
        setups = ["true", "estimated"]
    else:
        setups = [args.setup]

    print(f"GMM Monte Carlo: M={M}, N={N}, J={J}, n_reps={n_reps}")
    print(f"Specs: {spec_names}")
    print(f"Setups: {setups}")
    print(f"Base dir: {base_dir}")

    total_t0 = time.perf_counter()
    all_results = []

    # Load existing results if resuming
    csv_path = base_dir / "mc_results.csv"
    if args.skip_to_rep > 0 and csv_path.exists():
        prev_df = pd.read_csv(csv_path)
        all_results = prev_df.to_dict("records")
        print(f"Loaded {len(all_results)} results from previous run (reps 0-{args.skip_to_rep-1})")

    for rep in range(args.skip_to_rep, n_reps):
        rep_seed = args.seed + rep * 100000

        print(f"\n{'='*60}")
        print(f"Replication {rep+1}/{n_reps} (seed={rep_seed})")
        print(f"{'='*60}")

        try:
            rep_results = run_one_rep(
                rep=rep, seed=rep_seed,
                M=M, N=N, J=J, mpi_np=args.mpi_np,
                selected_specs=selected_specs,
                setups=setups,
                base_dir=base_dir,
            )
            all_results.extend(rep_results)
        except Exception as e:
            print(f"  Rep {rep} FAILED: {e}")
            import traceback; traceback.print_exc()
            continue

        # Save intermediate results after each rep
        df_tmp = pd.DataFrame(all_results)
        df_tmp_out = df_tmp[["rep", "seed", "setup", "spec_name",
                             "alpha_hat", "sigma_e_hat", "alpha_err", "sigma_e_err",
                             "Q_final", "converged", "n_iter", "time_s"]].copy()
        df_tmp_out.to_csv(csv_path, index=False)

        elapsed = time.perf_counter() - total_t0
        reps_done = rep - args.skip_to_rep + 1
        reps_left = n_reps - rep - 1
        if reps_done > 0 and reps_left > 0:
            avg_per_rep = elapsed / reps_done
            eta = avg_per_rep * reps_left
            print(f"  Elapsed: {elapsed/60:.0f}min, "
                  f"avg/rep: {avg_per_rep/60:.0f}min, "
                  f"ETA: {eta/60:.0f}min ({eta/3600:.1f}h)")

    # Final summary
    total_time = time.perf_counter() - total_t0
    print(f"\nTotal MC time: {total_time/60:.0f}min ({total_time/3600:.1f}h)")
    summarize(all_results, base_dir)


if __name__ == "__main__":
    main()
