#!/usr/bin/env python3
"""Scaling test for distributed augmented-Lagrangian hybrid estimator.

Runs the distributed hybrid AugLag solver at P=50 MPI ranks across
M = 50, 100, 250, 500, 1000 markets using the data_scaling dataset.

Collects parameter recovery, delta/tilde_q RMSE+correlation, timings,
outer iterations, and inner convergence rate.

Usage::

    cd code
    python -m tests.test_auglag_scaling [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("/proj/screening/rcly/data_v2")
RESULTS_DIR = DATA_DIR / "scaling_results" / "auglag"
P = 40  # MPI ranks — 40 out of 96 cores ≈ 42%
        # (XLA spawns ~250 idle threads per rank; 40 ranks keeps total
        #  CPU load under 60% despite scheduler overhead from threads)
M_LIST = [50, 100, 250, 500, 1000]

# Solver settings (max_outer_iter scaled by M to bound total runtime)
INNER_MAXITER = 200
INNER_TOL = 1e-6
GLOBAL_MAXITER = 100
GLOBAL_TOL = 1e-5

# Per-M settings: (max_outer_iter, timeout_minutes)
# Iteration time scales linearly with markets_per_rank (~3 min/iter at 1 mkt/rank)
M_SETTINGS = {
    50:   (30, 180),
    100:  (20, 180),
    250:  (10, 240),
    500:  (8,  360),
    1000: (5,  420),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cell(M: int, out_dir: Path) -> dict:
    """Launch one mpirun and parse the JSON results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "hybrid_auglag_distributed_estimates.json"

    firms = str(DATA_DIR / "clean" / "equilibrium_firms.csv")
    workers = str(DATA_DIR / "build" / "workers_dataset.csv")
    params = str(DATA_DIR / "raw" / "parameters_effective.csv")

    max_outer, timeout_min = M_SETTINGS.get(M, (10, 240))

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                        "intra_op_parallelism_threads=1")
    env["TF_NUM_INTEROP_THREADS"] = "1"
    env["TF_NUM_INTRAOP_THREADS"] = "1"
    env["SCREENING_DATA_DIR"] = str(DATA_DIR)

    cmd = [
        "mpirun", "--oversubscribe", "--bind-to", "core", "-np", str(P),
        sys.executable, "-m", "screening.analysis.auglag.run_distributed",
        "--firms_path", firms,
        "--workers_path", workers,
        "--params_path", params,
        "--out_dir", str(out_dir),
        "--M", str(M),
        "--max_outer_iter", str(max_outer),
        "--inner_maxiter", str(INNER_MAXITER),
        "--inner_tol", str(INNER_TOL),
        "--global_maxiter", str(GLOBAL_MAXITER),
        "--global_tol", str(GLOBAL_TOL),
    ]

    timeout_sec = timeout_min * 60
    log_path = out_dir / "stdout.log"

    print(f"\n{'='*60}")
    print(f"  M={M}, P={P}, max_outer={max_outer}  (timeout={timeout_min}min)")
    print(f"  cmd: {' '.join(cmd[:8])} ...")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd, env=env, cwd=str(Path(__file__).resolve().parents[1]),
            stdout=log_f, stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )

    wall = time.perf_counter() - t0
    print(f"  Finished in {wall/60:.1f} min  (exit={proc.returncode})")

    if proc.returncode != 0 or not json_path.exists():
        print(f"  FAILED — see {log_path}")
        return {"M": M, "P": P, "error": f"exit={proc.returncode}",
                "wall_min": wall / 60}

    with open(json_path) as f:
        res = json.load(f)

    return res


def compute_metrics(res: dict) -> dict:
    """Extract all metrics from the JSON result."""
    if "error" in res:
        return res

    tp = res["true_params"]
    tG = res["theta_G"]
    names = ["tau", "tilde_gamma", "alpha", "sigma_e", "eta", "gamma0"]
    true_vec = [tp["tau"], tp["tilde_gamma"], tp["alpha"],
                tp["sigma_e"], tp["eta"], tp["gamma0"]]

    # Per-parameter recovery
    param_recovery = {}
    for name, hat, true in zip(names, tG, true_vec):
        param_recovery[name] = {"hat": hat, "true": true, "err": hat - true}

    # Delta and tilde_q metrics: load inner estimates + firm data
    inner_path = Path(res.get("_out_dir", "")) / "hybrid_auglag_distributed_inner_estimates.csv"
    firms_path = DATA_DIR / "clean" / "equilibrium_firms.csv"

    delta_corrs, delta_rmses = [], []
    tq_corrs, tq_rmses = [], []

    if inner_path.exists() and firms_path.exists():
        inner_df = pd.read_csv(inner_path)
        firms_df = pd.read_csv(firms_path)

        eta_hat = tG[4]
        gamma0_hat = tp["gamma0"]
        sigma_e_hat = tp["sigma_e"]

        qbar_col = "qbar" if "qbar" in firms_df.columns else "c"

        for mid in sorted(inner_df["market_id"].unique()):
            idf = inner_df[inner_df["market_id"] == mid].sort_values("firm_j")
            fdf = firms_df[firms_df["market_id"] == mid].sort_values("firm_id")

            if len(idf) == 0 or len(fdf) == 0:
                continue

            delta_hat = idf["delta_hat"].values
            tq_hat = idf["tilde_q_hat"].values

            w_true = fdf["w"].values.astype(np.float64)
            xi_true = fdf["xi"].values.astype(np.float64)
            qbar_true = fdf[qbar_col].values.astype(np.float64)

            delta_true = tp["eta"] * np.log(np.maximum(w_true, 1e-300)) + xi_true
            ln_qbar_true = np.log(np.maximum(qbar_true, 1e-300))
            tq_true = (ln_qbar_true - gamma0_hat) / sigma_e_hat

            if len(delta_hat) == len(delta_true) and len(delta_hat) > 1:
                delta_corrs.append(float(np.corrcoef(delta_hat, delta_true)[0, 1]))
                delta_rmses.append(float(np.sqrt(np.mean((delta_hat - delta_true) ** 2))))
                tq_corrs.append(float(np.corrcoef(tq_hat, tq_true)[0, 1]))
                tq_rmses.append(float(np.sqrt(np.mean((tq_hat - tq_true) ** 2))))

    # Inner convergence on final outer iteration
    history = res.get("history", [])
    if history:
        last = history[-1]
        inner_conv_final = last.get("n_converged", 0)
        inner_conv_rate = inner_conv_final / max(res["M"], 1)
    else:
        inner_conv_final = 0
        inner_conv_rate = 0.0

    timings = res.get("timings", {})

    return {
        "M": res["M"],
        "P": res["mpi_ranks"],
        "total_J": res["total_J"],
        "converged": res["converged"],
        "n_outer_iters": res["n_outer_iters"],
        "obj": res["objective"],
        "nll": res["nll"],
        "g_bar_norm": float(np.linalg.norm(res["g_bar"])),
        # param recovery
        **{f"{n}_hat": param_recovery[n]["hat"] for n in names},
        **{f"{n}_true": param_recovery[n]["true"] for n in names},
        **{f"{n}_err": param_recovery[n]["err"] for n in names},
        # delta / tilde_q
        "delta_corr_mean": float(np.mean(delta_corrs)) if delta_corrs else None,
        "delta_rmse_mean": float(np.mean(delta_rmses)) if delta_rmses else None,
        "tq_corr_mean": float(np.mean(tq_corrs)) if tq_corrs else None,
        "tq_rmse_mean": float(np.mean(tq_rmses)) if tq_rmses else None,
        # inner convergence
        "inner_conv_final": inner_conv_final,
        "inner_conv_rate": inner_conv_rate,
        # timings
        "build_s": timings.get("build_time_sec", None),
        "solve_s": timings.get("solve_time_sec", None),
        "wall_s": timings.get("total_time_sec", None),
        "wall_min": timings.get("total_time_sec", 0) / 60,
    }


def write_summary_md(all_metrics: list, out_path: Path) -> None:
    """Write the results markdown table."""
    lines = []
    lines.append("# Distributed Augmented-Lagrangian Scaling Results")
    lines.append("")
    lines.append(f"**Solver:** `run_distributed_hybrid_auglag.py` (naive pooled init)")
    lines.append(f"**MPI ranks:** P={P} (of 96 cores)")
    lines.append(f"**Dataset:** `data_v2/` — N=2000/market, J=69-84/market")
    lines.append(f"**Settings:** inner_maxiter={INNER_MAXITER}, "
                 f"inner_tol={INNER_TOL}, global_maxiter={GLOBAL_MAXITER}")
    lines.append("")

    # --- Parameter recovery table ---
    lines.append("## Parameter Recovery")
    lines.append("")
    names = ["tau", "tilde_gamma", "alpha", "sigma_e", "eta", "gamma0"]
    header = "| M | Conv | Outer | " + " | ".join(names) + " |"
    sep = "|---:|:---:|---:|" + "|".join(["---:"] * len(names)) + "|"
    lines.append(header)
    lines.append(sep)

    for m in all_metrics:
        if "error" in m:
            lines.append(f"| {m['M']} | FAIL | — | " + " | ".join(["—"] * len(names)) + " |")
            continue
        conv_str = "Y" if m["converged"] else "N"
        errs = []
        for n in names:
            e = m.get(f"{n}_err")
            if e is not None:
                errs.append(f"{e:+.4f}")
            else:
                errs.append("—")
        lines.append(f"| {m['M']} | {conv_str} | {m['n_outer_iters']} | "
                     + " | ".join(errs) + " |")

    lines.append("")
    lines.append("*Errors = hat - true*")
    lines.append("")

    # --- Parameter estimates table ---
    lines.append("## Parameter Estimates")
    lines.append("")
    header2 = "| M | " + " | ".join(f"{n} (hat)" for n in names) + " |"
    sep2 = "|---:|" + "|".join(["---:"] * len(names)) + "|"
    lines.append(header2)
    lines.append(sep2)

    # True values row
    first_good = next((m for m in all_metrics if "error" not in m), None)
    if first_good:
        trues = [f"{first_good[f'{n}_true']:.4f}" for n in names]
        lines.append("| true | " + " | ".join(trues) + " |")

    for m in all_metrics:
        if "error" in m:
            lines.append(f"| {m['M']} | " + " | ".join(["—"] * len(names)) + " |")
            continue
        hats = [f"{m[f'{n}_hat']:.4f}" for n in names]
        lines.append(f"| {m['M']} | " + " | ".join(hats) + " |")

    lines.append("")

    # --- Local parameter recovery ---
    lines.append("## Local Parameter Recovery (market-averaged)")
    lines.append("")
    lines.append("| M | delta_corr | delta_RMSE | tq_corr | tq_RMSE |")
    lines.append("|---:|---:|---:|---:|---:|")
    for m in all_metrics:
        if "error" in m:
            lines.append(f"| {m['M']} | — | — | — | — |")
            continue
        dc = f"{m['delta_corr_mean']:.4f}" if m.get("delta_corr_mean") is not None else "—"
        dr = f"{m['delta_rmse_mean']:.4f}" if m.get("delta_rmse_mean") is not None else "—"
        tc = f"{m['tq_corr_mean']:.4f}" if m.get("tq_corr_mean") is not None else "—"
        tr = f"{m['tq_rmse_mean']:.4f}" if m.get("tq_rmse_mean") is not None else "—"
        lines.append(f"| {m['M']} | {dc} | {dr} | {tc} | {tr} |")

    lines.append("")

    # --- Timing and convergence ---
    lines.append("## Timing and Convergence")
    lines.append("")
    lines.append("| M | Build (s) | Solve (s) | Wall (min) | Outer iters | Inner conv (final) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for m in all_metrics:
        if "error" in m:
            lines.append(f"| {m['M']} | — | — | {m.get('wall_min', 0):.1f} | — | — |")
            continue
        bs = f"{m['build_s']:.1f}" if m.get("build_s") is not None else "—"
        ss = f"{m['solve_s']:.1f}" if m.get("solve_s") is not None else "—"
        wm = f"{m['wall_min']:.1f}" if m.get("wall_min") is not None else "—"
        oi = str(m["n_outer_iters"])
        ic = f"{m['inner_conv_final']}/{m['M']} ({m['inner_conv_rate']:.0%})"
        lines.append(f"| {m['M']} | {bs} | {ss} | {wm} | {oi} | {ic} |")

    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan without running")
    parser.add_argument("--M_list", type=str, default=None,
                        help="Override M list (comma-separated)")
    args = parser.parse_args()

    m_list = [int(x) for x in args.M_list.split(",")] if args.M_list else M_LIST

    print(f"Augmented-Lagrangian Scaling Test")
    print(f"  P={P}, M_list={m_list}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Results: {RESULTS_DIR}")

    if args.dry_run:
        for M in m_list:
            max_outer, tout = M_SETTINGS.get(M, (10, 240))
            print(f"  M={M:4d}  P={P}  max_outer={max_outer:3d}  timeout={tout}min  "
                  f"markets/rank={M/P:.1f}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for M in m_list:
        cell_dir = RESULTS_DIR / f"M{M}_P{P}"
        try:
            res = run_cell(M, cell_dir)
            res["_out_dir"] = str(cell_dir)
            metrics = compute_metrics(res)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT for M={M}")
            metrics = {"M": M, "P": P, "error": "timeout",
                       "wall_min": M_SETTINGS.get(M, (10, 240))[1]}
        except Exception as e:
            print(f"  ERROR for M={M}: {e}")
            metrics = {"M": M, "P": P, "error": str(e), "wall_min": 0}

        all_metrics.append(metrics)

        # Save incremental CSV
        csv_path = RESULTS_DIR / "scaling_grid_results.csv"
        pd.DataFrame([metrics]).to_csv(
            csv_path, mode="a", header=not csv_path.exists(), index=False,
        )

    # Write summary markdown
    write_summary_md(all_metrics, RESULTS_DIR / "auglag_scaling_summary.md")


if __name__ == "__main__":
    main()
