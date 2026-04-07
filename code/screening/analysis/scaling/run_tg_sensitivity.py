#!/usr/bin/env python3
"""Tilde-gamma initialization sensitivity test.

Maps the sensitivity of the distributed MLE solver to the initial value of
tilde_gamma (tg = gamma1 / sigma_e).  Generates a single dataset at
(M, N, J), then runs the solver across a grid of tg_init values.

Usage::

    # Smoke test (M=10, grid=[3, 7, 12])
    python -m screening.analysis.scaling.run_tg_sensitivity --smoke

    # Full run (M=100, 13-point grid + naive + true)
    python -m screening.analysis.scaling.run_tg_sensitivity

    # Custom grid
    python -m screening.analysis.scaling.run_tg_sensitivity --grid '[3, 5, 7, 10]'
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parents[3]      # code/ directory
PROJ_ROOT = CODE_DIR.parent                          # project root
DEFAULTS_YAML = PROJ_ROOT / "config" / "defaults.yaml"
OUTPUT_DIR = PROJ_ROOT / "output"
DEFAULT_BASE = OUTPUT_DIR / "scaling" / "tg_sensitivity"

TG_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30]


def load_true_params() -> dict:
    """Read canonical parameters from config/defaults.yaml."""
    with open(DEFAULTS_YAML) as f:
        params = yaml.safe_load(f)
    params["tilde_gamma"] = params["gamma1"] / params["sigma_e"]
    return params


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tilde-gamma initialization sensitivity test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE),
                   help="Output root directory")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--M", type=int, default=100)
    p.add_argument("--N", type=int, default=2500,
                   help="Workers per market")
    p.add_argument("--J", type=int, default=100,
                   help="Firms per market")
    p.add_argument("--mpi_np", type=int, default=None,
                   help="MPI ranks (default: min(M, 48))")
    p.add_argument("--inner_maxiter", type=int, default=500)
    p.add_argument("--inner_tol", type=float, default=1e-7)
    p.add_argument("--outer_maxiter", type=int, default=100)
    p.add_argument("--outer_tol", type=float, default=1e-6)
    p.add_argument("--grid", type=str, default=None,
                   help="JSON override for tg grid, e.g. '[3, 5, 7, 10]'")
    p.add_argument("--skip_data", action="store_true",
                   help="Skip data generation (assume data exists)")
    p.add_argument("--skip_mle", action="store_true",
                   help="Skip MLE runs (only collect results)")
    p.add_argument("--smoke", action="store_true",
                   help="Quick validation: M=10, grid=[3, 7, 12]")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def _run(cmd: List[str], label: str, cwd: Path,
         env_extra: Optional[Dict[str, str]] = None,
         timeout: int = 7200) -> bool:
    """Run subprocess, return True on success."""
    print(f"  [{label}] {' '.join(str(c) for c in cmd[:6])}...")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  [{label}] FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        if result.stdout:
            print(f"  STDOUT (last 2000 chars):\n{result.stdout[-2000:]}")
        if result.stderr:
            print(f"  STDERR (last 2000 chars):\n{result.stderr[-2000:]}")
        return False
    print(f"  [{label}] OK ({elapsed:.1f}s)")
    return True


# ---------------------------------------------------------------------------
# Phase 1: Data generation
# ---------------------------------------------------------------------------


def generate_data(base_dir: Path, M: int, N: int, J: int, seed: int,
                  params: dict) -> bool:
    """Run 01_prep_data -> 02_solve_equilibrium -> 03_draw_workers."""
    data_dir = base_dir / "data"
    raw = data_dir / "raw"
    clean = data_dir / "clean"
    build = data_dir / "build"

    py = sys.executable
    p = params
    data_env = {"SCREENING_DATA_DIR": str(data_dir)}

    # Stage 1: prep data
    if not _run([
        py, "-m", "screening.simulate.01_prep_data",
        "--J", str(J), "--N_workers", str(N), "--M", str(M),
        "--seed", str(seed),
        "--out_dir", str(raw),
        "--tau", str(p["tau"]), "--alpha", str(p["alpha"]),
        "--eta", str(p["eta"]),
        "--gamma0", str(p["gamma0"]), "--gamma1", str(p["gamma1"]),
        "--sigma_e", str(p["sigma_e"]),
        "--mu_v", str(p["mu_v"]), "--sigma_v", str(p["sigma_v"]),
        "--sigma_A", str(p["sigma_A"]),
        "--sigma_xi", str(p["sigma_xi"]), "--rho_Axi", str(p["rho_Axi"]),
        "--sigma_z1", str(p["sigma_z1"]), "--sigma_z2", str(p["sigma_z2"]),
        "--quad_n_x", "50", "--quad_n_y", "50",
        "--conduct_mode", "1",
    ], label="prep_data", cwd=CODE_DIR, env_extra=data_env):
        return False

    # Stage 2: solve equilibrium
    par = min(M, 48)
    if not _run([
        py, "-m", "screening.clean.02_solve_equilibrium",
        "--M", str(M), "--parallel_markets", str(par),
        "--firms_path", str(raw / "firms.csv"),
        "--support_path", str(raw / "support_points.csv"),
        "--params_path", str(raw / "parameters_effective.csv"),
        "--out_dir", str(clean),
        "--conduct_mode", "1", "--use_lsq", "--max_iter", "50000",
    ], label="solve_equil", cwd=CODE_DIR, env_extra=data_env,
       timeout=max(3600, M * 180)):
        return False

    # Stage 3: draw workers
    if not _run([
        py, "-m", "screening.build.03_draw_workers",
        "--M", str(M), "--seed", str(seed),
        "--params_path", str(raw / "parameters_effective.csv"),
        "--firms_path", str(clean / "equilibrium_firms.csv"),
        "--out_dir", str(build),
        "--drop_below_n", "5",
    ], label="draw_workers", cwd=CODE_DIR, env_extra=data_env):
        return False

    return True


# ---------------------------------------------------------------------------
# Phase 2: MLE grid
# ---------------------------------------------------------------------------


def run_mle_one(base_dir: Path, label: str, tg_val: Optional[float],
                M: int, mpi_np: int,
                inner_maxiter: int, inner_tol: float,
                outer_maxiter: int, outer_tol: float) -> bool:
    """Run distributed MLE for one tg_init value."""
    data_dir = base_dir / "data"
    raw = data_dir / "raw"
    clean = data_dir / "clean"
    build = data_dir / "build"
    est_dir = base_dir / f"est_tg_{label}"
    est_dir.mkdir(parents=True, exist_ok=True)

    data_env = {"SCREENING_DATA_DIR": str(data_dir)}

    cmd = [
        "mpirun", "--oversubscribe", "-np", str(mpi_np),
        sys.executable, "-m", "screening.analysis.mle.run_distributed",
        "--firms_path", str(clean / "equilibrium_firms.csv"),
        "--workers_path", str(build / "workers_dataset.csv"),
        "--params_path", str(raw / "parameters_effective.csv"),
        "--out_dir", str(est_dir),
        "--M", str(M),
        "--inner_maxiter", str(inner_maxiter),
        "--inner_tol", str(inner_tol),
        "--outer_maxiter", str(outer_maxiter),
        "--outer_tol", str(outer_tol),
    ]
    if tg_val is not None:
        cmd += ["--tg_init", str(tg_val)]

    return _run(cmd, label=f"MLE tg={label}", cwd=CODE_DIR,
                env_extra=data_env, timeout=max(3600, M * 300))


def parse_mle_result(base_dir: Path, label: str) -> Optional[dict]:
    """Parse MLE result JSON for one run."""
    json_path = base_dir / f"est_tg_{label}" / "mle_distributed_estimates.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase 3: Collect and report
# ---------------------------------------------------------------------------


def collect_row(label: str, tg_init_val: Optional[float],
                result: Optional[dict], true_params: dict) -> dict:
    """Build one row of the summary table."""
    row: Dict[str, Any] = {"label": label}

    if result is None:
        row["status"] = "FAILED"
        return row

    row["status"] = "OK"
    init = result.get("init_values", {})
    rec = result.get("recovery", {})
    hist = result.get("history", [])
    timings = result.get("timings", {})

    row["tg_init"] = init.get("tg_init", tg_init_val)
    row["naive_tg"] = init.get("naive_tg", None)
    row["tg_hat"] = result["theta_G"][1]
    row["tg_err"] = row["tg_hat"] - true_params["tilde_gamma"]
    row["tau_hat"] = result["theta_G"][0]
    row["tau_err"] = row["tau_hat"] - true_params["tau"]
    # Init recovery (before optimization)
    row["d_init_corr"] = init.get("delta_init_corr", None)
    row["d_init_rmse"] = init.get("delta_init_rmse", None)
    row["tq_init_corr"] = init.get("tq_init_corr", None)
    row["tq_init_rmse"] = init.get("tq_init_rmse", None)
    # Final recovery (after optimization)
    row["delta_corr"] = rec.get("delta_corr", 0)
    row["delta_rmse"] = rec.get("delta_rmse", 0)
    row["tq_corr"] = rec.get("tq_corr", 0)
    row["tq_rmse"] = rec.get("tq_rmse", 0)
    row["nll"] = result.get("objective", None)
    row["n_outer_iters"] = result.get("n_outer_iters", 0)
    row["converged"] = result.get("converged", False)
    row["inner_cvg"] = hist[-1]["inner_converged"] if hist else 0
    row["M"] = result.get("M", 0)
    row["wall_s"] = timings.get("total_time_sec", 0)

    return row


def _fmt(val, fmt_str):
    """Format a value, returning '-' for None."""
    if val is None:
        return "-"
    return f"{val:{fmt_str}}"


def print_table(rows: List[dict], true_params: dict) -> str:
    """Print and return markdown summary table."""
    lines: List[str] = []
    lines.append("### Tilde-Gamma Initialization Sensitivity\n")
    hdr = (
        "| tg_init | tg_hat | tg_err | tau_hat | tau_err "
        "| d_init_corr | d_init_RMSE | tq_init_corr | tq_init_RMSE "
        "| d_corr | d_RMSE | tq_corr | tq_RMSE "
        "| NLL | outer_it | inner_cvg/M | cvg | wall_s |"
    )
    sep = "|" + "|".join(["---"] * 18) + "|"
    lines.append(hdr)
    lines.append(sep)

    for r in rows:
        if r["status"] == "FAILED":
            lines.append(f"| {r['label']:>7s} | FAILED |" + " |" * 16)
            continue
        tg_init_s = (f"{r['tg_init']:.2f}" if r["tg_init"] is not None
                     else "naive")
        lines.append(
            f"| {tg_init_s:>7s} "
            f"| {r['tg_hat']:.3f} "
            f"| {r['tg_err']:+.3f} "
            f"| {r['tau_hat']:.4f} "
            f"| {r['tau_err']:+.4f} "
            f"| {_fmt(r['d_init_corr'], '.3f')} "
            f"| {_fmt(r['d_init_rmse'], '.3f')} "
            f"| {_fmt(r['tq_init_corr'], '.3f')} "
            f"| {_fmt(r['tq_init_rmse'], '.3f')} "
            f"| {r['delta_corr']:.3f} "
            f"| {r['delta_rmse']:.3f} "
            f"| {r['tq_corr']:.3f} "
            f"| {r['tq_rmse']:.3f} "
            f"| {r['nll']:.1f} "
            f"| {r['n_outer_iters']:>3d} "
            f"| {r['inner_cvg']}/{r['M']} "
            f"| {'Y' if r['converged'] else 'N'} "
            f"| {r['wall_s']:.0f} |"
        )

    tg_true = true_params["tilde_gamma"]
    lines.append(f"\nTrue: tau={true_params['tau']}, tg={tg_true:.3f} "
                 f"(gamma1={true_params['gamma1']}, sigma_e={true_params['sigma_e']})")

    md = "\n".join(lines)
    print("\n" + md)
    return md


def print_discussion():
    """Print suggested data-driven initialization improvements."""
    print("\n### Suggested Data-Driven Initialization Improvements\n")

    print("**A. Trimmed wage regression** -- Restrict OLS to workers at")
    print("least-selective firms (top quintile by employment count). These")
    print("firms have weak screening, so the wage-skill regression is less")
    print("contaminated by selection. The residual variance better")
    print("approximates true sigma_e, improving tg.\n")

    print("**B. Coarse profile scan** -- Before the main optimization,")
    print("evaluate the NLL at 5-7 coarse tg values (e.g., 2, 4, 6, 8, 10,")
    print("14) with a fast inner solve (inner_maxiter=50). Pick the tg with")
    print("lowest NLL as the starting point. Cost: ~6x one outer evaluation.\n")

    print("**C. Sorting-pattern estimator** -- Use the empirical relationship:")
    print("for each firm j, tq_j ~ tg * v_low_j. Regressing tq_naive_j")
    print("(from FOC) on v_low_j across firms gives a slope estimate of tg")
    print("that is independent of the wage regression.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    true_params = load_true_params()
    TRUE_TG = true_params["tilde_gamma"]

    # Apply smoke overrides
    if args.smoke:
        args.M = 10
        grid = [3.0, 7.0, 12.0]
    elif args.grid is not None:
        grid = json.loads(args.grid)
    else:
        grid = list(TG_GRID)

    if args.mpi_np is None:
        args.mpi_np = min(args.M, 48)

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Build run list: naive + grid + true
    run_list: List[Tuple[str, Optional[float]]] = [("naive", None)]
    for tg in grid:
        run_list.append((f"{tg:.1f}", float(tg)))
    run_list.append(("true", TRUE_TG))

    print("=" * 60)
    print("Tilde-Gamma Initialization Sensitivity Test")
    print("=" * 60)
    print(f"  Base dir:  {base_dir}")
    print(f"  Data:      M={args.M}, N={args.N}, J={args.J}")
    print(f"  MPI ranks: {args.mpi_np}")
    print(f"  Grid:      {[label for label, _ in run_list]}")
    print(f"  True tg:   {TRUE_TG:.4f}")
    if args.skip_data:
        print("  ** Skipping data generation **")
    if args.skip_mle:
        print("  ** Skipping MLE runs **")
    print()

    total_t0 = time.perf_counter()

    # ---- Phase 1: Data generation ----
    if not args.skip_data:
        print("-" * 60)
        print("Phase 1: Data generation")
        print("-" * 60)
        t0 = time.perf_counter()
        ok = generate_data(
            base_dir, args.M, args.N, args.J, args.seed, true_params)
        if not ok:
            print("Data generation FAILED. Aborting.")
            return
        print(f"Data generation complete ({time.perf_counter() - t0:.0f}s)\n")

    # ---- Phase 2: MLE grid ----
    if not args.skip_mle:
        print("-" * 60)
        print("Phase 2: MLE grid")
        print("-" * 60)
        for i, (label, tg_val) in enumerate(run_list):
            # Skip if result already exists with correct M
            existing = parse_mle_result(base_dir, label)
            if existing is not None and existing.get("M", 0) == args.M:
                print(f"\n[{i + 1}/{len(run_list)}] tg_init = {label}  (cached, M={args.M})")
                continue
            print(f"\n[{i + 1}/{len(run_list)}] tg_init = {label}")
            t0 = time.perf_counter()
            ok = run_mle_one(
                base_dir, label, tg_val,
                M=args.M, mpi_np=args.mpi_np,
                inner_maxiter=args.inner_maxiter, inner_tol=args.inner_tol,
                outer_maxiter=args.outer_maxiter, outer_tol=args.outer_tol,
            )
            elapsed = time.perf_counter() - t0
            if not ok:
                print(f"  MLE FAILED for tg={label} ({elapsed:.0f}s)")
            else:
                print(f"  Done ({elapsed:.0f}s)")

    # ---- Phase 3: Collect and report ----
    print("\n" + "-" * 60)
    print("Phase 3: Results")
    print("-" * 60)

    rows = []
    for label, tg_val in run_list:
        result = parse_mle_result(base_dir, label)
        rows.append(collect_row(label, tg_val, result, true_params))

    table_md = print_table(rows, true_params)
    print_discussion()

    # Save markdown table
    md_path = base_dir / "tg_sensitivity_results.md"
    with open(md_path, "w") as f:
        f.write(table_md + "\n")
    print(f"\nTable written to {md_path}")

    # Save JSON summary
    summary = {
        "grid": [{"label": label, "tg_val": tg_val}
                 for label, tg_val in run_list],
        "rows": rows,
        "true_params": true_params,
        "config": {
            "M": args.M, "N": args.N, "J": args.J,
            "seed": args.seed, "mpi_np": args.mpi_np,
            "inner_maxiter": args.inner_maxiter,
            "inner_tol": args.inner_tol,
            "outer_maxiter": args.outer_maxiter,
            "outer_tol": args.outer_tol,
        },
        "total_time_s": time.perf_counter() - total_t0,
    }
    summary_path = base_dir / "tg_sensitivity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    total_time = time.perf_counter() - total_t0
    print(f"Total wall time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
