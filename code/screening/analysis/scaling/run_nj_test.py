#!/usr/bin/env python3
"""N/J Scaling Test for MLE Inner Estimate Recovery.

Systematically varies (J, N) to map recovery quality as a function of N/J.
For each config, generates 1 market via the data environment pipeline,
then runs distributed MLE (inner-only, globals fixed at truth) via
``mpirun -np 1``.

Usage::

    # Smoke test (J=25, N=500)
    python -m screening.analysis.scaling.run_nj_test --configs '[[25, 500]]'

    # Default grid (J=100, N=2k..100k)
    python -m screening.analysis.scaling.run_nj_test

    # Include J variation
    python -m screening.analysis.scaling.run_nj_test --vary_J

    # Also run full outer optimization (not just inner-only)
    python -m screening.analysis.scaling.run_nj_test --no_freeze_globals
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Default test grid: J=100, varying N
DEFAULT_CONFIGS = [
    [100, 2000],     # N/J = 20
    [100, 5000],     # N/J = 50
    [100, 10000],    # N/J = 100
    [100, 25000],    # N/J = 250
    [100, 50000],    # N/J = 500
    [100, 100000],   # N/J = 1000
]

# Additional configs for --vary_J
VARY_J_CONFIGS = [
    [50, 1000],      # N/J = 20
    [50, 5000],      # N/J = 100
    [25, 500],       # N/J = 20
    [25, 2500],      # N/J = 100
]

# True parameters from config/defaults.yaml
TRUE_PARAMS = {
    "tau": 0.4, "alpha": 0.2, "gamma0": -7.2, "gamma1": 0.94,
    "sigma_e": 0.135, "mu_v": 10.84, "sigma_v": 0.25, "eta": 5.0,
    "sigma_xi": 0.35, "sigma_z2": 0.175,
}

# code/ directory (parent of screening/ package)
CODE_DIR = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="N/J scaling test for MLE inner estimate recovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_dir", type=str, default="/tmp/scaling_nj",
                   help="Output root directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inner_maxiter", type=int, default=1000)
    p.add_argument("--inner_tol", type=float, default=1e-8)
    p.add_argument("--outer_maxiter", type=int, default=200)
    p.add_argument("--outer_tol", type=float, default=1e-8)
    p.add_argument("--vary_J", action="store_true",
                   help="Include J=50,25 configs in addition to J=100")
    p.add_argument("--configs", type=str, default=None,
                   help="Explicit JSON list of [J,N] pairs (overrides default grid)")
    p.add_argument("--skip_data", action="store_true",
                   help="Skip data generation, only run MLE (for reruns)")
    p.add_argument("--skip_mle", action="store_true",
                   help="Skip MLE, only run data generation")
    p.add_argument("--no_freeze_globals", action="store_true",
                   help="Run full outer optimization instead of inner-only "
                        "(default: freeze tau, tilde_gamma at true values)")
    p.add_argument("--delta_only", action="store_true",
                   help="Fix tilde_q at true values; only solve for delta "
                        "(implies freeze globals)")
    p.add_argument("--profile_delta", action="store_true",
                   help="Profile out delta via BLP contraction; optimize "
                        "tilde_q only (implies freeze globals)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_cmd(cmd: list, label: str) -> bool:
    """Run a subprocess command from the code/ directory."""
    print(f"  [{label}] {' '.join(str(c) for c in cmd[:6])}...")
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True, text=True,
        cwd=str(CODE_DIR),
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
# Equilibrium convergence check
# ---------------------------------------------------------------------------

def read_equil_meta(cfg_dir: Path) -> dict:
    """Read equilibrium solver metadata (convergence, residual, etc.)."""
    meta_path = cfg_dir / "clean" / "equilibrium_firms.meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Per-config pipeline
# ---------------------------------------------------------------------------

def run_pipeline(J: int, N: int, cfg_dir: Path, seed: int,
                 args: argparse.Namespace) -> dict | None:
    """Run data generation + MLE for one (J, N) config.

    Returns the parsed MLE results JSON (augmented with equil_meta), or None
    on failure.
    """
    tag = f"J{J}_N{N}"
    raw_dir = cfg_dir / "raw"
    clean_dir = cfg_dir / "clean"
    build_dir = cfg_dir / "build"
    est_dir = cfg_dir / "est"
    py = sys.executable
    tp = TRUE_PARAMS

    # ------------------------------------------------------------------
    # Data generation stages
    # ------------------------------------------------------------------
    if not args.skip_data:
        # Stage 1: prep data (M=1)
        if not run_cmd([
            py, "-m", "screening.simulate.01_prep_data",
            "--J", str(J), "--N_workers", str(N), "--M", "1",
            "--seed", str(seed),
            "--out_dir", str(raw_dir),
            "--tau", str(tp["tau"]), "--alpha", str(tp["alpha"]),
            "--eta", str(tp["eta"]),
            "--gamma0", str(tp["gamma0"]), "--gamma1", str(tp["gamma1"]),
            "--sigma_e", str(tp["sigma_e"]),
            "--mu_v", str(tp["mu_v"]), "--sigma_v", str(tp["sigma_v"]),
            "--sigma_xi", str(tp["sigma_xi"]), "--sigma_z2", str(tp["sigma_z2"]),
            "--quad_n_x", "50", "--quad_n_y", "50",
            "--conduct_mode", "1",
        ], f"{tag}/prep"):
            return None

        # Stage 2: solve equilibrium (use LM/trf, more robust than hybr)
        if not run_cmd([
            py, "-m", "screening.clean.02_solve_equilibrium",
            "--firms_path", str(raw_dir / "firms.csv"),
            "--support_path", str(raw_dir / "support_points.csv"),
            "--params_path", str(raw_dir / "parameters_effective.csv"),
            "--out_dir", str(clean_dir),
            "--conduct_mode", "1",
            "--use_lsq",
            "--max_iter", "50000",
            "--tol", "1e-10",
        ], f"{tag}/equil"):
            return None

        # Stage 3: draw workers
        if not run_cmd([
            py, "-m", "screening.build.03_draw_workers",
            "--params_path", str(raw_dir / "parameters_effective.csv"),
            "--firms_path", str(clean_dir / "equilibrium_firms.csv"),
            "--out_dir", str(build_dir),
            "--seed", str(seed),
        ], f"{tag}/workers"):
            return None

        # Post-process: add market_id=1 (distributed MLE requires it)
        for csv_path in [
            clean_dir / "equilibrium_firms.csv",
            build_dir / "workers_dataset.csv",
        ]:
            if not csv_path.exists():
                print(f"  WARNING: {csv_path} not found, skipping market_id injection")
                continue
            df = pd.read_csv(csv_path)
            if "market_id" not in df.columns:
                df.insert(0, "market_id", 1)
                df.to_csv(csv_path, index=False)

        print(f"  [{tag}] Data generation complete")

    if args.skip_mle:
        return None

    # ------------------------------------------------------------------
    # Stage 4: run distributed MLE (mpirun -np 1)
    # ------------------------------------------------------------------
    freeze = not args.no_freeze_globals
    est_dir.mkdir(parents=True, exist_ok=True)
    mle_cmd = [
        "mpirun", "-np", "1",
        py, "-m", "screening.analysis.mle.run_distributed",
        "--firms_path", str(clean_dir / "equilibrium_firms.csv"),
        "--workers_path", str(build_dir / "workers_dataset.csv"),
        "--params_path", str(raw_dir / "parameters_effective.csv"),
        "--out_dir", str(est_dir),
        "--M", "1",
        "--inner_maxiter", str(args.inner_maxiter),
        "--inner_tol", str(args.inner_tol),
        "--outer_maxiter", str(args.outer_maxiter),
        "--outer_tol", str(args.outer_tol),
    ]
    if args.profile_delta:
        mle_cmd.append("--profile_delta")
    elif args.delta_only:
        mle_cmd.append("--freeze_tilde_q")
    elif freeze:
        mle_cmd.append("--freeze_globals")
    if not run_cmd(mle_cmd, f"{tag}/mle"):
        return None

    # Parse results
    json_path = est_dir / "mle_distributed_estimates.json"
    if not json_path.exists():
        print(f"  [{tag}] No results JSON found at {json_path}")
        return None

    with open(json_path) as f:
        result = json.load(f)

    # Attach equilibrium metadata
    result["equil_meta"] = read_equil_meta(cfg_dir)
    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_table(all_results: list[dict]) -> str:
    """Format results as a markdown table."""
    header = (
        "| J | N | N/J | eq_cvg | eq_resid | "
        "tau_hat | tau_err | tg_hat | tg_err | "
        "d_corr | d_RMSE | "
        "tq_corr | tq_RMSE | "
        "inner_cvg | outer_cvg | wall_s |"
    )
    sep = "|" + "|".join(["---"] * 16) + "|"
    lines = [header, sep]

    for r in all_results:
        J, N = r["J"], r["N"]
        if r["result"] is None:
            lines.append(
                f"| {J} | {N} | {N // J} | "
                + "FAILED |" + " |" * 12
            )
            continue

        res = r["result"]
        rec = res.get("recovery", {})

        # Equilibrium convergence
        # Note: screening cutoffs create a structural residual floor (~1e-3)
        # from quadrature discretization, so strict tol (1e-10) is never met.
        # residual < 0.01 is effectively converged for this model.
        eq = res.get("equil_meta", {})
        eq_resid = eq.get("residual", None)
        if eq_resid is not None:
            eq_cvg = "Y" if eq_resid < 0.01 else "N"
            eq_resid_s = f"{eq_resid:.2e}"
        else:
            eq_cvg = "?"
            eq_resid_s = "-"

        # Inner convergence from last history entry
        history = res.get("history", [])
        if history:
            last = history[-1]
            n_inner = last.get("inner_converged", "?")
            inner_cvg = str(n_inner)
        else:
            inner_cvg = "-"

        outer_cvg = "Y" if res.get("converged", False) else "N"
        wall_s = res.get("timings", {}).get("total_time_sec", 0)

        lines.append(
            f"| {J} | {N} | {N // J} | "
            f"{eq_cvg} | {eq_resid_s} | "
            f"{rec.get('tau_hat', 0):.4f} | {rec.get('tau_err', 0):+.4f} | "
            f"{rec.get('tg_hat', 0):.4f} | {rec.get('tg_err', 0):+.4f} | "
            f"{rec.get('delta_corr', 0):.4f} | {rec.get('delta_rmse', 0):.4f} | "
            f"{rec.get('tq_corr', 0):.4f} | {rec.get('tq_rmse', 0):.4f} | "
            f"{inner_cvg} | {outer_cvg} | {wall_s:.0f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    freeze = not args.no_freeze_globals

    # Determine configs
    if args.configs:
        configs = json.loads(args.configs)
    else:
        configs = list(DEFAULT_CONFIGS)
        if args.vary_J:
            configs.extend(VARY_J_CONFIGS)

    print("N/J Scaling Test")
    print(f"  Base dir: {base_dir}")
    print(f"  Seed: {args.seed}")
    print(f"  Configs ({len(configs)}): {configs}")
    if args.profile_delta:
        print("  Mode: PROFILE-DELTA (tau, tilde_gamma fixed; delta profiled via contraction)")
    elif args.delta_only:
        print("  Mode: DELTA-ONLY (tau, tilde_gamma, tilde_q fixed at true values)")
    elif freeze:
        print("  Mode: INNER-ONLY (tau, tilde_gamma fixed at true values)")
    else:
        print("  Mode: FULL (outer optimization over tau, tilde_gamma)")
    print(f"  MLE: inner_maxiter={args.inner_maxiter}, inner_tol={args.inner_tol}")
    if not freeze:
        print(f"       outer_maxiter={args.outer_maxiter}, outer_tol={args.outer_tol}")
    if args.skip_data:
        print("  ** Skipping data generation **")
    if args.skip_mle:
        print("  ** Skipping MLE **")
    print()

    all_results: list[dict] = []
    total_start = time.perf_counter()

    for i, (J, N) in enumerate(configs):
        tag = f"J{J}_N{N}"
        cfg_dir = base_dir / tag
        cfg_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(configs)}] {tag} (N/J={N // J})")
        print(f"{'=' * 60}")

        t0 = time.perf_counter()
        result = run_pipeline(J, N, cfg_dir, args.seed, args)
        wall = time.perf_counter() - t0

        all_results.append({
            "J": J, "N": N, "tag": tag,
            "wall_total": wall,
            "result": result,
        })

        if result:
            rec = result.get("recovery", {})
            eq = result.get("equil_meta", {})
            eq_s = "Y" if eq.get("converged") else f"N(res={eq.get('residual', '?'):.1e})"
            print(f"  => eq={eq_s}, tau_err={rec.get('tau_err', '?'):+.4f}, "
                  f"tq_corr={rec.get('tq_corr', '?'):.4f}, "
                  f"d_corr={rec.get('delta_corr', '?'):.4f}, "
                  f"wall={wall:.0f}s")

    total_wall = time.perf_counter() - total_start

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}\n")
    print(format_table(all_results))
    print(f"\nTotal wall time: {total_wall:.0f}s ({total_wall / 60:.1f} min)")

    # Write JSON summary
    summary_path = base_dir / "scaling_nj_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Full results written to: {summary_path}")


if __name__ == "__main__":
    main()
