#!/usr/bin/env python3
"""Generalized profile likelihood for tg or tau.

For each fixed value of the profiled parameter, optimizes the free global
parameter + inner (delta, tq) and records the minimized NLL.  This reveals
the true objective landscape along one dimension independent of initialization
artifacts.

Replaces the old ``run_tg_profile.py`` with support for both parameters.

Usage::

    # tg profile (backward-compatible with old run_tg_profile.py)
    python -m screening.analysis.scaling.run_profile --param tg

    # tau profile (initializes tg at the tg profile minimum)
    python -m screening.analysis.scaling.run_profile --param tau

    # Smoke tests
    python -m screening.analysis.scaling.run_profile --param tg --smoke
    python -m screening.analysis.scaling.run_profile --param tau --smoke

    # Skip MLE, just regenerate plots/tables from cached results
    python -m screening.analysis.scaling.run_profile --param tg --skip_mle
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parents[3]      # code/ directory
PROJ_ROOT = CODE_DIR.parent                          # project root
DEFAULTS_YAML = PROJ_ROOT / "config" / "defaults.yaml"
OUTPUT_DIR = PROJ_ROOT / "output"

# Per-parameter configuration
PARAM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tg": {
        "grid": [1, 2, 3, 4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 12, 15, 20],
        "smoke_grid": [3.0, 5.0, 7.0, 10.0, 15.0],
        "freeze_flag": "--freeze_tg",
        "init_flag": "--tg_init",
        "true_key": "tilde_gamma",
        "hat_idx": 1,               # theta_G[1]
        "free_hat_idx": 0,          # theta_G[0] = tau (the free param)
        "label": "tilde_gamma",
        "short_label": "tg",
        "free_label": "tau",
        "output_subdir": "tg_profile",
        "subdir_fmt": "profile_tg_{val:.2f}",
        "fixed_field": "tg_fixed",
        "sensitivity_subdir": "tg_sensitivity",
        "sensitivity_summary": "tg_sensitivity_summary.json",
        "sensitivity_init_key": "tg_init",
        "sensitivity_hat_key": "tg_hat",
    },
    "tau": {
        "grid": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.38, 0.4,
                 0.42, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
        "smoke_grid": [0.1, 0.3, 0.4, 0.5, 0.7],
        "freeze_flag": "--freeze_tau",
        "init_flag": "--tau_init",
        "true_key": "tau",
        "hat_idx": 0,               # theta_G[0]
        "free_hat_idx": 1,          # theta_G[1] = tg (the free param)
        "label": "tau",
        "short_label": "tau",
        "free_label": "tilde_gamma",
        "output_subdir": "tau_profile",
        "subdir_fmt": "profile_tau_{val:.2f}",
        "fixed_field": "tau_fixed",
        "sensitivity_subdir": "tau_sensitivity",
        "sensitivity_summary": "tau_sensitivity_summary.json",
        "sensitivity_init_key": "tau_init",
        "sensitivity_hat_key": "tau_hat",
    },
}

# Default data: reuse tg_sensitivity data (M=100, N=2500, J=100)
DEFAULT_DATA = OUTPUT_DIR / "scaling" / "tg_sensitivity" / "data"


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
        description="Generalized profile likelihood (tg or tau)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--param", type=str, default="tg",
                   choices=list(PARAM_CONFIGS.keys()),
                   help="Parameter to profile over")
    p.add_argument("--base_dir", type=str, default=None,
                   help="Output root (default: output/scaling/{param}_profile)")
    p.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA),
                   help="Data directory (reuse existing if present)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--M", type=int, default=100)
    p.add_argument("--N", type=int, default=2500,
                   help="Workers per market")
    p.add_argument("--J", type=int, default=100,
                   help="Firms per market")
    p.add_argument("--mpi_np", type=int, default=16,
                   help="MPI ranks")
    p.add_argument("--inner_maxiter", type=int, default=500)
    p.add_argument("--inner_tol", type=float, default=1e-7)
    p.add_argument("--outer_maxiter", type=int, default=100)
    p.add_argument("--outer_tol", type=float, default=1e-6)
    p.add_argument("--grid", type=str, default=None,
                   help="JSON override for grid, e.g. '[3, 5, 7, 10]'")
    p.add_argument("--skip_data", action="store_true",
                   help="Skip data generation (assume data exists)")
    p.add_argument("--skip_mle", action="store_true",
                   help="Skip MLE runs (only collect results)")
    p.add_argument("--smoke", action="store_true",
                   help="Quick validation with small M and coarse grid")
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


def generate_data(data_dir: Path, M: int, N: int, J: int, seed: int,
                  params: dict) -> bool:
    """Run 01_prep_data -> 02_solve_equilibrium -> 03_draw_workers."""
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
# Phase 2: Profile grid
# ---------------------------------------------------------------------------


def run_profile_one(cfg: dict, base_dir: Path, data_dir: Path, val: float,
                    M: int, mpi_np: int,
                    inner_maxiter: int, inner_tol: float,
                    outer_maxiter: int, outer_tol: float,
                    other_init: Optional[float] = None) -> bool:
    """Run distributed MLE with one global param frozen at val."""
    raw = data_dir / "raw"
    clean = data_dir / "clean"
    build = data_dir / "build"
    est_dir = base_dir / cfg["subdir_fmt"].format(val=val)
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
        cfg["freeze_flag"], str(val),
        cfg["init_flag"], str(val),
        "--inner_maxiter", str(inner_maxiter),
        "--inner_tol", str(inner_tol),
        "--outer_maxiter", str(outer_maxiter),
        "--outer_tol", str(outer_tol),
    ]

    # Initialize the free global parameter if requested
    if other_init is not None:
        # Determine the init flag for the OTHER (free) parameter
        other_param = "tau" if cfg["short_label"] == "tg" else "tg"
        other_cfg = PARAM_CONFIGS[other_param]
        cmd += [other_cfg["init_flag"], str(other_init)]

    short = cfg["short_label"]
    return _run(cmd, label=f"profile {short}={val:.2f}", cwd=CODE_DIR,
                env_extra=data_env, timeout=max(3600, M * 300))


def parse_profile_result(cfg: dict, base_dir: Path, val: float) -> Optional[dict]:
    """Parse MLE result JSON for one profile run."""
    subdir = cfg["subdir_fmt"].format(val=val)
    json_path = base_dir / subdir / "mle_distributed_estimates.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase 3: Collect, plot, and report
# ---------------------------------------------------------------------------


def collect_row(cfg: dict, val_fixed: float, result: Optional[dict],
                true_params: dict) -> dict:
    """Build one row of the summary table."""
    row: Dict[str, Any] = {cfg["fixed_field"]: val_fixed}

    if result is None:
        row["status"] = "FAILED"
        return row

    row["status"] = "OK"
    rec = result.get("recovery", {})
    timings = result.get("timings", {})

    row["free_hat"] = result["theta_G"][cfg["free_hat_idx"]]
    row["free_err"] = row["free_hat"] - true_params[cfg["free_label"]]
    row["delta_corr"] = rec.get("delta_corr", 0)
    row["tq_corr"] = rec.get("tq_corr", 0)
    row["nll"] = result.get("objective", None)
    row["n_outer_iters"] = result.get("n_outer_iters", 0)
    row["converged"] = result.get("converged", False)
    row["wall_s"] = timings.get("total_time_sec", 0)

    return row


def plot_profile(cfg: dict, rows: List[dict], true_params: dict,
                 out_path: Path, naive_val: float | None = None) -> None:
    """Plot profile likelihood curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping plot")
        return

    ok_rows = [r for r in rows if r["status"] == "OK" and r["nll"] is not None]
    if len(ok_rows) < 2:
        print("  Too few successful runs for plot")
        return

    fixed_field = cfg["fixed_field"]
    param_label = cfg["label"]
    true_val = true_params[cfg["true_key"]]

    x_vals = [r[fixed_field] for r in ok_rows]
    nll_vals = [r["nll"] for r in ok_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, nll_vals, "o-", color="steelblue", linewidth=2,
            markersize=6)

    ax.axvline(true_val, color="green", linestyle="--", linewidth=1.5,
               label=f"True {param_label} = {true_val:.3f}")

    if naive_val is not None:
        ax.axvline(naive_val, color="orange", linestyle=":", linewidth=1.5,
                   label=f"Naive {param_label} = {naive_val:.2f}")

    best_idx = int(np.argmin(nll_vals))
    ax.plot(x_vals[best_idx], nll_vals[best_idx], "*", color="red",
            markersize=15, zorder=5,
            label=f"Min NLL at {param_label}={x_vals[best_idx]:.2f}")

    ax.set_xlabel(f"{param_label} (fixed)", fontsize=12)
    ax.set_ylabel("Profile NLL", fontsize=12)
    ax.set_title(f"Profile Likelihood: NLL vs Fixed {param_label}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


def plot_combined(cfg: dict, rows: List[dict], true_params: dict,
                  out_path: Path, naive_val: float | None = None) -> None:
    """Overlay sensitivity init->final arrows on the profile curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available -- skipping combined plot")
        return

    # Load sensitivity data
    sens_dir = OUTPUT_DIR / "scaling" / cfg["sensitivity_subdir"]
    sens_path = sens_dir / cfg["sensitivity_summary"]
    if not sens_path.exists():
        print(f"  No sensitivity data at {sens_path} -- skipping combined plot")
        return

    with open(sens_path) as f:
        sens_data = json.load(f)
    sens_rows = [r for r in sens_data.get("rows", [])
                 if r.get("status") == "OK" and r.get("nll") is not None]
    if not sens_rows:
        print("  No valid sensitivity rows -- skipping combined plot")
        return

    # Profile curve data
    ok_rows = [r for r in rows if r["status"] == "OK" and r["nll"] is not None]
    if len(ok_rows) < 2:
        print("  Too few profile points for combined plot")
        return

    fixed_field = cfg["fixed_field"]
    param_label = cfg["label"]
    true_val = true_params[cfg["true_key"]]
    init_key = cfg["sensitivity_init_key"]
    hat_key = cfg["sensitivity_hat_key"]

    x_profile = np.array([r[fixed_field] for r in ok_rows])
    nll_profile = np.array([r["nll"] for r in ok_rows])
    best_nll = float(np.min(nll_profile))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Profile curve
    ax.plot(x_profile, nll_profile, "o-", color="steelblue", linewidth=2,
            markersize=6, label="Profile NLL", zorder=3)

    # True value
    ax.axvline(true_val, color="green", linestyle="--", linewidth=1.5,
               label=f"True {param_label} = {true_val:.3f}", zorder=1)

    # Naive value
    if naive_val is not None:
        ax.axvline(naive_val, color="orange", linestyle=":", linewidth=1.5,
                   label=f"Naive {param_label} = {naive_val:.2f}", zorder=1)

    # Profile minimum
    best_idx = int(np.argmin(nll_profile))
    ax.plot(x_profile[best_idx], nll_profile[best_idx], "*", color="red",
            markersize=15, zorder=5,
            label=f"Profile min at {param_label}={x_profile[best_idx]:.2f}")

    # Sensitivity arrows
    n_arrows = 0
    arrow_color = "darkorange"
    for sr in sens_rows:
        p_init = sr.get(init_key)
        p_hat = sr.get(hat_key)
        s_nll = sr.get("nll")
        if p_init is None or p_hat is None or s_nll is None:
            continue

        # Interpolate start y from profile curve
        y_start = float(np.interp(p_init, x_profile, nll_profile))
        y_end = s_nll
        n_arrows += 1

        # Draw arrow from (init, interp_nll) to (hat, sensitivity_nll)
        ax.annotate(
            "", xy=(p_hat, y_end), xytext=(p_init, y_start),
            arrowprops=dict(
                arrowstyle="->", color=arrow_color, lw=1.5, alpha=0.7,
                connectionstyle="arc3,rad=0.1",
            ),
            zorder=4,
        )
        # Open circle at start
        ax.plot(p_init, y_start, "o", color=arrow_color, markersize=5,
                markerfacecolor="white", markeredgewidth=1.5, alpha=0.7,
                zorder=4)

    # Legend entry for arrows
    if n_arrows > 0:
        ax.plot([], [], "o-", color=arrow_color, markerfacecolor="white",
                label=f"Sensitivity runs ({n_arrows})")

    ax.set_xlabel(f"{param_label}", fontsize=12)
    ax.set_ylabel("NLL", fontsize=12)
    ax.set_title(f"Profile Likelihood + Sensitivity: {param_label}", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Combined plot saved to {out_path}")


def print_table(cfg: dict, rows: List[dict], true_params: dict) -> str:
    """Print and return markdown summary table."""
    fixed_field = cfg["fixed_field"]
    param_label = cfg["label"]
    free_label = cfg["free_label"]
    true_val = true_params[cfg["true_key"]]

    lines: List[str] = []
    lines.append(f"### {param_label.replace('_', ' ').title()} Profile Likelihood\n")
    hdr = (
        f"| {fixed_field} | {free_label}_hat | {free_label}_err | d_corr | tq_corr "
        f"| NLL | outer_it | cvg | wall_s |"
    )
    sep = "|" + "|".join(["---"] * 9) + "|"
    lines.append(hdr)
    lines.append(sep)

    for r in rows:
        if r["status"] == "FAILED":
            lines.append(f"| {r[fixed_field]:.2f} | FAILED |" + " |" * 7)
            continue
        lines.append(
            f"| {r[fixed_field]:.2f} "
            f"| {r['free_hat']:.4f} "
            f"| {r['free_err']:+.4f} "
            f"| {r['delta_corr']:.3f} "
            f"| {r['tq_corr']:.3f} "
            f"| {r['nll']:.1f} "
            f"| {r['n_outer_iters']:>3d} "
            f"| {'Y' if r['converged'] else 'N'} "
            f"| {r['wall_s']:.0f} |"
        )

    lines.append(f"\nTrue: {param_label}={true_val:.4f}, "
                 f"{free_label}={true_params[cfg['free_label']]:.4f}")

    ok_rows = [r for r in rows if r["status"] == "OK" and r["nll"] is not None]
    if ok_rows:
        best = min(ok_rows, key=lambda r: r["nll"])
        lines.append(f"\nProfile minimum: {fixed_field}={best[fixed_field]:.2f}, "
                     f"NLL={best['nll']:.1f}, "
                     f"{free_label}_hat={best['free_hat']:.4f}")

    md = "\n".join(lines)
    print("\n" + md)
    return md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = PARAM_CONFIGS[args.param]
    true_params = load_true_params()
    true_val = true_params[cfg["true_key"]]

    # Resolve base_dir
    if args.base_dir is None:
        base_dir = OUTPUT_DIR / "scaling" / cfg["output_subdir"]
    else:
        base_dir = Path(args.base_dir)

    # Apply smoke overrides
    if args.smoke:
        args.M = 10
        grid = list(cfg["smoke_grid"])
    elif args.grid is not None:
        grid = json.loads(args.grid)
    else:
        grid = list(cfg["grid"])

    data_dir = Path(args.data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Determine other_init: for tau profile, load best tg from tg profile
    other_init: Optional[float] = None
    if args.param == "tau":
        tg_summary_path = OUTPUT_DIR / "scaling" / "tg_profile" / "tg_profile_summary.json"
        if tg_summary_path.exists():
            with open(tg_summary_path) as f:
                tg_summary = json.load(f)
            tg_ok = [r for r in tg_summary.get("rows", [])
                     if r.get("status") == "OK" and r.get("nll") is not None]
            if tg_ok:
                best_tg_row = min(tg_ok, key=lambda r: r["nll"])
                other_init = best_tg_row["tg_fixed"]
                print(f"  Loaded best tg={other_init:.2f} from tg profile")
        if other_init is None:
            print("  WARNING: tg profile not found, using naive tg init for tau profile")

    short = cfg["short_label"]
    param_label = cfg["label"]

    print("=" * 60)
    print(f"{param_label.replace('_', ' ').title()} Profile Likelihood")
    print("=" * 60)
    print(f"  Param:     {args.param}")
    print(f"  Base dir:  {base_dir}")
    print(f"  Data dir:  {data_dir}")
    print(f"  Data:      M={args.M}, N={args.N}, J={args.J}")
    print(f"  MPI ranks: {args.mpi_np}")
    print(f"  Grid:      {grid}")
    print(f"  True {short}: {true_val:.4f}")
    if other_init is not None:
        print(f"  Free param init: {cfg['free_label']}={other_init:.2f}")
    if args.skip_data:
        print("  ** Skipping data generation **")
    if args.skip_mle:
        print("  ** Skipping MLE runs **")
    print()

    total_t0 = time.perf_counter()

    # ---- Phase 1: Data ----
    build_workers = data_dir / "build" / "workers_dataset.csv"
    if build_workers.exists():
        print(f"Data found at {data_dir}, reusing.")
    elif not args.skip_data:
        print("-" * 60)
        print("Phase 1: Data generation")
        print("-" * 60)
        t0 = time.perf_counter()
        ok = generate_data(data_dir, args.M, args.N, args.J, args.seed,
                           true_params)
        if not ok:
            print("Data generation FAILED. Aborting.")
            return
        print(f"Data generation complete ({time.perf_counter() - t0:.0f}s)\n")
    else:
        print(f"WARNING: data not found at {data_dir} and --skip_data set")

    # ---- Phase 2: Profile grid ----
    if not args.skip_mle:
        print("-" * 60)
        print("Phase 2: Profile grid")
        print("-" * 60)
        for i, val in enumerate(grid):
            val = float(val)
            existing = parse_profile_result(cfg, base_dir, val)
            if existing is not None and existing.get("M", 0) == args.M:
                print(f"\n[{i+1}/{len(grid)}] {short}={val:.2f}  "
                      f"(cached, M={args.M})")
                continue
            print(f"\n[{i+1}/{len(grid)}] {short}={val:.2f}")
            t0 = time.perf_counter()
            ok = run_profile_one(
                cfg, base_dir, data_dir, val,
                M=args.M, mpi_np=args.mpi_np,
                inner_maxiter=args.inner_maxiter, inner_tol=args.inner_tol,
                outer_maxiter=args.outer_maxiter, outer_tol=args.outer_tol,
                other_init=other_init,
            )
            elapsed = time.perf_counter() - t0
            if not ok:
                print(f"  FAILED for {short}={val:.2f} ({elapsed:.0f}s)")
            else:
                print(f"  Done ({elapsed:.0f}s)")

    # ---- Phase 3: Collect, plot, report ----
    print("\n" + "-" * 60)
    print("Phase 3: Results")
    print("-" * 60)

    fixed_field = cfg["fixed_field"]
    rows = []
    naive_val = None
    for val in grid:
        val = float(val)
        result = parse_profile_result(cfg, base_dir, val)
        rows.append(collect_row(cfg, val, result, true_params))
        # Extract naive value from first available result
        if naive_val is None and result is not None:
            init_vals = result.get("init_values", {})
            naive_val = init_vals.get(f"naive_{short}")

    table_md = print_table(cfg, rows, true_params)

    # Profile plot
    prefix = f"{short}_profile"
    plot_path = base_dir / f"{prefix}_nll.png"
    plot_profile(cfg, rows, true_params, plot_path, naive_val=naive_val)

    # Combined plot (profile + sensitivity arrows)
    combined_path = base_dir / f"{prefix}_combined.png"
    plot_combined(cfg, rows, true_params, combined_path, naive_val=naive_val)

    # Save markdown
    md_path = base_dir / f"{prefix}_results.md"
    with open(md_path, "w") as f:
        f.write(table_md + "\n")
    print(f"\nTable written to {md_path}")

    # Save JSON summary
    summary = {
        "param": args.param,
        "grid": grid,
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
    summary_path = base_dir / f"{prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    total_time = time.perf_counter() - total_t0
    print(f"Total wall time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
