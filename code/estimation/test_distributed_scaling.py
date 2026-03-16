#!/usr/bin/env python3
"""Distributed MLE scaling test: vary M (markets) and P (MPI ranks).

Generates a large dataset (M=1000 by default), then runs the distributed MLE
estimator across a grid of (M, P) combinations, collecting timing, memory,
and parameter-recovery metrics.

Usage::

    # Dry run — print grid and estimates only
    python -m code.estimation.test_distributed_scaling \
        --data_dir /proj/screening/rcly/data_scaling --dry_run

    # Small test
    python -m code.estimation.test_distributed_scaling \
        --data_dir /proj/screening/rcly/data_scaling \
        --M_list 10,50 --P_list 10,20

    # Full grid (skip data generation if already done)
    python -m code.estimation.test_distributed_scaling \
        --data_dir /proj/screening/rcly/data_scaling \
        --skip_datagen
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Memory monitoring (multi-process, for mpirun)
# ---------------------------------------------------------------------------


class MPIMemoryMonitor:
    """Polls aggregate RSS of a process tree via /proc.

    Tracks the parent process (mpirun) and all descendant Python processes.
    Reports aggregate peak RSS across all ranks.
    """

    def __init__(self, pid: int, interval: float = 0.5):
        self.root_pid = pid
        self.interval = interval
        self.peak_total_rss_kb: int = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    @property
    def peak_total_rss_gb(self) -> float:
        return self.peak_total_rss_kb / (1024.0 * 1024.0)

    def _get_descendant_pids(self, parent_pid: int) -> List[int]:
        """Get all descendant PIDs by walking /proc/*/stat."""
        children = []
        try:
            proc_dirs = os.listdir("/proc")
        except OSError:
            return children
        # Build parent->children map
        parent_map: Dict[int, List[int]] = {}
        for entry in proc_dirs:
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/stat") as f:
                    stat = f.read()
                # Field 4 (0-indexed 3) is ppid
                parts = stat.rsplit(")", 1)[-1].split()
                ppid = int(parts[1])  # index 1 after the closing paren
                parent_map.setdefault(ppid, []).append(pid)
            except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
                continue
        # BFS from parent_pid
        queue = [parent_pid]
        while queue:
            p = queue.pop(0)
            for child in parent_map.get(p, []):
                children.append(child)
                queue.append(child)
        return children

    def _read_rss_kb(self, pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass
        return 0

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                all_pids = [self.root_pid] + self._get_descendant_pids(self.root_pid)
                total_rss = sum(self._read_rss_kb(p) for p in all_pids)
                if total_rss > self.peak_total_rss_kb:
                    self.peak_total_rss_kb = total_rss
            except Exception:
                pass
            self._stop.wait(self.interval)


# ---------------------------------------------------------------------------
# Time / memory estimation model
# ---------------------------------------------------------------------------

# Empirical constants (from prior runs)
T_PER_MARKET = 9.0       # seconds per market per outer iteration
N_OUTER_ITERS = 8        # typical convergence count
JIT_OVERHEAD = 60.0      # XLA compilation on first call (seconds)
MEM_PER_RANK_GB = 2.0    # conservative peak RSS per rank


def estimate_time_sec(M: int, P: int) -> float:
    """Estimate wall-clock time for one (M, P) cell."""
    markets_per_rank = ceil(M / P)
    return JIT_OVERHEAD + N_OUTER_ITERS * markets_per_rank * T_PER_MARKET


def estimate_mem_gb(P: int) -> float:
    """Estimate peak aggregate memory for P ranks."""
    return P * MEM_PER_RANK_GB


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def build_grid(M_list: List[int], P_list: List[int],
               max_time_min: float) -> List[Dict]:
    """Build (M, P) grid, filtering infeasible cells.

    Rules:
    - Skip P > M (idle ranks)
    - P=1 only for M <= 50
    - Skip cells exceeding max_time_min
    """
    cells = []
    for M in M_list:
        for P in P_list:
            # Skip if more ranks than markets
            if P > M:
                continue
            # P=1 only for small M
            if P == 1 and M > 50:
                continue

            est_sec = estimate_time_sec(M, P)
            est_min = est_sec / 60.0
            est_mem = estimate_mem_gb(P)

            if est_min > max_time_min:
                continue

            cells.append({
                "M": M,
                "P": P,
                "markets_per_rank": ceil(M / P),
                "est_time_min": round(est_min, 1),
                "est_mem_gb": round(est_mem, 1),
            })

    # Sort by estimated time (fastest first)
    cells.sort(key=lambda c: c["est_time_min"])
    return cells


def print_grid(cells: List[Dict]) -> None:
    """Print the grid as a formatted table."""
    header = f"{'M':>6s} {'P':>4s} {'M/P':>5s} {'est_min':>8s} {'est_GB':>7s}"
    print(header)
    print("-" * len(header))
    for c in cells:
        print(f"{c['M']:>6d} {c['P']:>4d} {c['markets_per_rank']:>5d} "
              f"{c['est_time_min']:>8.1f} {c['est_mem_gb']:>7.1f}")
    print(f"\nTotal cells: {len(cells)}")
    total_est = sum(c["est_time_min"] for c in cells)
    print(f"Total estimated time (sequential): {total_est:.0f} min ({total_est/60:.1f} hr)")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_scaling_data(data_dir: str, M: int = 1000,
                          parallel_markets: int = 80) -> Tuple[bool, str]:
    """Generate M-market dataset using the data pipeline."""
    env = os.environ.copy()
    env["SCREENING_DATA_DIR"] = data_dir
    env["PYTHONPATH"] = str(REPO_ROOT)

    steps = [
        (
            "01_prep_data",
            "code.data_environment.01_prep_data",
            [
                "--M", str(M), "--J", "100", "--N_workers", "2000",
                "--tau", "0.4", "--eta", "10", "--alpha", "0.2",
                "--mu_x_skill", "12", "--sigma_x_skill", "5",
                "--mu_e", "0", "--sigma_e", "0",
                "--quad_n_x", "25", "--quad_n_y", "25",
                "--conduct_mode", "1",
                "--rho_x_skill_ell_x", "0.3", "--rho_x_skill_ell_y", "0.3",
                "--worker_loc_mode", "cartesian", "--seed", "12345",
            ],
        ),
        (
            "02_solve_equilibrium",
            "code.data_environment.02_solve_equilibrium",
            ["--M", str(M), "--conduct_mode", "2", "--parallel_markets", str(parallel_markets)],
        ),
        (
            "03_draw_workers",
            "code.data_environment.03_draw_workers",
            ["--M", str(M)],
        ),
    ]

    for label, module, args in steps:
        print(f"  [{label}] Running...")
        t0 = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", module] + args,
            env=env, cwd=str(REPO_ROOT),
            capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"  [{label}] FAILED in {elapsed:.1f}s")
            return False, f"{label} failed: {result.stderr[-500:]}"
        print(f"  [{label}] OK in {elapsed:.1f}s")

    return True, ""


# ---------------------------------------------------------------------------
# Run a single (M, P) cell
# ---------------------------------------------------------------------------


def run_cell(M: int, P: int, data_dir: str, out_base: str,
             inner_maxiter: int, timeout_sec: float,
             ) -> Dict:
    """Run distributed MLE for one (M, P) cell.

    Returns a dict with all metrics for this cell.
    """
    cell_out = os.path.join(out_base, f"M{M}_P{P}")
    os.makedirs(cell_out, exist_ok=True)

    firms_path = os.path.join(data_dir, "clean", "equilibrium_firms.csv")
    workers_path = os.path.join(data_dir, "build", "workers_dataset.csv")
    params_path = os.path.join(data_dir, "raw", "parameters_effective.csv")

    cmd = [
        "mpirun", "-np", str(P), "--oversubscribe",
        sys.executable, "-m", "code.estimation.run_distributed_mle",
        "--M", str(M),
        "--outer_method", "L-BFGS-B",
        "--outer_maxiter", "20",
        "--inner_maxiter", str(inner_maxiter),
        "--freeze", "gamma,sigma_e,lambda_e",
        "--skip_statistics", "--skip_plot",
        "--firms_path", firms_path,
        "--workers_path", workers_path,
        "--params_path", params_path,
        "--out_dir", cell_out,
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)

    row = {
        "M": M,
        "P": P,
        "markets_per_rank": ceil(M / P),
        "est_time_min": round(estimate_time_sec(M, P) / 60, 2),
        "actual_time_min": None,
        "est_mem_gb": round(estimate_mem_gb(P), 1),
        "actual_peak_mem_gb": None,
        "converged": None,
        "n_outer_iters": None,
        "final_nll": None,
        "final_grad_norm": None,
        "tau_hat": None,
        "tau_error": None,
        "delta_rmse_over_sd": None,
        "delta_corr": None,
        "qbar_rmse_over_sd": None,
        "qbar_corr": None,
        "build_time_s": None,
        "solve_time_s": None,
        "total_time_s": None,
        "inner_conv_final": None,
        "error": "",
    }

    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(REPO_ROOT),
        )
        monitor = MPIMemoryMonitor(proc.pid)
        monitor.start()

        try:
            stdout_bytes, _ = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            monitor.stop()
            row["error"] = f"TIMEOUT after {timeout_sec/60:.0f} min"
            row["actual_time_min"] = round((time.perf_counter() - t0) / 60, 2)
            row["actual_peak_mem_gb"] = round(monitor.peak_total_rss_gb, 2)
            return row

        monitor.stop()
        wall = time.perf_counter() - t0
        row["actual_time_min"] = round(wall / 60, 2)
        row["actual_peak_mem_gb"] = round(monitor.peak_total_rss_gb, 2)

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            # Grab last 500 chars of output for error context
            row["error"] = f"rc={proc.returncode}: {stdout_text[-500:]}"
            return row

        # Save stdout log
        log_path = os.path.join(cell_out, "stdout.log")
        with open(log_path, "w") as f:
            f.write(stdout_text)

        # Parse results JSON
        results_path = os.path.join(cell_out, "mle_distributed_estimates_jax.json")
        if not os.path.exists(results_path):
            row["error"] = "No results JSON produced"
            return row

        with open(results_path) as f:
            res = json.load(f)

        row["converged"] = res.get("converged", False)
        row["n_outer_iters"] = res.get("nit", 0)
        row["final_nll"] = res.get("objective")
        row["final_grad_norm"] = res.get("grad_norm")

        dm = res.get("distance_metrics", {})
        row["tau_hat"] = dm.get("tau_hat")
        row["tau_error"] = dm.get("tau_error")
        row["delta_rmse_over_sd"] = dm.get("delta_rmse_over_sd")
        row["delta_corr"] = dm.get("delta_corr")
        row["qbar_rmse_over_sd"] = dm.get("qbar_rmse_over_sd")
        row["qbar_corr"] = dm.get("qbar_corr")

        timings = res.get("timings", {})
        row["build_time_s"] = timings.get("build_time_sec")
        row["solve_time_s"] = timings.get("solve_time_sec")
        row["total_time_s"] = timings.get("total_time_sec")

        # Inner convergence from last history entry
        history = res.get("history", [])
        if history:
            row["inner_conv_final"] = history[-1].get("inner_converged")

    except Exception as e:
        row["error"] = str(e)
        row["actual_time_min"] = round((time.perf_counter() - t0) / 60, 2)

    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed MLE scaling test: vary M and P",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--M_list", type=str, default="10,50,100,200,500,1000",
                        help="Comma-separated list of market counts")
    parser.add_argument("--P_list", type=str, default="1,10,20,30,40",
                        help="Comma-separated list of MPI rank counts")
    parser.add_argument("--max_time_min", type=float, default=120,
                        help="Skip cells with estimated time above this (minutes)")
    parser.add_argument("--inner_maxiter", type=int, default=400,
                        help="Inner solver max iterations per market")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory for scaling dataset (SCREENING_DATA_DIR)")
    parser.add_argument("--out_dir", type=str, default="output/scaling_test",
                        help="Base output directory for results")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print grid and estimates without running anything")
    parser.add_argument("--skip_datagen", action="store_true",
                        help="Skip data generation (assume data already exists)")
    parser.add_argument("--datagen_M", type=int, default=1000,
                        help="Number of markets to generate in data pipeline")
    parser.add_argument("--parallel_markets", type=int, default=80,
                        help="Parallel market solves during data generation")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    M_list = [int(x) for x in args.M_list.split(",")]
    P_list = [int(x) for x in args.P_list.split(",")]

    # Build grid
    cells = build_grid(M_list, P_list, args.max_time_min)

    print(f"\n{'='*60}")
    print("Distributed MLE Scaling Test")
    print(f"{'='*60}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.out_dir}")
    print(f"M values: {M_list}")
    print(f"P values: {P_list}")
    print(f"Max time per cell: {args.max_time_min} min")
    print(f"Inner maxiter: {args.inner_maxiter}")
    print()

    print_grid(cells)

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running any cells.")
        return

    # Data generation
    max_M = max(M_list)
    datagen_M = max(args.datagen_M, max_M)

    if not args.skip_datagen:
        print(f"\n{'='*60}")
        print(f"Generating {datagen_M}-market dataset...")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        ok, err = generate_scaling_data(args.data_dir, M=datagen_M,
                                        parallel_markets=args.parallel_markets)
        elapsed = time.perf_counter() - t0
        if not ok:
            print(f"DATA GENERATION FAILED ({elapsed:.0f}s): {err}")
            sys.exit(1)
        print(f"Data generation complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    else:
        # Verify data exists
        firms_path = os.path.join(args.data_dir, "clean", "equilibrium_firms.csv")
        if not os.path.exists(firms_path):
            print(f"ERROR: --skip_datagen but {firms_path} not found")
            sys.exit(1)
        print(f"\nSkipping data generation (--skip_datagen)")

    # Run grid
    print(f"\n{'='*60}")
    print(f"Running {len(cells)} cells...")
    print(f"{'='*60}")

    out_base = os.path.join(args.out_dir)
    os.makedirs(out_base, exist_ok=True)

    results = []
    for i, cell in enumerate(cells):
        M, P = cell["M"], cell["P"]
        timeout_sec = cell["est_time_min"] * 60 * 3  # 3x estimated time

        print(f"\n--- Cell {i+1}/{len(cells)}: M={M}, P={P} "
              f"(est {cell['est_time_min']} min) ---")

        row = run_cell(
            M=M, P=P,
            data_dir=args.data_dir,
            out_base=out_base,
            inner_maxiter=args.inner_maxiter,
            timeout_sec=timeout_sec,
        )
        results.append(row)

        # Print summary
        status = "OK" if not row["error"] else f"FAIL: {row['error'][:80]}"
        actual = row.get("actual_time_min")
        mem = row.get("actual_peak_mem_gb")
        print(f"  {status}  time={actual} min  mem={mem} GB")

        # Flush CSV after each cell (so partial results are saved)
        csv_path = os.path.join(out_base, "scaling_grid_results.csv")
        _write_csv(results, csv_path)

    # Final summary
    print(f"\n{'='*60}")
    print("Scaling Test Complete")
    print(f"{'='*60}")
    print(f"Results: {csv_path}")
    print(f"Total cells: {len(results)}")
    ok_count = sum(1 for r in results if not r["error"])
    print(f"Succeeded: {ok_count}/{len(results)}")

    # Print summary table
    print(f"\n{'M':>6s} {'P':>4s} {'est_min':>8s} {'act_min':>8s} "
          f"{'est_GB':>7s} {'act_GB':>7s} {'conv':>5s} {'tau_err':>8s} {'error':>10s}")
    print("-" * 75)
    for r in results:
        conv = str(r.get("converged", "")) if r.get("converged") is not None else ""
        tau_err = f"{r['tau_error']:.4f}" if r.get("tau_error") is not None else ""
        err_short = r["error"][:10] if r["error"] else ""
        act_min = f"{r['actual_time_min']:.1f}" if r.get("actual_time_min") is not None else ""
        act_gb = f"{r['actual_peak_mem_gb']:.1f}" if r.get("actual_peak_mem_gb") is not None else ""
        print(f"{r['M']:>6d} {r['P']:>4d} {r['est_time_min']:>8.1f} {act_min:>8s} "
              f"{r['est_mem_gb']:>7.1f} {act_gb:>7s} {conv:>5s} {tau_err:>8s} {err_short:>10s}")


def _write_csv(results: List[Dict], path: str) -> None:
    """Write results list to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
