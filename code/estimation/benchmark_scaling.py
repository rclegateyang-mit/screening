#!/usr/bin/env python3
"""Benchmark MLE estimation scaling in (N, J).

Runs the full data-generation -> MLE pipeline for each (N, J) combination
and collects timing, peak memory, and parameter-fit metrics.

Each (N, J) run uses isolated data/output directories via SCREENING_DATA_DIR
and SCREENING_OUTPUT_DIR environment variables, so runs can safely execute
in parallel.

Usage:
    # Single run
    python -m code.estimation.benchmark_scaling --N 5000 --J 10

    # Grid sweep
    python -m code.estimation.benchmark_scaling \
        --N_list 1000,5000,10000 --J_list 5,10,20

    # With MLE options
    python -m code.estimation.benchmark_scaling \
        --N_list 1000,5000 --J_list 5,10 --maxiter 200 --penalty_weight 0.0

    # Parallel grid sweep (4 workers)
    python -m code.estimation.benchmark_scaling \
        --N_list 1000,5000,10000 --J_list 5,10,20 --parallel 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------


class MemoryMonitor:
    """Polls peak RSS of a subprocess via /proc/{pid}/status."""

    def __init__(self, pid: int, interval: float = 0.2):
        self.pid = pid
        self.interval = interval
        self.peak_rss_kb: int = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_kb / 1024.0

    def _poll(self) -> None:
        status_path = f"/proc/{self.pid}/status"
        while not self._stop.is_set():
            try:
                with open(status_path) as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_kb = int(line.split()[1])
                            if rss_kb > self.peak_rss_kb:
                                self.peak_rss_kb = rss_kb
                            break
            except (FileNotFoundError, ProcessLookupError, ValueError):
                break
            self._stop.wait(self.interval)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    N: int
    J: int
    K: int  # parameter dimension = 5 + 2*J
    # Data generation
    datagen_time_sec: float = 0.0
    datagen_peak_rss_mb: float = 0.0
    # MLE build (JIT compilation + data loading)
    build_time_sec: float = 0.0
    # MLE solve
    solve_time_sec: float = 0.0
    total_time_sec: float = 0.0
    mle_peak_rss_mb: float = 0.0
    # Convergence
    objective: float = float("nan")
    nll: float = float("nan")
    penalty: float = float("nan")
    nit: int = 0
    grad_norm: float = float("nan")
    # Parameter fit
    tau_error: float = float("nan")
    alpha_error: float = float("nan")
    gamma_error: float = float("nan")
    sigma_e_error: float = float("nan")
    lambda_e_error: float = float("nan")
    delta_l2: float = float("nan")
    delta_max_abs: float = float("nan")
    qbar_l2: float = float("nan")
    qbar_max_abs: float = float("nan")
    # Status
    success: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------


def _run_module(module: str, args: List[str], env: Dict[str, str],
                label: str = "") -> Tuple[int, float, float, str, str]:
    """Run a Python module as subprocess, monitoring memory.

    Returns (returncode, wall_time_sec, peak_rss_mb, stdout, stderr).
    """
    cmd = [sys.executable, "-m", module] + args
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(REPO_ROOT),
    )
    monitor = MemoryMonitor(proc.pid)
    monitor.start()
    stdout_bytes, stderr_bytes = proc.communicate()
    monitor.stop()
    wall = time.perf_counter() - t0

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    if label:
        status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
        print(f"  [{label}] {status} in {wall:.1f}s, peak RSS={monitor.peak_rss_mb:.0f} MB")

    return proc.returncode, wall, monitor.peak_rss_mb, stdout, stderr


def generate_data(N: int, J: int, env: Dict[str, str], seed: int = 12345) -> Tuple[bool, float, float, str]:
    """Run the data generation pipeline for (N, J).

    Returns (success, wall_time_sec, peak_rss_mb, error_msg).
    """
    total_time = 0.0
    peak_rss = 0.0

    # Step 1: prep data
    rc, t, rss, _, stderr = _run_module(
        "code.data_environment.01_prep_data",
        [
            "--N_workers", str(N),
            "--J", str(J),
            "--quad_n_x", "25",
            "--quad_n_y", "25",
            "--conduct_mode", "1",
            "--tau", "0.4",
            "--eta", "10",
            "--seed", str(seed),
            "--alpha", "0.2",
            "--worker_loc_mode", "cartesian",
            "--rho_x_skill_ell_x", "0.3",
            "--rho_x_skill_ell_y", "0.3",
            "--mu_x_skill", "12",
            "--sigma_x_skill", "5",
            "--mu_e", "0",
            "--sigma_e", "0",
        ],
        env,
        label=f"N={N},J={J} prep_data",
    )
    total_time += t
    peak_rss = max(peak_rss, rss)
    if rc != 0:
        return False, total_time, peak_rss, f"01_prep_data failed: {stderr[-500:]}"

    # Step 2: solve equilibrium
    rc, t, rss, _, stderr = _run_module(
        "code.data_environment.02_solve_equilibrium",
        ["--conduct_mode", "2", "--profit_grid_n", "100", "--profit_grid_log_span", "0.2"],
        env,
        label=f"N={N},J={J} solve_eq",
    )
    total_time += t
    peak_rss = max(peak_rss, rss)
    if rc != 0:
        return False, total_time, peak_rss, f"02_solve_equilibrium failed: {stderr[-500:]}"

    # Step 3: draw workers
    rc, t, rss, _, stderr = _run_module(
        "code.data_environment.03_draw_workers",
        [],
        env,
        label=f"N={N},J={J} draw_workers",
    )
    total_time += t
    peak_rss = max(peak_rss, rss)
    if rc != 0:
        return False, total_time, peak_rss, f"03_draw_workers failed: {stderr[-500:]}"

    return True, total_time, peak_rss, ""


def run_mle(env: Dict[str, str], maxiter: int, extra_args: List[str]) -> Tuple[bool, float, float, Optional[dict], str]:
    """Run MLE estimation.

    Returns (success, wall_time_sec, peak_rss_mb, result_dict_or_None, error_msg).
    """
    est_dir = Path(env["SCREENING_OUTPUT_DIR"]) / "estimation"
    result_path = est_dir / "mle_tau_alpha_gamma_sigma_e_lambda_e_delta_qbar_penalty_estimates_jax.json"

    args = [
        "--maxiter", str(maxiter),
        "--skip_statistics",
        "--skip_plot",
    ] + extra_args

    rc, wall, rss, stdout, stderr = _run_module(
        "code.estimation.run_mle_penalty_phi_sigma_jax",
        args,
        env,
        label="MLE",
    )
    if rc != 0:
        return False, wall, rss, None, f"MLE failed: {stderr[-500:]}"

    try:
        with open(result_path) as f:
            result = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return False, wall, rss, None, f"Could not read MLE results: {e}"

    return True, wall, rss, result, ""


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------


def run_single_benchmark(
    N: int, J: int, maxiter: int, extra_mle_args: List[str],
    base_tmp_dir: Optional[str] = None, keep_tmp: bool = False,
) -> BenchmarkResult:
    """Run full pipeline for one (N, J) combination and return metrics."""

    result = BenchmarkResult(N=N, J=J, K=5 + 2 * J)
    print(f"\n{'='*60}")
    print(f"Benchmark: N={N}, J={J}, K={result.K}")
    print(f"{'='*60}")

    # Create isolated temp directories
    parent = base_tmp_dir or tempfile.gettempdir()
    tmp_root = tempfile.mkdtemp(prefix=f"bench_N{N}_J{J}_", dir=parent)
    data_dir = os.path.join(tmp_root, "data")
    output_dir = os.path.join(tmp_root, "output")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["SCREENING_DATA_DIR"] = data_dir
    env["SCREENING_OUTPUT_DIR"] = output_dir
    env["PYTHONPATH"] = str(REPO_ROOT)

    try:
        # Data generation
        ok, t_dg, rss_dg, err = generate_data(N, J, env)
        result.datagen_time_sec = t_dg
        result.datagen_peak_rss_mb = rss_dg
        if not ok:
            result.error = err
            print(f"  DATA GENERATION FAILED: {err}")
            return result

        # MLE estimation
        ok, t_mle, rss_mle, mle_out, err = run_mle(env, maxiter, extra_mle_args)
        result.mle_peak_rss_mb = rss_mle
        if not ok:
            result.error = err
            print(f"  MLE FAILED: {err}")
            return result

        # Extract metrics from MLE output JSON
        result.build_time_sec = mle_out["timings"]["build_time_sec"]
        result.solve_time_sec = mle_out["timings"]["solve_time_sec"]
        result.total_time_sec = mle_out["timings"]["total_time_sec"]
        result.objective = mle_out["objective"]
        result.nll = mle_out["objective_breakdown"]["neg_log_likelihood"]
        result.penalty = mle_out["objective_breakdown"]["penalty"]
        result.nit = mle_out["nit"]
        result.grad_norm = mle_out["grad_norm"]

        dm = mle_out["distance_metrics"]
        result.tau_error = dm["tau_error"]
        result.alpha_error = dm["alpha_error"]
        result.gamma_error = dm["gamma_error"]
        result.sigma_e_error = dm["sigma_e_error"]
        result.lambda_e_error = dm["lambda_e_error"]
        result.delta_l2 = dm["delta_l2"]
        result.delta_max_abs = dm["delta_max_abs"]
        result.qbar_l2 = dm["qbar_l2"]
        result.qbar_max_abs = dm["qbar_max_abs"]

        result.success = True
        print(f"  RESULT: obj={result.objective:.2f}, nll={result.nll:.2f}, "
              f"solve={result.solve_time_sec:.1f}s, RSS={rss_mle:.0f}MB")

    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            print(f"  Temp dir kept: {tmp_root}")

    return result


# A module-level wrapper so ProcessPoolExecutor can pickle it.
def _run_single_wrapper(args_tuple):
    N, J, maxiter, extra_mle_args, base_tmp_dir, keep_tmp = args_tuple
    return run_single_benchmark(N, J, maxiter, extra_mle_args, base_tmp_dir, keep_tmp)


# ---------------------------------------------------------------------------
# CLI and grid sweep
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MLE scaling in (N, J)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Grid specification
    parser.add_argument("--N", type=int, default=None, help="Single N value")
    parser.add_argument("--J", type=int, default=None, help="Single J value")
    parser.add_argument("--N_list", type=str, default=None,
                        help="Comma-separated list of N values for grid sweep")
    parser.add_argument("--J_list", type=str, default=None,
                        help="Comma-separated list of J values for grid sweep")
    # MLE options
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--freeze", type=str, default=None,
                        help="Passed to MLE --freeze")
    parser.add_argument("--penalty_weight", type=float, default=1.0,
                        help="Passed to MLE --penalty_weight")
    # Benchmark options
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers for grid sweep")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Output CSV path (default: output/estimation/scaling_benchmark.csv)")
    parser.add_argument("--keep_tmp", action="store_true",
                        help="Keep temporary data directories after each run")
    parser.add_argument("--tmp_dir", type=str, default=None,
                        help="Base directory for temporary run data")
    return parser.parse_args()


def build_grid(args: argparse.Namespace) -> List[Tuple[int, int]]:
    if args.N_list and args.J_list:
        Ns = [int(x) for x in args.N_list.split(",")]
        Js = [int(x) for x in args.J_list.split(",")]
        return [(n, j) for n in Ns for j in Js]
    if args.N is not None and args.J is not None:
        return [(args.N, args.J)]
    raise ValueError("Specify either --N/--J for a single run or --N_list/--J_list for a grid sweep.")


def main() -> None:
    args = parse_args()
    grid = build_grid(args)

    extra_mle_args: List[str] = []
    if args.freeze:
        extra_mle_args += ["--freeze", args.freeze]
    if args.penalty_weight != 1.0:
        extra_mle_args += ["--penalty_weight", str(args.penalty_weight)]

    print(f"Benchmark grid: {len(grid)} configurations")
    print(f"  (N, J) pairs: {grid}")
    print(f"  maxiter={args.maxiter}, parallel={args.parallel}")
    if extra_mle_args:
        print(f"  MLE args: {extra_mle_args}")

    t0 = time.perf_counter()

    if args.parallel <= 1:
        results = [
            run_single_benchmark(N, J, args.maxiter, extra_mle_args, args.tmp_dir, args.keep_tmp)
            for N, J in grid
        ]
    else:
        tasks = [
            (N, J, args.maxiter, extra_mle_args, args.tmp_dir, args.keep_tmp)
            for N, J in grid
        ]
        results = []
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_run_single_wrapper, t): t for t in tasks}
            for future in as_completed(futures):
                results.append(future.result())

    total_wall = time.perf_counter() - t0

    # Sort results by (N, J) for readability
    results.sort(key=lambda r: (r.N, r.J))

    # Write CSV
    if args.out_csv:
        out_path = Path(args.out_csv)
    else:
        from code import get_output_subdir, OUTPUT_ESTIMATION
        out_path = get_output_subdir(OUTPUT_ESTIMATION, create=True) / "scaling_benchmark.csv"

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Benchmark complete: {len(results)} runs in {total_wall:.1f}s")
    print(f"Results written to: {out_path}")

    # Print summary table
    cols = ["N", "J", "K", "solve_time_sec", "total_time_sec", "mle_peak_rss_mb",
            "objective", "nll", "tau_error", "delta_l2", "qbar_l2", "success"]
    print(f"\n{df[cols].to_string(index=False)}")


if __name__ == "__main__":
    main()
