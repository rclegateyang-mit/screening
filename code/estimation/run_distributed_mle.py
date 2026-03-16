#!/usr/bin/env python3
"""Distributed MLE estimation using MPI.

Usage::

    mpirun -np 96 python -m code.estimation.run_distributed_mle [options]

Architecture::

    Rank 0 (root)                      Ranks 1..P-1
    ─────────────                      ─────────────
    Load data from disk                (idle)
    Scatter market partitions ──────→  Receive local partition
                                        (held in memory for all iterations)
    ┌─ Outer iteration ──────────────────────────────────────────────┐
    │  Bcast θ_G ──────────────────→  Receive θ_G                   │
    │  Solve local markets             Solve local markets           │
    │  ←── Reduce(nll, grad, H) ────  Send local sums               │
    │  Compute Newton step                                           │
    │  Line search (Bcast/Reduce)      Participate in line search    │
    └────────────────────────────────────────────────────────────────┘
    Gather final θ_m ←─────────────  Send local θ_m
    Save results

Data persistence: each rank holds only its ~M/P markets (~20 MB for 1000
markets / 96 ranks), received once via ``comm.scatter`` at startup.
No data is re-serialised during the outer loop.
"""

from __future__ import annotations

# ── CPU allocation for MPI + JAX ──────────────────────────────────────
#
# Each MPI rank uses ~2-3 CPU cores due to three thread pools:
#   1. OpenBLAS (linear algebra): defaults to MAX_THREADS=64
#   2. XLA/Eigen (JAX runtime): defaults to num_cores threads
#   3. OpenMP (scipy internals): defaults to num_cores
#
# Setting T=1 for BLAS/OpenMP eliminates pool #1 and #3, but XLA's
# internal thread pool (#2) cannot be controlled via env vars and still
# uses ~1-2 extra cores per rank.
#
# To limit total CPU usage, wrap mpirun with taskset:
#   taskset -c 0-47 mpirun -np 40 python -m ...
# This hard-caps ALL ranks to the specified CPU set (here: 48 of 96
# logical CPUs = 50%), regardless of XLA's thread count.
#
# The env vars below pin BLAS/OpenMP to 1 thread per rank. Set them
# BEFORE mpirun for reliability (shared libs read them at dlopen time):
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#   taskset -c 0-47 mpirun -np 40 python -m ...
#
# The lines below are a fallback for when the caller forgets.
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = os.environ.get(_var, "1")
os.environ["XLA_FLAGS"] = os.environ.get(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false"
)

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI

# JAX init (every rank)
from code.estimation.distributed_worker import MarketData, compute_naive_init, init_jax
init_jax()

# Now safe to import JAX-dependent modules
from code.estimation.distributed_master import GlobalTransform, run_outer_loop

try:
    from .. import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_ESTIMATION
    from .helpers import read_parameters
    from .run_mle_penalty_phi_sigma_jax import (
        Baseline, distance_metrics, make_param_names_raw,
        transform_theta_and_jacobian,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from code import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_ESTIMATION
    from helpers import read_parameters  # type: ignore
    from run_mle_penalty_phi_sigma_jax import (  # type: ignore
        Baseline, distance_metrics, make_param_names_raw,
        transform_theta_and_jacobian,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    raw_dir = get_data_subdir(DATA_RAW, create=True)
    clean_dir = get_data_subdir(DATA_CLEAN, create=True)
    build_dir = get_data_subdir(DATA_BUILD, create=True)
    est_dir = get_output_subdir(OUTPUT_ESTIMATION, create=True)

    p = argparse.ArgumentParser(
        description="Distributed MLE via MPI: parallel per-market inner solve + outer Newton",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--workers_path", type=str,
                   default=str(build_dir / "workers_dataset.csv"))
    p.add_argument("--firms_path", type=str,
                   default=str(clean_dir / "equilibrium_firms.csv"))
    p.add_argument("--params_path", type=str,
                   default=str(raw_dir / "parameters_effective.csv"))
    p.add_argument("--out_dir", type=str, default=str(est_dir))
    p.add_argument("--theta0_file", type=str, default=None,
                   help="JSON file with previous estimates to warm-start from.")
    p.add_argument("--outer_method", type=str, default="L-BFGS-B",
                   choices=["L-BFGS-B", "trust-ncg", "Newton-CG"],
                   help="Outer scipy.optimize method. 'L-BFGS-B' (gradient-only) "
                        "or 'trust-ncg'/'Newton-CG' (with Schur Hessian).")
    p.add_argument("--outer_maxiter", type=int, default=50)
    p.add_argument("--outer_tol", type=float, default=1e-5)
    p.add_argument("--inner_maxiter", type=int, default=200)
    p.add_argument("--inner_tol", type=float, default=1e-6)
    p.add_argument("--freeze", type=str, default=None,
                   help="Comma-separated global param names to freeze "
                        "(tau, alpha, gamma, sigma_e, lambda_e).")
    p.add_argument("--M", type=int, default=None,
                   help="Use only the first M markets from the dataset (default: all)")
    p.add_argument("--skip_naive_init", action="store_true",
                   help="Skip naive initialization; use true parameter values "
                        "for delta/qbar init (for debugging/comparison).")
    p.add_argument("--skip_statistics", action="store_true")
    p.add_argument("--skip_plot", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (rank 0 only)
# ---------------------------------------------------------------------------


def load_market_datas(firms_path: str, workers_path: str, mu_e: float,
                      M_subset: int | None = None,
                      ) -> Tuple[List[MarketData], dict]:
    """Load multi-market data and split into per-market MarketData objects."""
    from scipy.spatial.distance import cdist

    firms_df = pd.read_csv(firms_path)
    workers_df = pd.read_csv(workers_path)

    if 'market_id' not in firms_df.columns:
        raise ValueError("Distributed MLE requires multi-market data.")

    firms_df = firms_df.sort_values(['market_id', 'firm_id']).reset_index(drop=True)
    workers_df = workers_df.sort_values(['market_id']).reset_index(drop=True)

    markets = sorted(firms_df['market_id'].unique())
    if M_subset is not None:
        markets = markets[:M_subset]
        firms_df = firms_df[firms_df['market_id'].isin(markets)].reset_index(drop=True)
        workers_df = workers_df[workers_df['market_id'].isin(markets)].reset_index(drop=True)
    M = len(markets)

    qbar_col = 'qbar' if 'qbar' in firms_df.columns else 'c'

    market_datas = []
    J_per_list = []
    w_flat, xi_flat, qbar_flat, Y_flat = [], [], [], []

    for i, mid in enumerate(markets):
        fdf = firms_df[firms_df['market_id'] == mid].sort_values('firm_id')
        wdf = workers_df[workers_df['market_id'] == mid]

        J_m = len(fdf)
        N_m = len(wdf)

        X_m = wdf['x_skill'].values.astype(np.float64)
        choice_m = wdf['chosen_firm'].values.astype(np.int32)
        w_m = fdf['w'].values.astype(np.float64)
        Y_m = fdf['Y'].values.astype(np.float64)

        loc_firms = fdf[['x', 'y']].values
        worker_locs = np.column_stack([wdf['ell_x'].values, wdf['ell_y'].values])
        D_m = cdist(worker_locs, loc_firms, metric='euclidean').astype(np.float64)

        counts = np.bincount(choice_m, minlength=J_m + 1).astype(np.float64)
        labor_m = counts[1:]

        market_datas.append(MarketData(
            market_id=i, X_m=X_m, choice_m=choice_m, D_m=D_m,
            w_m=w_m, Y_m=Y_m, labor_m=labor_m,
            mu_e=mu_e, J_per=J_m, N_per=N_m,
        ))
        J_per_list.append(J_m)

        w_flat.append(w_m)
        xi_flat.append(fdf['xi'].values.astype(np.float64))
        qbar_flat.append(fdf[qbar_col].values.astype(np.float64))
        Y_flat.append(Y_m)

    meta = {
        "M": M, "J_per_list": J_per_list,
        "w_flat": np.concatenate(w_flat),
        "xi_flat": np.concatenate(xi_flat),
        "qbar_flat": np.concatenate(qbar_flat),
        "Y_flat": np.concatenate(Y_flat),
    }
    return market_datas, meta


def partition_markets(market_datas: List[MarketData], size: int,
                      ) -> List[List[MarketData]]:
    """Round-robin partition of M markets into `size` chunks."""
    partitions: List[List[MarketData]] = [[] for _ in range(size)]
    for i, md in enumerate(market_datas):
        partitions[i % size].append(md)
    return partitions


# ---------------------------------------------------------------------------
# Frozen parameter parsing
# ---------------------------------------------------------------------------

_GLOBAL_PARAM_MAP = {"tau": 0, "alpha": 1, "gamma": 2, "sigma_e": 3, "lambda_e": 4}


def parse_frozen(freeze_str: str | None) -> list[int]:
    if not freeze_str:
        return []
    indices = []
    for tok in freeze_str.split(","):
        tok = tok.strip()
        if tok in _GLOBAL_PARAM_MAP:
            indices.append(_GLOBAL_PARAM_MAP[tok])
        elif tok:
            print(f"  WARNING: '{tok}' is not a global parameter, ignoring")
    return indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()  # all ranks parse (same argv)
    out_dir = Path(args.out_dir)

    total_start = time.perf_counter()

    # ---- Load parameters (all ranks, small file) ----
    params = read_parameters(args.params_path)
    alpha_baseline = float(params.get("alpha", params.get("beta", 0.5)))
    gamma_baseline = float(params.get("gamma", 1.0))
    mu_e = float(params.get("mu_e", params.get("mu_a", 0.0)))
    sigma_baseline = float(params.get("sigma_e", params.get("sigma_a", 0.12)))
    tau_baseline = float(params.get("tau", params.get("gamma", 0.05)))
    eta_param = float(params.get("eta", 1.0))

    # ---- Rank 0 loads data and partitions ----
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"MPI distributed MLE: {size} ranks")
        print("Loading multi-market data...")

        market_datas, meta = load_market_datas(
            args.firms_path, args.workers_path, mu_e, M_subset=args.M)
        M = meta["M"]
        J_per_list = meta["J_per_list"]
        J_total = sum(J_per_list)

        partitions = partition_markets(market_datas, size)
        sizes_str = ", ".join(str(len(p)) for p in partitions[:5])
        J_min, J_max = min(J_per_list), max(J_per_list)
        j_str = str(J_min) if J_min == J_max else f"{J_min}-{J_max}"
        print(f"  M={M}, J_per={j_str}, J_total={J_total}, partitioned into "
              f"{size} ranks [{sizes_str}, ...]")
    else:
        partitions = None
        meta = None

    # ---- Scatter market partitions (data sent once, held for all iters) ----
    local_markets: List[MarketData] = comm.scatter(partitions, root=0)
    meta = comm.bcast(meta, root=0)

    M = meta["M"]
    J_per_list = meta["J_per_list"]
    J_total = sum(J_per_list)

    if rank == 0:
        data_per_rank = sum(
            md.D_m.nbytes + md.X_m.nbytes + md.choice_m.nbytes +
            md.w_m.nbytes + md.Y_m.nbytes + md.labor_m.nbytes
            for md in local_markets) / 1e6
        print(f"  Rank 0 holds {len(local_markets)} markets "
              f"({data_per_rank:.1f} MB)")

    # ---- Initialization: naive (from data) or true-value fallback ----
    if args.skip_naive_init:
        # Legacy path: initialize delta/qbar from true values (requires xi, qbar in data)
        if rank == 0:
            print("Using true-value initialization (--skip_naive_init)")
            w_flat = meta["w_flat"]
            xi_flat = meta["xi_flat"]
            qbar_flat = meta["qbar_flat"]
            delta0_flat = eta_param * np.log(np.maximum(w_flat, 1e-300)) + xi_flat
            qbar0_flat = np.maximum(qbar_flat, 1e-10)

            J_offsets = np.cumsum([0] + J_per_list)
            all_theta_m_inits = []
            for i in range(M):
                dm = delta0_flat[J_offsets[i]:J_offsets[i + 1]]
                qm = qbar0_flat[J_offsets[i]:J_offsets[i + 1]]
                all_theta_m_inits.append(np.concatenate([dm, qm]))

            init_partitions = [[] for _ in range(size)]
            for i, tm in enumerate(all_theta_m_inits):
                init_partitions[i % size].append(tm)
        else:
            init_partitions = None
            delta0_flat = None
            qbar0_flat = None

        local_theta_m_inits = comm.scatter(init_partitions, root=0)
        theta_G_init = np.array([
            tau_baseline, alpha_baseline, gamma_baseline, sigma_baseline, 0.0,
        ], dtype=np.float64)
    else:
        # Naive initialization: MNL for (tau, delta), FOC for qbar — all from data
        if rank == 0:
            print("Computing naive initial guesses (MNL + FOC)...")
        init_start = time.perf_counter()

        local_tau_sum = 0.0
        local_N = 0
        local_theta_m_inits = []
        for md in local_markets:
            tau_hat, theta_m = compute_naive_init(
                md, alpha_baseline, gamma_baseline)
            local_tau_sum += tau_hat * md.N_per
            local_N += md.N_per
            local_theta_m_inits.append(theta_m)

        # Weighted average of tau across all markets
        global_tau_sum = comm.reduce(local_tau_sum, op=MPI.SUM, root=0)
        global_N = comm.reduce(local_N, op=MPI.SUM, root=0)
        if rank == 0:
            tau_avg = global_tau_sum / global_N
        else:
            tau_avg = None
        tau_avg = comm.bcast(tau_avg, root=0)

        theta_G_init = np.array([
            tau_avg, alpha_baseline, gamma_baseline, sigma_baseline, 0.0,
        ], dtype=np.float64)

        # Build flat init arrays for reporting (rank 0 only)
        if rank == 0:
            # Gather all theta_m inits for delta0_flat / qbar0_flat
            all_theta_m_list = comm.gather(local_theta_m_inits, root=0)
            # Reconstruct in market order (round-robin was used for partitioning)
            all_theta_m_ordered = [None] * M
            for r_idx, chunk in enumerate(all_theta_m_list):
                for local_idx, tm in enumerate(chunk):
                    global_idx = local_idx * size + r_idx
                    all_theta_m_ordered[global_idx] = tm
            delta0_flat = np.concatenate(
                [tm[:len(tm) // 2] for tm in all_theta_m_ordered])
            qbar0_flat = np.concatenate(
                [tm[len(tm) // 2:] for tm in all_theta_m_ordered])
        else:
            comm.gather(local_theta_m_inits, root=0)
            delta0_flat = None
            qbar0_flat = None

        init_time = time.perf_counter() - init_start
        if rank == 0:
            print(f"  Naive init: tau_avg={tau_avg:.4f}  ({init_time:.1f}s)")

    # Warm-start from previous estimates
    if args.theta0_file:
        if rank == 0:
            print(f"Loading warm-start from {args.theta0_file} ...")
            with open(args.theta0_file) as f:
                prev = json.load(f)
            prev_theta = np.asarray(prev["theta_hat"], dtype=np.float64)
            theta_G_init = prev_theta[:5].copy()
            prev_delta = prev_theta[5:5 + J_total]
            prev_qbar = prev_theta[5 + J_total:]
            J_offsets = np.cumsum([0] + J_per_list)
            all_prev_inits = []
            for i in range(M):
                dm = prev_delta[J_offsets[i]:J_offsets[i + 1]]
                qm = prev_qbar[J_offsets[i]:J_offsets[i + 1]]
                all_prev_inits.append(np.concatenate([dm, qm]))
            prev_partitions = [[] for _ in range(size)]
            for i, tm in enumerate(all_prev_inits):
                prev_partitions[i % size].append(tm)
        else:
            prev_partitions = None
        theta_G_init = comm.bcast(theta_G_init, root=0)
        local_theta_m_inits = comm.scatter(prev_partitions, root=0)

    # ---- Frozen parameters ----
    frozen_indices = parse_frozen(args.freeze)
    if rank == 0 and frozen_indices:
        names = [k for k, v in _GLOBAL_PARAM_MAP.items() if v in frozen_indices]
        print(f"Freezing global parameters: {names}")

    # ---- Baseline for distance_metrics (true values from DGP) ----
    if rank == 0:
        w_true = meta["w_flat"]
        xi_true = meta["xi_flat"]
        qbar_true = meta["qbar_flat"]
        delta_true = eta_param * np.log(np.maximum(w_true, 1e-300)) + xi_true
        baseline = Baseline(
            tau=tau_baseline, alpha=alpha_baseline,
            gamma=gamma_baseline, sigma_e_baseline=sigma_baseline,
            sigma_e=sigma_baseline, lambda_e=0.0,
            delta=delta_true.astype(float),
            Qbar=np.maximum(qbar_true, 1e-10).astype(float),
        )

    build_time = time.perf_counter() - total_start
    if rank == 0:
        print(f"\nStarting outer loop ({args.outer_method}, "
              f"{size} ranks, {M} markets)...")

    # ---- Run outer loop (SPMD — all ranks) ----
    needs_hessian = args.outer_method in ("trust-ncg", "Newton-CG")
    solve_start = time.perf_counter()
    result = run_outer_loop(
        comm, local_markets, local_theta_m_inits, theta_G_init,
        max_outer_iter=args.outer_maxiter,
        outer_tol=args.outer_tol,
        inner_maxiter=args.inner_maxiter,
        inner_tol=args.inner_tol,
        compute_hessian=needs_hessian,
        frozen_indices=frozen_indices,
        method=args.outer_method,
        verbose=(rank == 0),
    )
    solve_time = time.perf_counter() - solve_start
    total_time = time.perf_counter() - total_start

    # ---- Rank 0: reconstruct full θ and save ----
    if rank == 0:
        theta_G_hat = result["theta_G_hat"]
        theta_m_hats = result["theta_m_hats"]

        delta_all = np.concatenate([tm[:len(tm) // 2] for tm in theta_m_hats])
        qbar_all = np.concatenate([tm[len(tm) // 2:] for tm in theta_m_hats])
        theta_hat_full = np.concatenate([theta_G_hat, delta_all, qbar_all])

        # Save per-market inner estimates
        inner_rows = []
        for m_idx, tm in enumerate(theta_m_hats):
            J_m = len(tm) // 2
            for j in range(J_m):
                inner_rows.append({
                    "market_id": m_idx + 1,
                    "firm_j": j + 1,
                    "delta_hat": float(tm[j]),
                    "qbar_hat": float(tm[J_m + j]),
                })
        inner_df = pd.DataFrame(inner_rows)
        inner_path = out_dir / "mle_distributed_inner_estimates.csv"
        inner_df.to_csv(inner_path, index=False)
        print(f"Inner estimates ({len(inner_rows)} firm×market) written to {inner_path}")

        total_nll = result["total_nll"]
        converged = result["converged"]
        n_outer = result["n_outer_iters"]
        final_grad = result["final_grad_norm_z"]

        scipy_info = result.get("scipy_result", {})

        print(f"\n{'='*60}")
        print(f"Distributed MLE complete")
        print(f"  MPI ranks: {size}")
        print(f"  Method: {args.outer_method}")
        print(f"  Converged: {converged}")
        if scipy_info:
            print(f"  scipy message: {scipy_info.get('message', '')}")
        print(f"  Outer iterations: {n_outer}")
        print(f"  Final NLL: {total_nll:.4f}")
        print(f"  Final |grad_z|: {final_grad:.3e}")
        print(f"  Build: {build_time:.2f}s  Solve: {solve_time:.2f}s  "
              f"Total: {total_time:.2f}s")
        print(f"{'='*60}")

        dist = distance_metrics(theta_hat_full, baseline)
        print("\n=== Parameter Recovery ===")
        for k in ("tau", "alpha", "gamma", "sigma_e"):
            print(f"  {k:10s}: hat={dist[f'{k}_hat']:.6f}  "
                  f"true={dist[f'{k}_true']:.6f}  "
                  f"err={dist[f'{k}_error']:+.6f}")
        print(f"  {'delta':10s}: RMSE/SD={dist['delta_rmse_over_sd']:.4f}  "
              f"corr={dist['delta_corr']:.4f}")
        print(f"  {'qbar':10s}: RMSE/SD={dist['qbar_rmse_over_sd']:.4f}  "
              f"corr={dist['qbar_corr']:.4f}")

        theta_hat_tilde, _ = transform_theta_and_jacobian(theta_hat_full, J_total)
        history = result["history"]
        param_names = make_param_names_raw(J_total)
        frozen_names = [k for k, v in _GLOBAL_PARAM_MAP.items()
                        if v in frozen_indices] if frozen_indices else []

        out = {
            "solver": f"distributed_mpi_{args.outer_method}",
            "frozen_params": frozen_names,
            "penalty_weight": 0.0,
            "objective": total_nll,
            "objective_breakdown": {
                "neg_log_likelihood": total_nll,
                "penalty": 0.0,
            },
            "nit": n_outer,
            "grad_norm": final_grad,
            "theta0": np.concatenate([
                theta_G_init, delta0_flat, qbar0_flat,
            ]).tolist(),
            "theta_hat": theta_hat_full.tolist(),
            "theta_hat_transformed": theta_hat_tilde.tolist(),
            "delta_method": "distributed_mpi",
            "time_sec": total_time,
            "timings": {
                "build_time_sec": build_time,
                "solve_time_sec": solve_time,
                "total_time_sec": total_time,
                "per_iter_sec": [h.wall_sec for h in history],
            },
            "distance_metrics": dist,
            "param_names": param_names,
            "outer_method": args.outer_method,
            "mpi_ranks": size,
            "inner_maxiter": args.inner_maxiter,
            "inner_tol": args.inner_tol,
            "outer_maxiter": args.outer_maxiter,
            "outer_tol": args.outer_tol,
            "M": M, "J_per_list": J_per_list, "J_total": J_total,
            "converged": converged,
            "true_params": {
                "tau": baseline.tau, "alpha": baseline.alpha,
                "gamma": baseline.gamma,
                "sigma_e_baseline": baseline.sigma_e_baseline,
                "sigma_e": baseline.sigma_e, "lambda_e": baseline.lambda_e,
                "delta": baseline.delta.tolist(),
                "qbar": baseline.Qbar.tolist(),
            },
            "history": [
                {"iter": h.iteration, "nll": h.nll,
                 "grad_norm_z": h.grad_norm_z,
                 "grad_norm_theta": h.grad_norm_theta,
                 "step_size": h.step_size,
                 "inner_converged": h.n_converged_inner,
                 "wall_sec": h.wall_sec}
                for h in history
            ],
        }

        out_path = out_dir / "mle_distributed_estimates_jax.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {out_path}")

        # ---- Append to scaling run log ----
        import datetime
        log_path = out_dir / "mle_distributed_run_log.csv"
        log_row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "M": M, "J_total": J_total,
            "N_total": sum(md.N_per for md in local_markets),
            "K": 5 + 2 * J_total,
            "method": f"distributed_{args.outer_method}",
            "mpi_ranks": size,
            "outer_maxiter": args.outer_maxiter, "nit": n_outer,
            "inner_maxiter": args.inner_maxiter,
            "mle_converged": converged, "grad_norm": f"{final_grad:.4e}",
            "nll": f"{total_nll:.4f}",
            "penalty_weight": 0.0,
            "frozen_params": ",".join(frozen_names),
            "tau_hat": f"{dist['tau_hat']:.6f}",
            "tau_true": f"{dist['tau_true']:.6f}",
            "tau_error": f"{dist['tau_error']:.6f}",
            "alpha_hat": f"{dist['alpha_hat']:.6f}",
            "alpha_true": f"{dist['alpha_true']:.6f}",
            "delta_rmse_over_sd": f"{dist['delta_rmse_over_sd']:.4f}",
            "delta_corr": f"{dist['delta_corr']:.4f}",
            "qbar_rmse_over_sd": f"{dist['qbar_rmse_over_sd']:.4f}",
            "qbar_corr": f"{dist['qbar_corr']:.4f}",
            "build_time_s": f"{build_time:.2f}",
            "solve_time_s": f"{solve_time:.2f}",
            "total_time_s": f"{total_time:.2f}",
            "per_iter_s": f"{solve_time / max(n_outer, 1):.2f}",
        }
        log_df = pd.DataFrame([log_row])
        write_header = not log_path.exists()
        log_df.to_csv(log_path, mode="a", header=write_header, index=False)
        print(f"Run logged to {log_path}")

    MPI.Finalize()


if __name__ == "__main__":
    main()
