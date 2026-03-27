#!/usr/bin/env python3
"""Distributed MLE estimation using MPI.

Optimizes θ_G = [τ, γ̃] (tau and tilde_gamma) globally, profiling out
per-market δ (delta) and q̃ (tilde_q) via parallel inner solves.
Other parameters (α, σ_e, γ₀) are fixed at values from the pooled
naive initializer.

Usage::

    mpirun -np 40 python -m screening.analysis.mle.run_distributed \\
        --firms_path /path/to/equilibrium_firms.csv \\
        --workers_path /path/to/workers_dataset.csv \\
        --params_path /path/to/parameters_effective.csv \\
        --M 100
"""

from __future__ import annotations

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
from screening.analysis.mle.distributed_worker import MarketData, init_jax
init_jax()

from screening.analysis.mle.distributed_master import (
    run_inner_delta_only, run_inner_only, run_inner_profile_delta, run_outer_loop,
)

try:
    from ... import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_ESTIMATION
    from ..lib.helpers import read_parameters
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from screening import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_ESTIMATION
    from screening.analysis.lib.helpers import read_parameters  # type: ignore


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    raw_dir = get_data_subdir(DATA_RAW, create=True)
    clean_dir = get_data_subdir(DATA_CLEAN, create=True)
    build_dir = get_data_subdir(DATA_BUILD, create=True)
    est_dir = get_output_subdir(OUTPUT_ESTIMATION, create=True)

    p = argparse.ArgumentParser(
        description="Distributed MLE: optimize (tau, tilde_gamma) via MPI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--workers_path", type=str,
                   default=str(build_dir / "workers_dataset.csv"))
    p.add_argument("--firms_path", type=str,
                   default=str(clean_dir / "equilibrium_firms.csv"))
    p.add_argument("--params_path", type=str,
                   default=str(raw_dir / "parameters_effective.csv"))
    p.add_argument("--out_dir", type=str, default=str(est_dir))
    p.add_argument("--outer_method", type=str, default="L-BFGS-B",
                   choices=["L-BFGS-B", "trust-ncg", "Newton-CG"])
    p.add_argument("--outer_maxiter", type=int, default=200)
    p.add_argument("--outer_tol", type=float, default=1e-8)
    p.add_argument("--inner_maxiter", type=int, default=1000)
    p.add_argument("--inner_tol", type=float, default=1e-8)
    p.add_argument("--M", type=int, default=None,
                   help="Use only the first M markets (default: all)")
    p.add_argument("--freeze_globals", action="store_true",
                   help="Fix theta_G at true values; only solve inner (delta, tilde_q)")
    p.add_argument("--freeze_tilde_q", action="store_true",
                   help="Fix tilde_q at true values; only solve delta "
                        "(implies --freeze_globals)")
    p.add_argument("--profile_delta", action="store_true",
                   help="Profile out delta via BLP contraction; optimize "
                        "tilde_q only (implies --freeze_globals)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (rank 0 only)
# ---------------------------------------------------------------------------


def load_market_datas(firms_path: str, workers_path: str,
                      gamma0: float, alpha: float, sigma_e: float,
                      M_subset: int | None = None,
                      ) -> Tuple[List[MarketData], dict]:
    """Load multi-market data into MarketData objects with fixed params."""
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
    has_z1 = 'z1' in firms_df.columns
    has_z2 = 'z2' in firms_df.columns

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

        z1_m = fdf['z1'].values.astype(np.float64) if has_z1 else np.zeros(J_m, dtype=np.float64)
        z2_m = fdf['z2'].values.astype(np.float64) if has_z2 else np.zeros(J_m, dtype=np.float64)

        market_datas.append(MarketData(
            market_id=i, X_m=X_m, choice_m=choice_m, D_m=D_m,
            w_m=w_m, Y_m=Y_m, labor_m=labor_m,
            gamma0=gamma0, alpha=alpha, sigma_e=sigma_e,
            J_per=J_m, N_per=N_m,
            z1_m=z1_m, z2_m=z2_m,
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()
    out_dir = Path(args.out_dir)

    total_start = time.perf_counter()

    # ---- Load parameters (all ranks, small file) ----
    params = read_parameters(args.params_path)
    gamma0_val = float(params.get("gamma0", 0.0))
    alpha_val = float(params.get("alpha", params.get("beta", 0.2)))
    sigma_e_val = float(params.get("sigma_e", params.get("sigma_a", 0.135)))
    eta_param = float(params.get("eta", 7.0))
    tau_true = float(params.get("tau", 0.4))
    gamma1_val = float(params.get("gamma1", params.get("gamma", 0.94)))
    tilde_gamma_true = gamma1_val / sigma_e_val if sigma_e_val > 0 else 0.0

    # ---- Rank 0 loads data and partitions ----
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Distributed MLE: {size} ranks, θ_G = [tau, tilde_gamma]")
        print("Loading multi-market data...")

        market_datas, meta = load_market_datas(
            args.firms_path, args.workers_path,
            gamma0_val, alpha_val, sigma_e_val,
            M_subset=args.M)
        M = meta["M"]
        J_per_list = meta["J_per_list"]
        J_total = sum(J_per_list)

        partitions = partition_markets(market_datas, size)
        J_min, J_max = min(J_per_list), max(J_per_list)
        j_str = str(J_min) if J_min == J_max else f"{J_min}-{J_max}"
        print(f"  M={M}, J_per={j_str}, J_total={J_total}")
        print(f"  Fixed: alpha={alpha_val}, sigma_e={sigma_e_val}, gamma0={gamma0_val}")
        print(f"  True: tau={tau_true}, tilde_gamma={tilde_gamma_true:.4f}")
    else:
        partitions = None
        meta = None
        market_datas = None

    # ---- Scatter market partitions ----
    local_markets: List[MarketData] = comm.scatter(partitions, root=0)
    meta = comm.bcast(meta, root=0)

    M = meta["M"]
    J_per_list = meta["J_per_list"]
    J_total = sum(J_per_list)

    # ---- Pooled naive initialization (rank 0 pre-scatter) ----
    # We need the HybridMarketData format for naive_init; convert MarketData
    if rank == 0:
        print("Computing pooled naive initial guesses...")
        init_start = time.perf_counter()

        from screening.analysis.lib.naive_init import compute_pooled_naive_init
        # naive_init expects HybridMarketData; create lightweight adapter
        from screening.analysis.auglag.worker import HybridMarketData
        hybrid_mds = []
        for md in market_datas:
            hmd = HybridMarketData(
                market_id=md.market_id,
                v=md.X_m,
                choice_idx=md.choice_m,
                D=md.D_m,
                w=md.w_m,
                R=md.Y_m,
                L=md.labor_m,
                z1=md.z1_m if md.z1_m is not None else np.zeros(md.J_per),
                z2=md.z2_m if md.z2_m is not None else np.zeros(md.J_per),
                z3=np.ones(md.J_per, dtype=np.float64),
                omega=md.J_per / J_total,
                J=md.J_per,
                N=md.N_per,
            )
            hybrid_mds.append(hmd)

        theta_G_full, all_deltas, all_tilde_qs = compute_pooled_naive_init(hybrid_mds)
        # theta_G_full = [tau, tilde_gamma, alpha, sigma_e, eta, gamma0]
        tau_init = theta_G_full[0]
        tilde_gamma_init = theta_G_full[1]
        theta_G_init = np.array([tau_init, tilde_gamma_init], dtype=np.float64)

        init_time = time.perf_counter() - init_start
        print(f"  tau_init={tau_init:.4f}, tilde_gamma_init={tilde_gamma_init:.4f}  "
              f"({init_time:.1f}s)")

        # Build per-market theta_m = [delta, tilde_q] and partition
        init_partitions = [[] for _ in range(size)]
        for i, md in enumerate(market_datas):
            theta_m = np.concatenate([all_deltas[i], all_tilde_qs[i]])
            init_partitions[i % size].append(theta_m)
    else:
        init_partitions = None
        theta_G_init = None

    local_theta_m_inits = comm.scatter(init_partitions, root=0)
    theta_G_init = comm.bcast(theta_G_init, root=0)

    # ---- Compute true tilde_q for delta-only mode ----
    if args.freeze_tilde_q:
        if rank == 0:
            mu_v_val = float(params.get("mu_v", 10.84))
            tilde_gamma_true_val = gamma1_val / sigma_e_val
            tq_partitions = [[] for _ in range(size)]
            for i, md in enumerate(market_datas):
                qbar_m = meta["qbar_flat"][sum(J_per_list[:i]):sum(J_per_list[:i+1])]
                tq_true_m = (np.log(np.maximum(qbar_m, 1e-300)) - gamma0_val) / sigma_e_val
                tq_partitions[i % size].append(tq_true_m.astype(np.float64))
        else:
            tq_partitions = None
        local_tilde_q_fixed = comm.scatter(tq_partitions, root=0)
    else:
        local_tilde_q_fixed = None

    build_time = time.perf_counter() - total_start

    # ---- Run solve (SPMD — all ranks) ----
    solve_start = time.perf_counter()

    if args.profile_delta:
        # Profile out delta via contraction, optimize tilde_q only
        theta_G_true = np.array([tau_true, tilde_gamma_true], dtype=np.float64)
        if rank == 0:
            print(f"\nProfile-delta mode: theta_G fixed at true values "
                  f"[tau={tau_true}, tg={tilde_gamma_true:.4f}]")
            print(f"  Delta profiled via BLP contraction; optimizing tilde_q")
        result = run_inner_profile_delta(
            comm, local_markets, local_theta_m_inits, theta_G_true,
            inner_maxiter=args.inner_maxiter,
            inner_tol=args.inner_tol,
            verbose=(rank == 0),
        )
    elif args.freeze_tilde_q:
        # Fix theta_G AND tilde_q at true values, only solve delta
        theta_G_true = np.array([tau_true, tilde_gamma_true], dtype=np.float64)
        if rank == 0:
            print(f"\nDelta-only mode: theta_G fixed at true values "
                  f"[tau={tau_true}, tg={tilde_gamma_true:.4f}]")
            print(f"  tilde_q fixed at true values")
            print(f"  Running delta-only solve ({size} ranks, {M} markets)...")
        result = run_inner_delta_only(
            comm, local_markets, local_theta_m_inits, local_tilde_q_fixed,
            theta_G_true,
            inner_maxiter=args.inner_maxiter,
            inner_tol=args.inner_tol,
            verbose=(rank == 0),
        )
    elif args.freeze_globals:
        # Fix theta_G at true values, only solve inner (delta, tilde_q)
        theta_G_true = np.array([tau_true, tilde_gamma_true], dtype=np.float64)
        if rank == 0:
            print(f"\nFreeze-globals mode: theta_G fixed at true values "
                  f"[tau={tau_true}, tg={tilde_gamma_true:.4f}]")
            print(f"  Running inner-only solve ({size} ranks, {M} markets)...")
        result = run_inner_only(
            comm, local_markets, local_theta_m_inits, theta_G_true,
            inner_maxiter=args.inner_maxiter,
            inner_tol=args.inner_tol,
            verbose=(rank == 0),
        )
    else:
        needs_hessian = args.outer_method in ("trust-ncg", "Newton-CG")
        if rank == 0:
            print(f"\nStarting outer loop ({args.outer_method}, "
                  f"{size} ranks, {M} markets)...")
        result = run_outer_loop(
            comm, local_markets, local_theta_m_inits, theta_G_init,
            max_outer_iter=args.outer_maxiter,
            outer_tol=args.outer_tol,
            inner_maxiter=args.inner_maxiter,
            inner_tol=args.inner_tol,
            compute_hessian=needs_hessian,
            method=args.outer_method,
            verbose=(rank == 0),
        )
    solve_time = time.perf_counter() - solve_start
    total_time = time.perf_counter() - total_start

    # ---- Rank 0: output ----
    if rank == 0:
        theta_G_hat = result["theta_G_hat"]
        theta_m_hats = result["theta_m_hats"]

        tau_hat, tg_hat = theta_G_hat[0], theta_G_hat[1]

        delta_all = np.concatenate([tm[:len(tm) // 2] for tm in theta_m_hats])
        tq_all = np.concatenate([tm[len(tm) // 2:] for tm in theta_m_hats])

        # Save per-market inner estimates
        inner_rows = []
        for m_idx, tm in enumerate(theta_m_hats):
            J_m = len(tm) // 2
            for j in range(J_m):
                inner_rows.append({
                    "market_id": m_idx + 1,
                    "firm_j": j + 1,
                    "delta_hat": float(tm[j]),
                    "tilde_q_hat": float(tm[J_m + j]),
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

        # Compute delta/tilde_q recovery metrics
        w_true = meta["w_flat"]
        xi_true = meta["xi_flat"]
        qbar_true = meta["qbar_flat"]
        delta_true = eta_param * np.log(np.maximum(w_true, 1e-300)) + xi_true
        tq_true = (np.log(np.maximum(qbar_true, 1e-300)) - gamma0_val) / sigma_e_val

        delta_corr = float(np.corrcoef(delta_all, delta_true)[0, 1]) if len(delta_all) > 1 else 0.0
        delta_rmse = float(np.sqrt(np.mean((delta_all - delta_true) ** 2)))
        tq_corr = float(np.corrcoef(tq_all, tq_true)[0, 1]) if len(tq_all) > 1 else 0.0
        tq_rmse = float(np.sqrt(np.mean((tq_all - tq_true) ** 2)))

        print(f"\n{'='*60}")
        print(f"Distributed MLE complete")
        print(f"  MPI ranks: {size}")
        print(f"  Converged: {converged}")
        if scipy_info:
            print(f"  scipy: {scipy_info.get('message', '')}")
        print(f"  Outer iterations: {n_outer}")
        print(f"  Final NLL: {total_nll:.4f}")
        print(f"  Final |grad_z|: {final_grad:.3e}")
        print(f"  Build: {build_time:.2f}s  Solve: {solve_time:.2f}s  "
              f"Total: {total_time:.2f}s")
        print(f"{'='*60}")

        print("\n=== Parameter Recovery ===")
        print(f"  {'tau':15s}: hat={tau_hat:.6f}  true={tau_true:.6f}  "
              f"err={tau_hat - tau_true:+.6f}")
        print(f"  {'tilde_gamma':15s}: hat={tg_hat:.6f}  true={tilde_gamma_true:.6f}  "
              f"err={tg_hat - tilde_gamma_true:+.6f}")
        print(f"  {'delta':15s}: corr={delta_corr:.4f}  RMSE={delta_rmse:.4f}")
        print(f"  {'tilde_q':15s}: corr={tq_corr:.4f}  RMSE={tq_rmse:.4f}")

        history = result["history"]

        out = {
            "solver": f"distributed_mpi_{args.outer_method}",
            "theta_G": theta_G_hat.tolist(),
            "theta_G_names": ["tau", "tilde_gamma"],
            "fixed_params": {
                "alpha": alpha_val,
                "sigma_e": sigma_e_val,
                "gamma0": gamma0_val,
            },
            "true_params": {
                "tau": tau_true,
                "tilde_gamma": tilde_gamma_true,
                "alpha": alpha_val,
                "sigma_e": sigma_e_val,
                "gamma0": gamma0_val,
                "eta": eta_param,
            },
            "objective": total_nll,
            "converged": converged,
            "n_outer_iters": n_outer,
            "final_grad_norm_z": final_grad,
            "mpi_ranks": size,
            "M": M, "J_per_list": J_per_list, "J_total": J_total,
            "recovery": {
                "tau_hat": tau_hat, "tau_true": tau_true,
                "tau_err": tau_hat - tau_true,
                "tg_hat": tg_hat, "tg_true": tilde_gamma_true,
                "tg_err": tg_hat - tilde_gamma_true,
                "delta_corr": delta_corr, "delta_rmse": delta_rmse,
                "tq_corr": tq_corr, "tq_rmse": tq_rmse,
            },
            "timings": {
                "build_time_sec": build_time,
                "solve_time_sec": solve_time,
                "total_time_sec": total_time,
                "per_iter_sec": [h.wall_sec for h in history],
            },
            "history": [
                {"iter": h.iteration, "nll": h.nll,
                 "grad_norm_z": h.grad_norm_z,
                 "grad_norm_theta": h.grad_norm_theta,
                 "inner_converged": h.n_converged_inner,
                 "wall_sec": h.wall_sec}
                for h in history
            ],
        }

        out_path = out_dir / "mle_distributed_estimates.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {out_path}")

        # ---- Run log ----
        import datetime
        log_path = out_dir / "mle_distributed_run_log.csv"
        log_row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "M": M, "J_total": J_total, "mpi_ranks": size,
            "outer_maxiter": args.outer_maxiter, "nit": n_outer,
            "converged": converged,
            "nll": f"{total_nll:.4f}",
            "tau_hat": f"{tau_hat:.6f}", "tau_true": f"{tau_true:.6f}",
            "tg_hat": f"{tg_hat:.6f}", "tg_true": f"{tilde_gamma_true:.6f}",
            "delta_corr": f"{delta_corr:.4f}", "delta_rmse": f"{delta_rmse:.4f}",
            "tq_corr": f"{tq_corr:.4f}", "tq_rmse": f"{tq_rmse:.4f}",
            "build_s": f"{build_time:.2f}",
            "solve_s": f"{solve_time:.2f}",
            "total_s": f"{total_time:.2f}",
        }
        log_df = pd.DataFrame([log_row])
        write_header = not log_path.exists()
        log_df.to_csv(log_path, mode="a", header=write_header, index=False)
        print(f"Run logged to {log_path}")

    MPI.Finalize()


if __name__ == "__main__":
    main()
