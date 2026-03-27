#!/usr/bin/env python3
"""Distributed augmented-Lagrangian hybrid MLE+GMM estimation using MPI.

Usage::

    mpirun -np P python -m screening.analysis.auglag.run_distributed [options]

Architecture::

    Rank 0 (root)                       Ranks 1..P-1
    ─────────────                       ─────────────
    Load data from disk                 (idle)
    Scatter market partitions ───────→  Receive local partition
    Compute W = (B'B)^{-1}, bcast ───→  Receive W
    ┌─ Outer iteration ───────────────────────────────────────────────┐
    │  Bcast CMD_MARKET_BLOCK ────────→  Market block (Step 2)        │
    │  [Bcast CMD_GLOBAL_EVAL ────────→  Global eval] × L-BFGS evals │
    │  mu/nu update, convergence check                                 │
    └──────────────────────────────────────────────────────────────────┘
    Gather final states ←────────────  Send local states
    Save results

Data persistence: each rank holds only its ~M/P markets,
received once via ``comm.scatter`` at startup.
"""

from __future__ import annotations

# ── CPU allocation for MPI + JAX ─────────────────────────────────────
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
from scipy.spatial.distance import cdist

# JAX init (every rank)
from screening.analysis.auglag.worker import (
    HybridMarketData,
    HybridMarketState,
    forward_eval,
    init_jax,
)
init_jax()

from screening.analysis.auglag.master import run_auglag_distributed

from screening import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_ESTIMATION


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    raw_dir = get_data_subdir(DATA_RAW, create=True)
    clean_dir = get_data_subdir(DATA_CLEAN, create=True)
    build_dir = get_data_subdir(DATA_BUILD, create=True)
    est_dir = get_output_subdir(OUTPUT_ESTIMATION, create=True)

    p = argparse.ArgumentParser(
        description="Distributed augmented-Lagrangian hybrid MLE+GMM via MPI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--firms_path", type=str,
                   default=str(clean_dir / "equilibrium_firms.csv"))
    p.add_argument("--workers_path", type=str,
                   default=str(build_dir / "workers_dataset.csv"))
    p.add_argument("--params_path", type=str,
                   default=str(raw_dir / "parameters_effective.csv"))
    p.add_argument("--out_dir", type=str, default=str(est_dir))
    p.add_argument("--M", type=int, default=None,
                   help="Use only the first M markets (default: all)")
    p.add_argument("--true_init", action="store_true",
                   help="Initialize from true parameter values")
    p.add_argument("--max_outer_iter", type=int, default=200)
    p.add_argument("--inner_maxiter", type=int, default=200)
    p.add_argument("--inner_tol", type=float, default=1e-6)
    p.add_argument("--global_maxiter", type=int, default=100)
    p.add_argument("--global_tol", type=float, default=1e-5)
    p.add_argument("--W_path", type=str, default=None,
                   help="JSON file with pre-computed W for two-step optimal GMM")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (rank 0 only)
# ---------------------------------------------------------------------------


def load_hybrid_market_datas(
    firms_path: str,
    workers_path: str,
    params_path: str,
    M_subset: int | None = None,
) -> Tuple[List[HybridMarketData], dict]:
    """Load multi-market data and return per-market HybridMarketData (np arrays)."""
    firms_df = pd.read_csv(firms_path)
    workers_df = pd.read_csv(workers_path)
    params_df = pd.read_csv(params_path)

    # Parse true parameters (with fallbacks for older parameter naming)
    pdict = dict(zip(params_df['parameter'], params_df['value']))
    _gamma1 = float(pdict.get('gamma1', pdict.get('gamma', 0.94)))
    _gamma0 = float(pdict.get('gamma0', 0.0))
    _sigma_e = float(pdict.get('sigma_e', pdict.get('sigma_a', 0.135)))
    true_params = {
        'tau': float(pdict['tau']),
        'alpha': float(pdict.get('alpha', pdict.get('beta', 0.2))),
        'gamma0': _gamma0,
        'gamma1': _gamma1,
        'sigma_e': _sigma_e,
        'eta': float(pdict['eta']),
    }
    true_params['tilde_gamma'] = true_params['gamma1'] / true_params['sigma_e']

    firms_df = firms_df.sort_values(['market_id', 'firm_id']).reset_index(drop=True)
    workers_df = workers_df.sort_values(['market_id']).reset_index(drop=True)

    markets = sorted(firms_df['market_id'].unique())
    if M_subset is not None:
        markets = markets[:M_subset]
        firms_df = firms_df[firms_df['market_id'].isin(markets)].reset_index(drop=True)
        workers_df = workers_df[workers_df['market_id'].isin(markets)].reset_index(drop=True)

    total_J = sum(len(firms_df[firms_df['market_id'] == m]) for m in markets)

    market_datas = []
    meta_firms = []
    J_per_list = []

    for mid in markets:
        fdf = firms_df[firms_df['market_id'] == mid].sort_values('firm_id')
        wdf = workers_df[workers_df['market_id'] == mid]

        J_m = len(fdf)
        N_m = len(wdf)

        v = wdf['x_skill'].values.astype(np.float64)
        choice_idx = wdf['chosen_firm'].values.astype(np.int32)
        w = fdf['w'].values.astype(np.float64)
        R = fdf['Y'].values.astype(np.float64)

        loc_firms = fdf[['x', 'y']].values
        worker_locs = np.column_stack([wdf['ell_x'].values, wdf['ell_y'].values])
        D = cdist(worker_locs, loc_firms, metric='euclidean').astype(np.float64)

        counts = np.bincount(choice_idx, minlength=J_m + 1).astype(np.float64)
        L = counts[1:]

        z1 = (fdf['z1'].values.astype(np.float64)
              if 'z1' in fdf.columns else np.zeros(J_m, dtype=np.float64))
        z2 = (fdf['z2'].values.astype(np.float64)
              if 'z2' in fdf.columns else np.zeros(J_m, dtype=np.float64))
        z3 = np.ones(J_m, dtype=np.float64)

        omega = J_m / total_J  # will be recomputed if M_subset changes

        md = HybridMarketData(
            market_id=int(mid), J=J_m, N=N_m,
            v=v, choice_idx=choice_idx, D=D,
            w=w, R=R, L=L, z1=z1, z2=z2, z3=z3,
            omega=omega,
        )
        market_datas.append(md)
        J_per_list.append(J_m)

        # Store firm-level true values for initialization
        qbar_col = 'qbar' if 'qbar' in fdf.columns else 'c'
        meta_firms.append({
            'w': w.copy(),
            'xi': fdf['xi'].values.astype(np.float64),
            'qbar': fdf[qbar_col].values.astype(np.float64),
            'Y': R.copy(),
        })

    meta = {
        'true_params': true_params,
        'meta_firms': meta_firms,
        'M': len(markets),
        'total_J': total_J,
        'J_per_list': J_per_list,
    }
    return market_datas, meta


# ---------------------------------------------------------------------------
# Weighting matrix W
# ---------------------------------------------------------------------------


def compute_2sls_W(market_datas: List[HybridMarketData]) -> np.ndarray:
    """W = (B'B)^{-1} where B is (total_J, 4) instrument matrix.

    Columns: z1, z2, z3, ones — where z3 is typically ones already
    (serving as a constant instrument).  If z3 ≡ 1, the last two
    columns would be collinear, so we detect this and use z1, z2, 1
    as a 3-instrument design, padding W to (4,4).
    """
    # Check if z3 is all-ones (constant)
    z3_is_const = all(np.allclose(md.z3, 1.0) for md in market_datas)

    if z3_is_const:
        # z3 = 1 is the constant instrument; moments m3 and m4 share it.
        # Use B = [z1, z2, ones] → W_33 = (B'B)^{-1} is (3,3).
        # Expand to (4,4): W[i,j] for i,j in {0,1} use z1/z2 block,
        # W[2,2] = W[3,3] = W_33[2,2], etc.
        B_blocks = []
        for md in market_datas:
            B_blocks.append(np.column_stack([md.z1, md.z2, np.ones(md.J)]))
        B = np.vstack(B_blocks)  # (total_J, 3)
        BtB = B.T @ B            # (3, 3)
        W3 = np.linalg.inv(BtB)  # (3, 3)
        # Map to (4,4): moment indices [m1→z1, m2→z2, m3→1, m4→1]
        W = np.zeros((4, 4), dtype=np.float64)
        W[:2, :2] = W3[:2, :2]          # z1, z2 block
        W[:2, 2] = W3[:2, 2]            # z1/z2 cross with constant
        W[:2, 3] = W3[:2, 2]            # same cross (m4 also uses constant)
        W[2, :2] = W3[2, :2]
        W[3, :2] = W3[2, :2]
        W[2, 2] = W3[2, 2]
        W[2, 3] = W3[2, 2]
        W[3, 2] = W3[2, 2]
        W[3, 3] = W3[2, 2]
    else:
        B_blocks = []
        for md in market_datas:
            B_blocks.append(np.column_stack([md.z1, md.z2, md.z3, np.ones(md.J)]))
        B = np.vstack(B_blocks)  # (total_J, 4)
        BtB = B.T @ B            # (4, 4)
        W = np.linalg.inv(BtB)

    return W


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def partition_markets(
    market_datas: List[HybridMarketData], size: int,
) -> List[List[HybridMarketData]]:
    """Round-robin partition of M markets into `size` chunks."""
    partitions: List[List[HybridMarketData]] = [[] for _ in range(size)]
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

    args = parse_args()  # all ranks parse (same argv)
    out_dir = Path(args.out_dir)

    total_start = time.perf_counter()

    # ---- Rank 0 loads data and partitions ----
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Distributed hybrid AugLag: {size} ranks")
        print("Loading multi-market data...")

        market_datas, meta = load_hybrid_market_datas(
            args.firms_path, args.workers_path, args.params_path,
            M_subset=args.M,
        )
        tp = meta['true_params']
        M = meta['M']
        total_J = meta['total_J']
        J_per_list = meta['J_per_list']

        J_min, J_max = min(J_per_list), max(J_per_list)
        j_str = str(J_min) if J_min == J_max else f"{J_min}-{J_max}"
        print(f"  M={M}, J_per={j_str}, total_J={total_J}")
        print(f"  True params: tau={tp['tau']}, alpha={tp['alpha']}, "
              f"gamma0={tp['gamma0']}, sigma_e={tp['sigma_e']}, "
              f"eta={tp['eta']}, tilde_gamma={tp['tilde_gamma']:.4f}")

        # Compute W
        if args.W_path:
            print(f"Loading W from {args.W_path}")
            with open(args.W_path) as f:
                W_data = json.load(f)
            W = np.array(W_data["W"], dtype=np.float64)
        else:
            W = compute_2sls_W(market_datas)
            print(f"  W (2SLS) = diag({np.diag(W).tolist()})")

        partitions = partition_markets(market_datas, size)
        sizes_str = ", ".join(str(len(p)) for p in partitions[:5])
        print(f"  Partitioned into {size} ranks [{sizes_str}, ...]")
    else:
        partitions = None
        meta = None
        W = None

    # ---- Scatter market partitions (data sent once) ----
    local_markets: List[HybridMarketData] = comm.scatter(partitions, root=0)
    meta = comm.bcast(meta, root=0)
    W = comm.bcast(W, root=0)

    tp = meta['true_params']
    M = meta['M']
    total_J = meta['total_J']

    if rank == 0:
        data_per_rank = sum(
            md.D.nbytes + md.v.nbytes + md.choice_idx.nbytes
            + md.w.nbytes + md.R.nbytes + md.L.nbytes
            for md in local_markets) / 1e6
        print(f"  Rank 0 holds {len(local_markets)} markets "
              f"({data_per_rank:.1f} MB)")

    # ---- Initialization ----
    if args.true_init:
        if rank == 0:
            print("Initializing from true values...")
            eta = tp['eta']
            gamma0 = tp['gamma0']
            sigma_e = tp['sigma_e']

            all_states: List[List[HybridMarketState]] = [[] for _ in range(size)]
            for i, md in enumerate(market_datas):
                mf = meta['meta_firms'][i]
                delta_true = eta * np.log(np.maximum(mf['w'], 1e-300)) + mf['xi']
                ln_qbar_true = np.log(np.maximum(mf['qbar'], 1e-300))
                tq_true = (ln_qbar_true - gamma0) / sigma_e
                state = HybridMarketState(
                    delta=delta_true.astype(np.float64),
                    tilde_q=tq_true.astype(np.float64),
                    g_m=np.zeros(4, dtype=np.float64),
                    nll=0.0,
                )
                all_states[i % size].append(state)
        else:
            all_states = None

        local_states: List[HybridMarketState] = comm.scatter(all_states, root=0)

        theta_G_init = np.array([
            tp['tau'], tp['tilde_gamma'], tp['alpha'],
            tp['sigma_e'], tp['eta'], tp['gamma0'],
        ], dtype=np.float64)

    else:
        # Pooled naive init on rank 0 pre-scatter
        if rank == 0:
            print("Computing pooled naive initial guesses...")
            from screening.analysis.lib.naive_init import compute_pooled_naive_init
            init_start = time.perf_counter()
            theta_G_init, all_deltas, all_tilde_qs = compute_pooled_naive_init(market_datas)
            init_time = time.perf_counter() - init_start
            print(f"  theta_G_init = {theta_G_init}  ({init_time:.1f}s)")

            # Build states and partition for scatter
            all_states: List[List[HybridMarketState]] = [[] for _ in range(size)]
            for i, md in enumerate(market_datas):
                state = HybridMarketState(
                    delta=all_deltas[i],
                    tilde_q=all_tilde_qs[i],
                    g_m=np.zeros(4, dtype=np.float64),
                    nll=0.0,
                )
                all_states[i % size].append(state)
        else:
            all_states = None
            theta_G_init = None

        local_states: List[HybridMarketState] = comm.scatter(all_states, root=0)
        theta_G_init = comm.bcast(theta_G_init, root=0)

    # Broadcast theta_G_init so all ranks have the same value
    theta_G_init = comm.bcast(theta_G_init, root=0)

    if rank == 0:
        print(f"  theta_G_init = {theta_G_init}")
        print(f"\nStarting augmented-Lagrangian solver "
              f"(M={M}, P={size})...")

    build_time = time.perf_counter() - total_start
    solve_start = time.perf_counter()

    # ---- Run outer loop (SPMD — all ranks) ----
    result = run_auglag_distributed(
        comm, local_markets, local_states, theta_G_init, W,
        max_outer_iter=args.max_outer_iter,
        inner_maxiter=args.inner_maxiter,
        inner_tol=args.inner_tol,
        global_maxiter=args.global_maxiter,
        global_tol=args.global_tol,
        verbose=(rank == 0),
    )

    solve_time = time.perf_counter() - solve_start
    total_time = time.perf_counter() - total_start

    # ---- Gather final states from all ranks ----
    final_local = result.get('local_states', local_states)
    all_local_pairs = comm.gather(
        [(md.market_id, st.delta, st.tilde_q, st.g_m, st.nll)
         for md, st in zip(local_markets, final_local)],
        root=0,
    )

    # ---- Rank 0: output ----
    if rank == 0:
        tG = result['theta_G']

        # Flatten and sort by market_id
        all_pairs = [pair for chunk in all_local_pairs for pair in chunk]
        all_pairs.sort(key=lambda p: p[0])

        deltas = [p[1] for p in all_pairs]
        tilde_qs = [p[2] for p in all_pairs]
        g_ms = [p[3] for p in all_pairs]

        print(f"\n{'='*60}")
        print(f"Distributed Augmented-Lagrangian Hybrid MLE+GMM")
        print(f"  MPI ranks: {size}")
        print(f"  Converged: {result['converged']}")
        print(f"  Outer iterations: {result['n_outer_iters']}")
        print(f"  Final objective: {result['obj']:.4f}")
        print(f"  Final NLL: {result['total_nll']:.4f}")
        print(f"  |g_bar|: {np.linalg.norm(result['g_bar']):.4e}")
        print(f"  Build: {build_time:.2f}s  Solve: {solve_time:.2f}s  "
              f"Total: {total_time:.2f}s")
        print(f"{'='*60}")

        names = ['tau', 'tilde_gamma', 'alpha', 'sigma_e', 'eta', 'gamma0']
        true_vals = [tp['tau'], tp['tilde_gamma'], tp['alpha'],
                     tp['sigma_e'], tp['eta'], tp['gamma0']]
        print("\n=== Parameter Recovery ===")
        for name, hat, true in zip(names, tG, true_vals):
            print(f"  {name:14s}: hat={hat:.6f}  true={true:.6f}  "
                  f"err={hat-true:+.6f}")

        # Delta and tilde_q correlation
        for i, mid_pair in enumerate(all_pairs):
            mid = mid_pair[0]
            mf = meta['meta_firms'][i]
            delta_true = tp['eta'] * np.log(np.maximum(mf['w'], 1e-300)) + mf['xi']
            ln_qbar_true = np.log(np.maximum(mf['qbar'], 1e-300))
            tq_true = (ln_qbar_true - tp['gamma0']) / tp['sigma_e']

            d_corr = float(np.corrcoef(deltas[i], delta_true)[0, 1])
            tq_corr = float(np.corrcoef(tilde_qs[i], tq_true)[0, 1])
            d_rmse = float(np.sqrt(np.mean((deltas[i] - delta_true) ** 2)))
            tq_rmse = float(np.sqrt(np.mean((tilde_qs[i] - tq_true) ** 2)))
            print(f"  Market {mid}: delta_corr={d_corr:.4f} rmse={d_rmse:.4f}  "
                  f"tq_corr={tq_corr:.4f} rmse={tq_rmse:.4f}")

        print(f"\n  g_bar = {result['g_bar']}")

        # Save results
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "hybrid_auglag_distributed_estimates.json"

        out = {
            "solver": "distributed_auglag_hybrid",
            "M": M, "total_J": total_J,
            "mpi_ranks": size,
            "converged": bool(result['converged']),
            "n_outer_iters": int(result['n_outer_iters']),
            "objective": float(result['obj']),
            "nll": float(result['total_nll']),
            "theta_G": tG.tolist(),
            "g_bar": result['g_bar'].tolist(),
            "mu": result['mu'].tolist(),
            "nu": result['nu'].tolist(),
            "rho": float(result['rho']),
            "wall_sec": total_time,
            "timings": {
                "build_time_sec": build_time,
                "solve_time_sec": solve_time,
                "total_time_sec": total_time,
            },
            "true_params": tp,
            "W": W.tolist(),
            "history": result['history'],
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {out_path}")

        # Per-market inner estimates CSV
        inner_rows = []
        for i, pair in enumerate(all_pairs):
            mid = pair[0]
            for j in range(len(deltas[i])):
                inner_rows.append({
                    "market_id": mid,
                    "firm_j": j + 1,
                    "delta_hat": float(deltas[i][j]),
                    "tilde_q_hat": float(tilde_qs[i][j]),
                })
        inner_df = pd.DataFrame(inner_rows)
        inner_path = out_dir / "hybrid_auglag_distributed_inner_estimates.csv"
        inner_df.to_csv(inner_path, index=False)
        print(f"Inner estimates ({len(inner_rows)} firm×market) written to {inner_path}")

        # Optimal W for two-step GMM
        g_stack = np.array([gm for gm in g_ms])  # (M, 4)
        S_hat = g_stack.T @ g_stack / M
        try:
            W_opt = np.linalg.inv(S_hat)
        except np.linalg.LinAlgError:
            W_opt = None
        if W_opt is not None:
            w_opt_path = out_dir / "hybrid_auglag_distributed_W_optimal.json"
            with open(w_opt_path, "w") as f:
                json.dump({"W": W_opt.tolist(), "S_hat": S_hat.tolist()}, f, indent=2)
            print(f"Optimal W for two-step GMM written to {w_opt_path}")

        # Run log
        import datetime
        log_path = out_dir / "hybrid_auglag_distributed_run_log.csv"
        log_row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "M": M, "total_J": total_J, "mpi_ranks": size,
            "max_outer_iter": args.max_outer_iter,
            "n_outer_iters": result['n_outer_iters'],
            "converged": result['converged'],
            "nll": f"{result['total_nll']:.4f}",
            "obj": f"{result['obj']:.4f}",
            "g_bar_norm": f"{np.linalg.norm(result['g_bar']):.4e}",
            "tau_hat": f"{tG[0]:.6f}",
            "tau_true": f"{tp['tau']:.6f}",
            "alpha_hat": f"{tG[2]:.6f}",
            "alpha_true": f"{tp['alpha']:.6f}",
            "build_time_s": f"{build_time:.2f}",
            "solve_time_s": f"{solve_time:.2f}",
            "total_time_s": f"{total_time:.2f}",
        }
        log_df = pd.DataFrame([log_row])
        write_header = not log_path.exists()
        log_df.to_csv(log_path, mode="a", header=write_header, index=False)
        print(f"Run logged to {log_path}")

    MPI.Finalize()


if __name__ == "__main__":
    main()
