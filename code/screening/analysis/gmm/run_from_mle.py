#!/usr/bin/env python3
"""Standalone GMM: recover [alpha, sigma_e, eta, gamma0] from MLE inner estimates.

Takes the distributed MLE outputs (tau_hat, tilde_gamma_hat, delta_hat, tilde_q_hat)
as fixed and minimizes the quadratic GMM objective over only the 4 macro parameters.

Usage::

    cd code/
    SCREENING_DATA_DIR=../data python -m screening.analysis.gmm.run_from_mle \
        --mle_results ../data_v2/scaling_results/mle/mle_distributed_estimates.json \
        --inner_estimates ../data_v2/scaling_results/mle/mle_distributed_inner_estimates.csv \
        --M 50
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
from typing import List, Tuple

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import minimize as sp_minimize
from scipy.spatial.distance import cdist

try:
    from ... import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_ESTIMATION
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from screening import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_ESTIMATION

from screening.analysis.lib.model_components import compute_tilde_Q_M, compute_gmm_moments_M
from screening.analysis.lib.helpers import read_parameters


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    raw_dir = get_data_subdir(DATA_RAW, create=True)
    clean_dir = get_data_subdir(DATA_CLEAN, create=True)
    build_dir = get_data_subdir(DATA_BUILD, create=True)
    est_dir = get_output_subdir(OUTPUT_ESTIMATION, create=True)

    p = argparse.ArgumentParser(
        description="Standalone GMM: recover [alpha, sigma_e, eta, gamma0] from MLE inner estimates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--firms_path", type=str,
                   default=str(clean_dir / "equilibrium_firms.csv"))
    p.add_argument("--workers_path", type=str,
                   default=str(build_dir / "workers_dataset.csv"))
    p.add_argument("--params_path", type=str,
                   default=str(raw_dir / "parameters_effective.csv"))
    p.add_argument("--mle_results", type=str, required=True,
                   help="Path to mle_distributed_estimates.json")
    p.add_argument("--inner_estimates", type=str, required=True,
                   help="Path to mle_distributed_inner_estimates.csv")
    p.add_argument("--M", type=int, default=None,
                   help="Use only the first M markets (default: all)")
    p.add_argument("--true_init", action="store_true",
                   help="Initialize from true parameter values")
    p.add_argument("--maxiter", type=int, default=200,
                   help="L-BFGS-B max iterations")
    p.add_argument("--out_dir", type=str, default=str(est_dir))
    p.add_argument("--identity_W", action="store_true",
                   help="Use W = I (identity) instead of 2SLS weighting")
    p.add_argument("--demean_z", action="store_true",
                   help="Demean z1, z2 within each market (equiv. to intercept)")
    p.add_argument("--true_delta", action="store_true",
                   help="Use true delta/tilde_q instead of MLE inner estimates")
    p.add_argument("--fix_gamma0", type=float, default=None,
                   help="Fix gamma0 at this value and drop m4; search only [alpha, sigma_e, eta]")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_market_data_with_inner(
    firms_path: str,
    workers_path: str,
    params_path: str,
    inner_df: pd.DataFrame | None,
    M_subset: int | None = None,
    use_true_delta: bool = False,
) -> Tuple[list, dict]:
    """Load market data and attach delta/tilde_q estimates.

    If use_true_delta=True, computes delta_true and tilde_q_true from structural
    parameters and firm-level data (xi, qbar). Otherwise uses MLE inner estimates.
    """
    firms_df = pd.read_csv(firms_path)
    workers_df = pd.read_csv(workers_path)
    params = read_parameters(params_path)

    gamma0_true = float(params.get("gamma0", 0.0))
    gamma1_true = float(params.get("gamma1", params.get("gamma", 0.94)))
    sigma_e_true = float(params.get("sigma_e", params.get("sigma_a", 0.135)))
    alpha_true = float(params.get("alpha", params.get("beta", 0.2)))
    eta_true = float(params.get("eta", 7.0))
    tau_true = float(params.get("tau", 0.4))
    tilde_gamma_true = gamma1_true / sigma_e_true if sigma_e_true > 0 else 0.0

    true_params = {
        "tau": tau_true, "alpha": alpha_true, "gamma0": gamma0_true,
        "gamma1": gamma1_true, "sigma_e": sigma_e_true, "eta": eta_true,
        "tilde_gamma": tilde_gamma_true,
    }

    firms_df = firms_df.sort_values(["market_id", "firm_id"]).reset_index(drop=True)
    workers_df = workers_df.sort_values(["market_id"]).reset_index(drop=True)

    all_markets = sorted(firms_df["market_id"].unique())
    if not use_true_delta and inner_df is not None:
        inner_markets = sorted(inner_df["market_id"].unique())
        markets = sorted(set(all_markets) & set(inner_markets))
    else:
        markets = all_markets
    if M_subset is not None:
        markets = markets[:M_subset]

    firms_df = firms_df[firms_df["market_id"].isin(markets)].reset_index(drop=True)
    workers_df = workers_df[workers_df["market_id"].isin(markets)].reset_index(drop=True)
    if inner_df is not None:
        inner_sub = inner_df[inner_df["market_id"].isin(markets)]
    else:
        inner_sub = None

    total_J = sum(len(firms_df[firms_df["market_id"] == m]) for m in markets)

    has_z1 = "z1" in firms_df.columns
    has_z2 = "z2" in firms_df.columns
    has_xi = "xi" in firms_df.columns
    qbar_col = "qbar" if "qbar" in firms_df.columns else "c"

    market_list = []
    J_per_list = []

    for mid in markets:
        fdf = firms_df[firms_df["market_id"] == mid].sort_values("firm_id")
        wdf = workers_df[workers_df["market_id"] == mid]

        J_m = len(fdf)
        N_m = len(wdf)

        if use_true_delta:
            w_arr = fdf["w"].values.astype(np.float64)
            xi_arr = fdf["xi"].values.astype(np.float64)
            qbar_arr = fdf[qbar_col].values.astype(np.float64)
            delta_hat = eta_true * np.log(np.maximum(w_arr, 1e-300)) + xi_arr
            tilde_q_hat = (np.log(np.maximum(qbar_arr, 1e-300)) - gamma0_true) / sigma_e_true
        else:
            idf = inner_sub[inner_sub["market_id"] == mid].sort_values("firm_j")
            delta_hat = idf["delta_hat"].values.astype(np.float64)
            tilde_q_hat = idf["tilde_q_hat"].values.astype(np.float64)
            if len(delta_hat) != J_m:
                raise ValueError(
                    f"Market {mid}: inner estimates have {len(delta_hat)} firms "
                    f"but data has {J_m}"
                )

        v = wdf["x_skill"].values.astype(np.float64)
        choice_idx = wdf["chosen_firm"].values.astype(np.int32)
        w = fdf["w"].values.astype(np.float64)
        R = fdf["Y"].values.astype(np.float64)

        loc_firms = fdf[["x", "y"]].values
        worker_locs = np.column_stack([wdf["ell_x"].values, wdf["ell_y"].values])
        D = cdist(worker_locs, loc_firms, metric="euclidean").astype(np.float64)

        counts = np.bincount(choice_idx, minlength=J_m + 1).astype(np.float64)
        L = counts[1:]

        z1 = fdf["z1"].values.astype(np.float64) if has_z1 else np.zeros(J_m, dtype=np.float64)
        z2 = fdf["z2"].values.astype(np.float64) if has_z2 else np.zeros(J_m, dtype=np.float64)
        z3 = np.ones(J_m, dtype=np.float64)
        xi = fdf["xi"].values.astype(np.float64) if has_xi else np.zeros(J_m, dtype=np.float64)

        # v_bar_j = average observable skill among workers matched to firm j
        v_bar = np.zeros(J_m, dtype=np.float64)
        for j in range(J_m):
            mask_j = choice_idx == (j + 1)
            if np.any(mask_j):
                v_bar[j] = np.mean(v[mask_j])
            else:
                v_bar[j] = np.mean(v)

        omega = J_m / total_J

        market_list.append({
            "market_id": int(mid), "J": J_m, "N": N_m,
            "delta_hat": delta_hat, "tilde_q_hat": tilde_q_hat,
            "v": v, "choice_idx": choice_idx, "D": D,
            "w": w, "R": R, "L": L,
            "z1": z1, "z2": z2, "z3": z3,
            "xi": xi, "v_bar": v_bar,
            "omega": omega,
        })
        J_per_list.append(J_m)

    meta = {
        "true_params": true_params,
        "M": len(markets),
        "total_J": total_J,
        "J_per_list": J_per_list,
    }
    return market_list, meta


# ---------------------------------------------------------------------------
# Weighting matrix (reused from run_distributed_hybrid_auglag.py)
# ---------------------------------------------------------------------------


def compute_2sls_W(market_list: list, n_moments: int = 4) -> np.ndarray:
    """W = (B'B)^{-1} where B is the instrument matrix.

    If n_moments=3, build a 3x3 W for moments [m1, m2, m3] only (no m4).
    """
    z3_is_const = all(np.allclose(md["z3"], 1.0) for md in market_list)

    if n_moments == 3:
        # 3 moments → 3 instruments: z1, z2, z3(=ones)
        B_blocks = []
        for md in market_list:
            B_blocks.append(np.column_stack([md["z1"], md["z2"], np.ones(md["J"])]))
        B = np.vstack(B_blocks)
        BtB = B.T @ B
        return np.linalg.inv(BtB)

    if z3_is_const:
        B_blocks = []
        for md in market_list:
            B_blocks.append(np.column_stack([md["z1"], md["z2"], np.ones(md["J"])]))
        B = np.vstack(B_blocks)
        BtB = B.T @ B
        W3 = np.linalg.inv(BtB)
        W = np.zeros((4, 4), dtype=np.float64)
        W[:2, :2] = W3[:2, :2]
        W[:2, 2] = W3[:2, 2]
        W[:2, 3] = W3[:2, 2]
        W[2, :2] = W3[2, :2]
        W[3, :2] = W3[2, :2]
        W[2, 2] = W3[2, 2]
        W[2, 3] = W3[2, 2]
        W[3, 2] = W3[2, 2]
        W[3, 3] = W3[2, 2]
    else:
        B_blocks = []
        for md in market_list:
            B_blocks.append(np.column_stack([md["z1"], md["z2"], md["z3"], np.ones(md["J"])]))
        B = np.vstack(B_blocks)
        BtB = B.T @ B
        W = np.linalg.inv(BtB)

    return W


# ---------------------------------------------------------------------------
# Naive initialization for [alpha, sigma_e, eta, gamma0]
# ---------------------------------------------------------------------------


def naive_init_gmm_params(market_list: list) -> np.ndarray:
    """Data-driven starting values using pooled naive init.

    Converts market dicts to HybridMarketData for compute_pooled_naive_init.
    """
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _HMD:
        market_id: int
        J: int
        N: int
        v: np.ndarray
        choice_idx: np.ndarray
        D: np.ndarray
        w: np.ndarray
        R: np.ndarray
        L: np.ndarray
        z1: np.ndarray
        z2: np.ndarray
        z3: np.ndarray
        omega: float

    hybrid_mds = []
    for md in market_list:
        hybrid_mds.append(_HMD(
            market_id=md["market_id"], J=md["J"], N=md["N"],
            v=md["v"], choice_idx=md["choice_idx"], D=md["D"],
            w=md["w"], R=md["R"], L=md["L"],
            z1=md["z1"], z2=md["z2"], z3=md["z3"],
            omega=md["omega"],
        ))

    from screening.analysis.lib.naive_init import compute_pooled_naive_init
    theta_G_full, _, _ = compute_pooled_naive_init(hybrid_mds)
    # theta_G_full = [tau, tilde_gamma, alpha, sigma_e, eta, gamma0]
    return np.array([theta_G_full[2], theta_G_full[3], theta_G_full[4], theta_G_full[5]],
                    dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    # 1. Load MLE globals
    print("Loading MLE results...")
    with open(args.mle_results) as f:
        mle_json = json.load(f)
    if args.true_delta:
        # Use true tau, tilde_gamma when using true deltas
        params_tmp = read_parameters(args.params_path)
        tau_fixed = float(params_tmp.get("tau", 0.4))
        gamma1_tmp = float(params_tmp.get("gamma1", params_tmp.get("gamma", 0.94)))
        sigma_e_tmp = float(params_tmp.get("sigma_e", params_tmp.get("sigma_a", 0.135)))
        tg_fixed = gamma1_tmp / sigma_e_tmp
        print(f"  Fixed (true): tau={tau_fixed:.6f}, tilde_gamma={tg_fixed:.6f}")
    else:
        tau_fixed = float(mle_json["theta_G"][0])
        tg_fixed = float(mle_json["theta_G"][1])
        print(f"  Fixed from MLE: tau={tau_fixed:.6f}, tilde_gamma={tg_fixed:.6f}")

    # 2. Load inner estimates (if needed)
    if args.true_delta:
        inner_df = None
        print("  Using TRUE delta and tilde_q from structural parameters")
    else:
        inner_df = pd.read_csv(args.inner_estimates)
        print(f"  Inner estimates: {len(inner_df)} rows")

    # 3. Load market data and merge with inner estimates
    print("Loading market data...")
    market_list, meta = load_market_data_with_inner(
        args.firms_path, args.workers_path, args.params_path,
        inner_df, M_subset=args.M, use_true_delta=args.true_delta,
    )
    tp = meta["true_params"]
    M = meta["M"]
    total_J = meta["total_J"]
    J_per_list = meta["J_per_list"]
    J_min, J_max = min(J_per_list), max(J_per_list)
    j_str = str(J_min) if J_min == J_max else f"{J_min}-{J_max}"
    print(f"  M={M}, J_per={j_str}, total_J={total_J}")
    print(f"  True: alpha={tp['alpha']}, sigma_e={tp['sigma_e']}, "
          f"eta={tp['eta']}, gamma0={tp['gamma0']}")

    # 3b. Demean instruments if requested
    if args.demean_z:
        print("  Demeaning z1, z2 within each market")
        for md in market_list:
            md["z1"] = md["z1"] - np.mean(md["z1"])
            md["z2"] = md["z2"] - np.mean(md["z2"])

    # Determine dimensionality
    fix_gamma0 = args.fix_gamma0
    if fix_gamma0 is not None:
        n_moments = 3
        n_params = 3
        print(f"  Fixing gamma0 = {fix_gamma0:.4f}, using 3 moments (m1, m2, m3)")
    else:
        n_moments = 4
        n_params = 4

    # 4. Compute W matrix
    if args.identity_W:
        W_np = np.eye(n_moments, dtype=np.float64)
        print(f"  W = I (identity, {n_moments}x{n_moments})")
    else:
        W_np = compute_2sls_W(market_list, n_moments=n_moments)
        print(f"  W (2SLS) diag = {np.diag(W_np).tolist()}")

    # 5. Initial values
    if fix_gamma0 is not None:
        if args.true_init:
            theta0 = np.array([tp["alpha"], tp["sigma_e"], tp["eta"]],
                              dtype=np.float64)
            print(f"  Init (true): {theta0}")
        else:
            print("  Computing naive initialization...")
            theta0_full = naive_init_gmm_params(market_list)
            theta0 = theta0_full[:3]  # [alpha, sigma_e, eta]
            print(f"  Init (naive): {theta0}")
    else:
        if args.true_init:
            theta0 = np.array([tp["alpha"], tp["sigma_e"], tp["eta"], tp["gamma0"]],
                              dtype=np.float64)
            print(f"  Init (true): {theta0}")
        else:
            print("  Computing naive initialization...")
            theta0 = naive_init_gmm_params(market_list)
            print(f"  Init (naive): {theta0}")

    # 6. Convert market arrays to JAX (once)
    tau_j = jnp.float64(tau_fixed)
    tg_j = jnp.float64(tg_fixed)

    jax_markets = []
    for md in market_list:
        jax_markets.append({
            "delta": jnp.array(md["delta_hat"]),
            "tq": jnp.array(md["tilde_q_hat"]),
            "v": jnp.array(md["v"]),
            "D": jnp.array(md["D"]),
            "cidx": jnp.array(md["choice_idx"]),
            "w": jnp.array(md["w"]),
            "R": jnp.array(md["R"]),
            "L": jnp.array(md["L"]),
            "z1": jnp.array(md["z1"]),
            "z2": jnp.array(md["z2"]),
            "z3": jnp.array(md["z3"]),
            "omega": md["omega"],
        })

    # 7. JIT per-market moments function
    if fix_gamma0 is not None:
        gamma0_fixed_j = jnp.float64(fix_gamma0)

        @jax.jit
        def _market_g(theta_gmm, tau, tg, delta, tq, v, D, cidx, w, R, L, z1, z2, z3):
            alpha, sigma_e, eta = theta_gmm[0], theta_gmm[1], theta_gmm[2]
            tilde_Q = compute_tilde_Q_M(sigma_e, tg, tau, delta, tq, v, D, cidx)
            g4 = compute_gmm_moments_M(delta, tq, tilde_Q, gamma0_fixed_j, sigma_e,
                                        alpha, eta, w, R, L, z1, z2, z3)
            return g4[:3]  # drop m4
    else:
        @jax.jit
        def _market_g(theta_gmm, tau, tg, delta, tq, v, D, cidx, w, R, L, z1, z2, z3):
            alpha, sigma_e, eta, gamma0 = theta_gmm[0], theta_gmm[1], theta_gmm[2], theta_gmm[3]
            tilde_Q = compute_tilde_Q_M(sigma_e, tg, tau, delta, tq, v, D, cidx)
            return compute_gmm_moments_M(delta, tq, tilde_Q, gamma0, sigma_e,
                                         alpha, eta, w, R, L, z1, z2, z3)

    _market_g_jac = jax.jit(jax.jacobian(_market_g, argnums=0))

    # 8. Objective + gradient
    n_evals = [0]
    if fix_gamma0 is not None:
        param_names = ["alpha", "sigma_e", "eta"]
    else:
        param_names = ["alpha", "sigma_e", "eta", "gamma0"]

    def val_and_grad(theta_np):
        theta_jax = jnp.array(theta_np)
        m_bar = np.zeros(n_moments, dtype=np.float64)
        J_bar = np.zeros((n_moments, n_params), dtype=np.float64)

        for jm in jax_markets:
            g_m = _market_g(theta_jax, tau_j, tg_j,
                            jm["delta"], jm["tq"], jm["v"], jm["D"], jm["cidx"],
                            jm["w"], jm["R"], jm["L"], jm["z1"], jm["z2"], jm["z3"])
            jac_m = _market_g_jac(theta_jax, tau_j, tg_j,
                                  jm["delta"], jm["tq"], jm["v"], jm["D"], jm["cidx"],
                                  jm["w"], jm["R"], jm["L"], jm["z1"], jm["z2"], jm["z3"])
            m_bar += jm["omega"] * np.asarray(g_m)
            J_bar += jm["omega"] * np.asarray(jac_m)

        val = 0.5 * m_bar @ W_np @ m_bar
        grad = J_bar.T @ W_np @ m_bar

        n_evals[0] += 1
        if n_evals[0] % 5 == 1:
            print(f"  eval {n_evals[0]:4d}: Q={val:.6e}  "
                  f"|m|={np.linalg.norm(m_bar):.4e}  "
                  f"theta=[{', '.join(f'{x:.4f}' for x in theta_np)}]")

        return float(val), grad.astype(np.float64)

    # 9. Evaluate at initial point
    print("\n--- Evaluating at initial point ---")
    val0, grad0 = val_and_grad(theta0)
    print(f"  Q(theta0) = {val0:.6e}, |grad| = {np.linalg.norm(grad0):.4e}")

    # 10. Optimize
    if fix_gamma0 is not None:
        bounds = [
            (0.01, 0.99),    # alpha
            (0.01, 2.0),     # sigma_e
            (0.1, 50.0),     # eta
        ]
    else:
        bounds = [
            (0.01, 0.99),    # alpha
            (0.01, 2.0),     # sigma_e
            (0.1, 50.0),     # eta
            (-5.0, 5.0),     # gamma0
        ]

    print(f"\n--- Starting L-BFGS-B (maxiter={args.maxiter}) ---")
    n_evals[0] = 0
    opt_start = time.perf_counter()

    result = sp_minimize(
        val_and_grad, theta0, method="L-BFGS-B", jac=True,
        bounds=bounds,
        options={"maxiter": args.maxiter, "ftol": 1e-12, "gtol": 1e-6},
    )

    opt_time = time.perf_counter() - opt_start
    total_time = time.perf_counter() - total_start

    theta_hat = result.x
    if fix_gamma0 is not None:
        alpha_hat, sigma_e_hat, eta_hat = theta_hat
        gamma0_hat = fix_gamma0
    else:
        alpha_hat, sigma_e_hat, eta_hat, gamma0_hat = theta_hat

    # 11. Final moments
    theta_jax_final = jnp.array(theta_hat)
    m_bar_final = np.zeros(n_moments, dtype=np.float64)
    for jm in jax_markets:
        g_m = _market_g(theta_jax_final, tau_j, tg_j,
                        jm["delta"], jm["tq"], jm["v"], jm["D"], jm["cidx"],
                        jm["w"], jm["R"], jm["L"], jm["z1"], jm["z2"], jm["z3"])
        m_bar_final += jm["omega"] * np.asarray(g_m)

    Q_final = 0.5 * m_bar_final @ W_np @ m_bar_final

    # 12. Report
    print(f"\n{'='*60}")
    print(f"GMM from MLE — Results")
    print(f"  M={M}, total_J={total_J}")
    print(f"  Fixed: tau={tau_fixed:.6f}, tilde_gamma={tg_fixed:.6f}")
    if fix_gamma0 is not None:
        print(f"  Fixed: gamma0={fix_gamma0:.6f}")
    print(f"  Converged: {result.success} ({result.message})")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evals: {result.nfev}")
    print(f"  Final Q: {Q_final:.6e}")
    print(f"  Opt time: {opt_time:.1f}s  Total: {total_time:.1f}s")
    print(f"{'='*60}")

    if fix_gamma0 is not None:
        true_vals = [tp["alpha"], tp["sigma_e"], tp["eta"]]
    else:
        true_vals = [tp["alpha"], tp["sigma_e"], tp["eta"], tp["gamma0"]]
    print("\n=== Parameter Recovery ===")
    print(f"  {'param':12s} {'hat':>12s} {'true':>12s} {'error':>12s}")
    print(f"  {'-'*48}")
    for name, hat, true in zip(param_names, theta_hat, true_vals):
        print(f"  {name:12s} {hat:12.6f} {true:12.6f} {hat - true:+12.6f}")
    if fix_gamma0 is not None:
        print(f"  {'gamma0':12s} {fix_gamma0:12.6f} {tp['gamma0']:12.6f} "
              f"{fix_gamma0 - tp['gamma0']:+12.6f}  (FIXED)")

    print(f"\n  Moments at solution: {m_bar_final}")
    print(f"  |m_bar| = {np.linalg.norm(m_bar_final):.6e}")

    # 13. Save
    out_path = out_dir / "gmm_from_mle_results.json"
    theta_hat_dict = {
        "alpha": float(alpha_hat),
        "sigma_e": float(sigma_e_hat),
        "eta": float(eta_hat),
        "gamma0": float(gamma0_hat),
    }
    out_data = {
        "solver": "gmm_from_mle_lbfgsb",
        "M": M, "total_J": total_J,
        "fixed_from_mle": {
            "tau": tau_fixed,
            "tilde_gamma": tg_fixed,
        },
        "theta_hat": theta_hat_dict,
        "true_params": tp,
        "init_values": theta0.tolist(),
        "init_mode": "true" if args.true_init else "naive",
        "fix_gamma0": fix_gamma0,
        "objective": float(Q_final),
        "moments_at_solution": m_bar_final.tolist(),
        "converged": bool(result.success),
        "message": str(result.message),
        "n_iter": int(result.nit),
        "n_fev": int(result.nfev),
        "W": W_np.tolist(),
        "recovery": {
            name: {"hat": float(hat), "true": float(true), "err": float(hat - true)}
            for name, hat, true in zip(param_names, theta_hat, true_vals)
        },
        "timings": {
            "opt_time_sec": opt_time,
            "total_time_sec": total_time,
        },
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
