#!/usr/bin/env python3
"""Single-process augmented-Lagrangian hybrid MLE+GMM solver.

Implements estimation.tex §6.2: combines micro NLL with macro GMM penalty
on 4 firm-level moments via an augmented-Lagrangian decomposition.

Usage::

    cd code
    python -m screening.analysis.auglag.run_pooled [options]

See estimation.tex eqs (6.1)-(6.12) for the algorithm specification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# JAX init
# ---------------------------------------------------------------------------

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                      "intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from screening.analysis.lib.model_components import (
    choice_probabilities,
    per_obs_nll,
    compute_tilde_Q_M,
    compute_gmm_moments_M,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridMarketData:
    """Per-market data bundle for the hybrid estimator."""
    market_id: int
    J: int
    N: int
    v: jnp.ndarray           # (N,) observable skill
    choice_idx: jnp.ndarray  # (N,) int32
    D: jnp.ndarray           # (N, J) distances
    w: jnp.ndarray           # (J,) wages
    R: jnp.ndarray           # (J,) revenue (= Y column)
    L: jnp.ndarray           # (J,) labor counts
    z1: jnp.ndarray          # (J,) TFP instrument
    z2: jnp.ndarray          # (J,) amenity instrument
    z3: jnp.ndarray          # (J,) instrument for m3 (default: ones)
    omega: float             # J_m / sum(J_j) market weight


@dataclass
class AugLagState:
    """Mutable state for the augmented-Lagrangian outer loop."""
    theta_G: np.ndarray       # (6,) [tau, tg, alpha, sigma_e, eta, gamma0]
    deltas: List[np.ndarray]  # per-market delta arrays
    tilde_qs: List[np.ndarray]  # per-market tilde_q arrays
    mu: np.ndarray            # (4,)
    nu: np.ndarray            # (4,)
    rho: float
    g_ms: List[np.ndarray]    # per-market (4,) moment vectors
    g_bar: np.ndarray         # (4,) aggregate moment
    iteration: int = 0


# ---------------------------------------------------------------------------
# JIT cache
# ---------------------------------------------------------------------------

_jit_cache: Dict[int, dict] = {}


def _get_jitted_fns(J: int) -> dict:
    """Return JIT'd functions for the given J, creating if needed."""
    if J in _jit_cache:
        return _jit_cache[J]

    # --- inner value+grad for market subproblem ---
    def _inner_obj(z_m, tau, tilde_gamma, alpha, sigma_e, eta, gamma0,
                   nu_vec, rho, c_m, omega,
                   v, choice_idx, D, w, R, L, z1, z2, z3):
        delta = z_m[:J]
        tilde_q = z_m[J:]
        P = choice_probabilities(tau, tilde_gamma, delta, tilde_q, v, D)
        nll = jnp.sum(per_obs_nll(P, choice_idx))
        tilde_Q = compute_tilde_Q_M(sigma_e, tilde_gamma, tau, delta, tilde_q,
                                     v, D, choice_idx)
        g_m = compute_gmm_moments_M(delta, tilde_q, tilde_Q, gamma0, sigma_e,
                                     alpha, eta, w, R, L, z1, z2, z3)
        penalty_linear = -omega * jnp.dot(nu_vec, g_m)
        penalty_quad = (rho / 2.0) * jnp.sum((c_m - omega * g_m) ** 2)
        return nll + penalty_linear + penalty_quad

    inner_vg = jax.jit(jax.value_and_grad(_inner_obj, argnums=0))

    # --- forward pass: nll + moments ---
    def _forward(tau, tilde_gamma, alpha, sigma_e, eta, gamma0,
                 delta, tilde_q, v, choice_idx, D, w, R, L, z1, z2, z3):
        P = choice_probabilities(tau, tilde_gamma, delta, tilde_q, v, D)
        nll = jnp.sum(per_obs_nll(P, choice_idx))
        tilde_Q = compute_tilde_Q_M(sigma_e, tilde_gamma, tau, delta, tilde_q,
                                     v, D, choice_idx)
        g_m = compute_gmm_moments_M(delta, tilde_q, tilde_Q, gamma0, sigma_e,
                                     alpha, eta, w, R, L, z1, z2, z3)
        return nll, g_m

    forward_fn = jax.jit(_forward)

    # --- forward + jacobian w.r.t. theta_G = [tau, tg, alpha, sigma_e, eta, gamma0] ---
    def _forward_for_global(theta_G_vec, delta, tilde_q,
                            v, choice_idx, D, w, R, L, z1, z2, z3):
        tau = theta_G_vec[0]
        tilde_gamma = theta_G_vec[1]
        alpha = theta_G_vec[2]
        sigma_e = theta_G_vec[3]
        eta = theta_G_vec[4]
        gamma0 = theta_G_vec[5]
        P = choice_probabilities(tau, tilde_gamma, delta, tilde_q, v, D)
        nll = jnp.sum(per_obs_nll(P, choice_idx))
        tilde_Q = compute_tilde_Q_M(sigma_e, tilde_gamma, tau, delta, tilde_q,
                                     v, D, choice_idx)
        g_m = compute_gmm_moments_M(delta, tilde_q, tilde_Q, gamma0, sigma_e,
                                     alpha, eta, w, R, L, z1, z2, z3)
        return nll, g_m

    # grad of nll w.r.t. theta_G (6,)
    _grad_nll_fn = jax.grad(lambda tG, *a: _forward_for_global(tG, *a)[0], argnums=0)
    # jacobian of g_m (4,) w.r.t. theta_G (6,) -> (4,6)
    _jac_g_fn = jax.jacobian(lambda tG, *a: _forward_for_global(tG, *a)[1], argnums=0)

    def _forward_with_jac(theta_G_vec, delta, tilde_q,
                          v, choice_idx, D, w, R, L, z1, z2, z3):
        nll, g_m = _forward_for_global(theta_G_vec, delta, tilde_q,
                                        v, choice_idx, D, w, R, L, z1, z2, z3)
        grad_nll = _grad_nll_fn(theta_G_vec, delta, tilde_q,
                                 v, choice_idx, D, w, R, L, z1, z2, z3)
        jac_g = _jac_g_fn(theta_G_vec, delta, tilde_q,
                           v, choice_idx, D, w, R, L, z1, z2, z3)
        return nll, g_m, grad_nll, jac_g

    forward_with_jac_fn = jax.jit(_forward_with_jac)

    cache = {
        "inner_vg": inner_vg,
        "forward": forward_fn,
        "forward_with_jac": forward_with_jac_fn,
    }
    _jit_cache[J] = cache
    return cache


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_hybrid_data(
    firms_path: str,
    workers_path: str,
    params_path: str,
    M_subset: Optional[int] = None,
) -> Tuple[List[HybridMarketData], dict]:
    """Load multi-market data and return per-market HybridMarketData objects."""
    from scipy.spatial.distance import cdist

    firms_df = pd.read_csv(firms_path)
    workers_df = pd.read_csv(workers_path)
    params_df = pd.read_csv(params_path)

    # Parse true parameters
    pdict = dict(zip(params_df['parameter'], params_df['value']))
    true_params = {
        'tau': float(pdict['tau']),
        'alpha': float(pdict['alpha']),
        'gamma0': float(pdict['gamma0']),
        'gamma1': float(pdict['gamma1']),
        'sigma_e': float(pdict['sigma_e']),
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
    meta_firms = []  # for true-value init

    for mid in markets:
        fdf = firms_df[firms_df['market_id'] == mid].sort_values('firm_id')
        wdf = workers_df[workers_df['market_id'] == mid]

        J_m = len(fdf)
        N_m = len(wdf)

        v = jnp.asarray(wdf['x_skill'].values, dtype=jnp.float64)
        choice_idx = jnp.asarray(wdf['chosen_firm'].values, dtype=jnp.int32)
        w = jnp.asarray(fdf['w'].values, dtype=jnp.float64)
        R = jnp.asarray(fdf['Y'].values, dtype=jnp.float64)

        loc_firms = fdf[['x', 'y']].values
        worker_locs = np.column_stack([wdf['ell_x'].values, wdf['ell_y'].values])
        D = jnp.asarray(cdist(worker_locs, loc_firms, metric='euclidean'), dtype=jnp.float64)

        counts = np.bincount(np.array(wdf['chosen_firm'].values), minlength=J_m + 1).astype(np.float64)
        L = jnp.asarray(counts[1:], dtype=jnp.float64)

        z1 = jnp.asarray(fdf['z1'].values, dtype=jnp.float64) if 'z1' in fdf.columns else jnp.zeros(J_m)
        z2 = jnp.asarray(fdf['z2'].values, dtype=jnp.float64) if 'z2' in fdf.columns else jnp.zeros(J_m)
        z3 = jnp.ones(J_m, dtype=jnp.float64)

        omega = J_m / total_J

        md = HybridMarketData(
            market_id=int(mid), J=J_m, N=N_m,
            v=v, choice_idx=choice_idx, D=D,
            w=w, R=R, L=L, z1=z1, z2=z2, z3=z3,
            omega=omega,
        )
        market_datas.append(md)

        # Store firm-level true values for initialization
        qbar_col = 'qbar' if 'qbar' in fdf.columns else 'c'
        meta_firms.append({
            'w': fdf['w'].values.astype(np.float64),
            'xi': fdf['xi'].values.astype(np.float64),
            'qbar': fdf[qbar_col].values.astype(np.float64),
            'Y': fdf['Y'].values.astype(np.float64),
        })

    meta = {
        'true_params': true_params,
        'meta_firms': meta_firms,
        'M': len(markets),
        'total_J': total_J,
    }
    return market_datas, meta


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_from_true(market_datas: List[HybridMarketData], meta: dict
                   ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Initialize from true parameter values."""
    tp = meta['true_params']
    sigma_e = tp['sigma_e']
    gamma0 = tp['gamma0']
    eta = tp['eta']
    tilde_gamma = tp['tilde_gamma']

    theta_G = np.array([tp['tau'], tilde_gamma, tp['alpha'],
                        sigma_e, eta, gamma0], dtype=np.float64)

    deltas = []
    tilde_qs = []
    for i, md in enumerate(market_datas):
        mf = meta['meta_firms'][i]
        w_np = mf['w']
        xi_np = mf['xi']
        qbar_np = mf['qbar']
        delta_true = eta * np.log(np.maximum(w_np, 1e-300)) + xi_np
        ln_qbar_true = np.log(np.maximum(qbar_np, 1e-300))
        tilde_q_true = (ln_qbar_true - gamma0) / sigma_e
        deltas.append(delta_true.astype(np.float64))
        tilde_qs.append(tilde_q_true.astype(np.float64))

    return theta_G, deltas, tilde_qs


def init_naive(market_datas: List[HybridMarketData], meta: dict
               ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Data-driven initialization (pooled 6-step procedure)."""
    from screening.analysis.lib.naive_init import compute_pooled_naive_init
    return compute_pooled_naive_init(market_datas)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_all_moments(
    theta_G: np.ndarray,
    deltas: List[np.ndarray],
    tilde_qs: List[np.ndarray],
    market_datas: List[HybridMarketData],
) -> Tuple[List[np.ndarray], List[float], np.ndarray]:
    """Compute per-market moments and nll at current parameters."""
    tau, tg, alpha, sigma_e, eta, gamma0 = theta_G
    g_ms = []
    nlls = []
    g_bar = np.zeros(4, dtype=np.float64)

    for i, md in enumerate(market_datas):
        fns = _get_jitted_fns(md.J)
        delta_j = jnp.asarray(deltas[i], dtype=jnp.float64)
        tq_j = jnp.asarray(tilde_qs[i], dtype=jnp.float64)
        nll_m, g_m = fns["forward"](
            jnp.float64(tau), jnp.float64(tg), jnp.float64(alpha),
            jnp.float64(sigma_e), jnp.float64(eta), jnp.float64(gamma0),
            delta_j, tq_j,
            md.v, md.choice_idx, md.D, md.w, md.R, md.L, md.z1, md.z2, md.z3,
        )
        g_m_np = np.asarray(g_m, dtype=np.float64)
        g_ms.append(g_m_np)
        nlls.append(float(nll_m))
        g_bar += md.omega * g_m_np

    return g_ms, nlls, g_bar


def solve_market_subproblem(
    md: HybridMarketData,
    theta_G: np.ndarray,
    delta_init: np.ndarray,
    tq_init: np.ndarray,
    nu: np.ndarray,
    rho: float,
    c_m: np.ndarray,
    inner_maxiter: int = 200,
    inner_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Solve the market block subproblem (Step 2).

    Returns (delta_new, tq_new, g_m_new, nll_new, converged).
    """
    from scipy.optimize import minimize as sp_minimize

    J = md.J
    fns = _get_jitted_fns(J)
    inner_vg_fn = fns["inner_vg"]

    tau, tg, alpha, sigma_e, eta, gamma0 = theta_G
    tau_j = jnp.float64(tau)
    tg_j = jnp.float64(tg)
    alpha_j = jnp.float64(alpha)
    sigma_e_j = jnp.float64(sigma_e)
    eta_j = jnp.float64(eta)
    gamma0_j = jnp.float64(gamma0)
    nu_j = jnp.asarray(nu, dtype=jnp.float64)
    rho_j = jnp.float64(rho)
    c_m_j = jnp.asarray(c_m, dtype=jnp.float64)
    omega_j = jnp.float64(md.omega)

    data_args = (md.v, md.choice_idx, md.D, md.w, md.R, md.L,
                 md.z1, md.z2, md.z3)

    # Data-driven tq re-init: place tq in active screening region
    choice_np = np.asarray(md.choice_idx, dtype=np.int32)
    v_np = np.asarray(md.v)
    tq_override = np.empty(J, dtype=np.float64)
    for j in range(J):
        mask_j = choice_np == (j + 1)
        if np.any(mask_j):
            v_low = float(np.quantile(v_np[mask_j], 0.05))
            tq_override[j] = tg * v_low
        else:
            tq_override[j] = tq_init[j]
    z0 = np.concatenate([delta_init, tq_override]).astype(np.float64)

    def scipy_callback(z_flat):
        z_jax = jnp.asarray(z_flat, dtype=jnp.float64)
        val, grad = inner_vg_fn(
            z_jax, tau_j, tg_j, alpha_j, sigma_e_j, eta_j, gamma0_j,
            nu_j, rho_j, c_m_j, omega_j, *data_args,
        )
        return float(val), np.asarray(grad, dtype=np.float64)

    result = sp_minimize(
        scipy_callback, z0, method="L-BFGS-B", jac=True,
        options={"maxiter": inner_maxiter, "gtol": inner_tol, "ftol": 1e-15},
    )

    z_hat = result.x
    delta_new = z_hat[:J].astype(np.float64)
    tq_new = z_hat[J:].astype(np.float64)

    # Evaluate forward pass at solution
    fwd = fns["forward"]
    nll_new, g_m_new = fwd(
        tau_j, tg_j, alpha_j, sigma_e_j, eta_j, gamma0_j,
        jnp.asarray(delta_new), jnp.asarray(tq_new), *data_args,
    )

    return (delta_new, tq_new,
            np.asarray(g_m_new, dtype=np.float64),
            float(nll_new), result.success)


def solve_global_subproblem(
    market_datas: List[HybridMarketData],
    deltas: List[np.ndarray],
    tilde_qs: List[np.ndarray],
    theta_G_init: np.ndarray,
    W: np.ndarray,
    nu: np.ndarray,
    rho: float,
    global_maxiter: int = 100,
    global_tol: float = 1e-5,
) -> Tuple[np.ndarray, float]:
    """Solve the global subproblem (Step 4).

    Returns (theta_G_new, final_obj).
    """
    from scipy.optimize import minimize as sp_minimize

    W_plus_rhoI = W + rho * np.eye(4)
    W_plus_rhoI_inv = np.linalg.inv(W_plus_rhoI)

    def phi_and_grad(theta_G_flat):
        theta_G_flat = np.asarray(theta_G_flat, dtype=np.float64)
        tG_jax = jnp.asarray(theta_G_flat, dtype=jnp.float64)

        total_nll = 0.0
        grad_nll_total = np.zeros(6, dtype=np.float64)
        g_bar = np.zeros(4, dtype=np.float64)
        jac_g_bar = np.zeros((4, 6), dtype=np.float64)

        for i, md in enumerate(market_datas):
            fns = _get_jitted_fns(md.J)
            delta_j = jnp.asarray(deltas[i], dtype=jnp.float64)
            tq_j = jnp.asarray(tilde_qs[i], dtype=jnp.float64)
            data_args = (md.v, md.choice_idx, md.D, md.w, md.R, md.L,
                         md.z1, md.z2, md.z3)

            nll_m, g_m, grad_nll_m, jac_g_m = fns["forward_with_jac"](
                tG_jax, delta_j, tq_j, *data_args,
            )

            total_nll += float(nll_m)
            grad_nll_total += np.asarray(grad_nll_m, dtype=np.float64)
            g_bar += md.omega * np.asarray(g_m, dtype=np.float64)
            jac_g_bar += md.omega * np.asarray(jac_g_m, dtype=np.float64)

        # Concentrate out mu*
        mu_star = W_plus_rhoI_inv @ (rho * g_bar - nu)

        # Phi_k value
        residual = mu_star - g_bar
        phi = (total_nll
               + 0.5 * mu_star @ W @ mu_star
               + nu @ residual
               + (rho / 2.0) * np.dot(residual, residual))

        # Gradient via envelope theorem
        lambda_eff = nu + rho * residual  # = nu + rho*(mu* - g_bar)
        grad_phi = grad_nll_total - jac_g_bar.T @ lambda_eff

        return phi, grad_phi

    bounds = [
        (1e-6, 1.0 - 1e-6),   # tau
        (1e-6, None),           # tilde_gamma
        (1e-6, 1.0 - 1e-6),   # alpha
        (1e-6, None),           # sigma_e
        (1e-6, None),           # eta
        (None, None),           # gamma0
    ]

    result = sp_minimize(
        phi_and_grad, theta_G_init, method="L-BFGS-B", jac=True,
        bounds=bounds,
        options={"maxiter": global_maxiter, "gtol": global_tol, "ftol": 1e-15},
    )

    return result.x.astype(np.float64), float(result.fun)


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------


def run_auglag(
    market_datas: List[HybridMarketData],
    W: np.ndarray,
    theta_G_init: np.ndarray,
    deltas_init: List[np.ndarray],
    tilde_qs_init: List[np.ndarray],
    *,
    max_outer_iter: int = 200,
    inner_maxiter: int = 200,
    inner_tol: float = 1e-6,
    global_maxiter: int = 100,
    global_tol: float = 1e-5,
    verbose: bool = True,
) -> dict:
    """Run the augmented-Lagrangian hybrid estimator."""
    M = len(market_datas)

    # Initialize state
    theta_G = theta_G_init.copy()
    deltas = [d.copy() for d in deltas_init]
    tilde_qs = [q.copy() for q in tilde_qs_init]

    # Initial moments
    g_ms, nlls, g_bar = compute_all_moments(theta_G, deltas, tilde_qs, market_datas)
    total_nll = sum(nlls)

    # rho_0 = tr(W) / d_g
    d_g = 4
    rho = float(np.trace(W)) / d_g

    # nu_0 = 0
    nu = np.zeros(d_g, dtype=np.float64)

    # mu_0 = (W + rho*I)^{-1} (rho * g_bar)
    W_plus_rhoI_inv = np.linalg.inv(W + rho * np.eye(d_g))
    mu = W_plus_rhoI_inv @ (rho * g_bar)

    history = []
    converged = False

    if verbose:
        obj = total_nll + 0.5 * g_bar @ W @ g_bar
        print(f"AugLag init: nll={total_nll:.4f}  |g_bar|={np.linalg.norm(g_bar):.4e}  "
              f"obj={obj:.4f}  rho={rho:.4f}")
        print(f"  theta_G = {theta_G}")

    for k in range(max_outer_iter):
        iter_start = time.perf_counter()
        theta_G_old = theta_G.copy()
        g_bar_old = g_bar.copy()

        # --- Step 1+2: Market block ---
        n_converged = 0
        for i, md in enumerate(market_datas):
            c_m = mu - g_bar + md.omega * g_ms[i]
            delta_new, tq_new, g_m_new, nll_new, conv = solve_market_subproblem(
                md, theta_G, deltas[i], tilde_qs[i],
                nu, rho, c_m, inner_maxiter, inner_tol,
            )
            deltas[i] = delta_new
            tilde_qs[i] = tq_new
            g_ms[i] = g_m_new
            nlls[i] = nll_new
            n_converged += int(conv)

        # Aggregate after market block
        g_bar_mid = np.zeros(d_g, dtype=np.float64)
        for i, md in enumerate(market_datas):
            g_bar_mid += md.omega * g_ms[i]
        total_nll_mid = sum(nlls)

        # --- Step 3: Intermediate mu update ---
        W_plus_rhoI_inv = np.linalg.inv(W + rho * np.eye(d_g))
        mu_mid = W_plus_rhoI_inv @ (rho * g_bar_mid - nu)

        # --- Step 4: Global block ---
        theta_G_new, phi_val = solve_global_subproblem(
            market_datas, deltas, tilde_qs, theta_G, W, nu, rho,
            global_maxiter, global_tol,
        )
        theta_G = theta_G_new

        # --- Step 5: Recompute moments at new globals, update mu and nu ---
        g_ms, nlls, g_bar = compute_all_moments(theta_G, deltas, tilde_qs, market_datas)
        total_nll = sum(nlls)

        mu = W_plus_rhoI_inv @ (rho * g_bar - nu)
        nu_new = nu + rho * (mu - g_bar)

        # --- Step 6: Convergence check ---
        primal_res = np.linalg.norm(mu - g_bar)
        dual_res = rho * np.linalg.norm(g_bar - g_bar_old)
        param_change = np.linalg.norm(theta_G - theta_G_old) / max(1.0, np.linalg.norm(theta_G))

        rel_primal = primal_res / max(1.0, np.linalg.norm(mu))
        rel_dual = dual_res / max(1.0, np.linalg.norm(nu_new))

        obj = total_nll + 0.5 * g_bar @ W @ g_bar

        iter_time = time.perf_counter() - iter_start
        entry = {
            'iter': k, 'obj': obj, 'nll': total_nll,
            'primal_res': primal_res, 'dual_res': dual_res,
            'rel_primal': rel_primal, 'rel_dual': rel_dual,
            'param_change': param_change, 'rho': rho,
            'theta_G': theta_G.tolist(), 'g_bar': g_bar.tolist(),
            'n_converged': n_converged, 'wall_sec': iter_time,
        }
        history.append(entry)

        if verbose:
            print(f"  [iter {k:3d}] obj={obj:.4f}  nll={total_nll:.4f}  "
                  f"|g_bar|={np.linalg.norm(g_bar):.4e}  "
                  f"primal={rel_primal:.3e}  dual={rel_dual:.3e}  "
                  f"dparam={param_change:.3e}  rho={rho:.2f}  "
                  f"inner_conv={n_converged}/{M}  {iter_time:.1f}s")

        nu = nu_new

        converged = (rel_primal < 1e-4 and rel_dual < 1e-4 and param_change < 1e-5)
        if converged:
            if verbose:
                print(f"  Converged at iteration {k}.")
            break

        # --- rho update (residual balancing) ---
        if primal_res > 10.0 * dual_res:
            rho = min(2.0 * rho, 1e4)
        elif dual_res > 10.0 * primal_res:
            rho = max(rho / 2.0, 1e-4)

    return {
        'theta_G': theta_G,
        'deltas': deltas,
        'tilde_qs': tilde_qs,
        'mu': mu,
        'nu': nu,
        'g_bar': g_bar,
        'g_ms': g_ms,
        'total_nll': total_nll,
        'obj': total_nll + 0.5 * g_bar @ W @ g_bar,
        'converged': converged,
        'n_outer_iters': k + 1,
        'history': history,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-process augmented-Lagrangian hybrid MLE+GMM solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _base = str(Path(__file__).resolve().parents[3])
    p.add_argument("--firms_path", type=str,
                   default=os.path.join(_base, "data_v2", "clean", "equilibrium_firms.csv"))
    p.add_argument("--workers_path", type=str,
                   default=os.path.join(_base, "data_v2", "build", "workers_dataset.csv"))
    p.add_argument("--params_path", type=str,
                   default=os.path.join(_base, "data_v2", "raw", "parameters_effective.csv"))
    p.add_argument("--out_dir", type=str,
                   default=os.path.join(_base, "output", "estimation"))
    p.add_argument("--M", type=int, default=2)
    p.add_argument("--true_init", action="store_true",
                   help="Initialize from true parameter values")
    p.add_argument("--max_outer_iter", type=int, default=200)
    p.add_argument("--inner_maxiter", type=int, default=200)
    p.add_argument("--inner_tol", type=float, default=1e-6)
    p.add_argument("--global_maxiter", type=int, default=100)
    p.add_argument("--global_tol", type=float, default=1e-5)
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading data...")
    market_datas, meta = load_hybrid_data(
        args.firms_path, args.workers_path, args.params_path,
        M_subset=args.M,
    )
    tp = meta['true_params']
    M = meta['M']
    total_J = meta['total_J']
    print(f"  M={M}, total_J={total_J}")
    print(f"  True params: tau={tp['tau']}, alpha={tp['alpha']}, "
          f"gamma0={tp['gamma0']}, sigma_e={tp['sigma_e']}, "
          f"eta={tp['eta']}, tilde_gamma={tp['tilde_gamma']:.4f}")

    # Initialize
    if args.true_init:
        print("Initializing from true values...")
        theta_G, deltas, tilde_qs = init_from_true(market_datas, meta)
    else:
        print("Computing naive initialization...")
        theta_G, deltas, tilde_qs = init_naive(market_datas, meta)
    print(f"  theta_G_init = {theta_G}")

    # Identity weighting matrix
    W = np.eye(4, dtype=np.float64)

    print(f"\nStarting augmented-Lagrangian solver (M={M})...")
    t0 = time.perf_counter()
    result = run_auglag(
        market_datas, W, theta_G, deltas, tilde_qs,
        max_outer_iter=args.max_outer_iter,
        inner_maxiter=args.inner_maxiter,
        inner_tol=args.inner_tol,
        global_maxiter=args.global_maxiter,
        global_tol=args.global_tol,
        verbose=args.verbose,
    )
    wall = time.perf_counter() - t0

    # --- Report ---
    tG = result['theta_G']
    print(f"\n{'='*60}")
    print(f"Augmented-Lagrangian Hybrid MLE+GMM")
    print(f"  Converged: {result['converged']}")
    print(f"  Outer iterations: {result['n_outer_iters']}")
    print(f"  Final objective: {result['obj']:.4f}")
    print(f"  Final NLL: {result['total_nll']:.4f}")
    print(f"  |g_bar|: {np.linalg.norm(result['g_bar']):.4e}")
    print(f"  Wall time: {wall:.1f}s")
    print(f"{'='*60}")

    names = ['tau', 'tilde_gamma', 'alpha', 'sigma_e', 'eta', 'gamma0']
    true_vals = [tp['tau'], tp['tilde_gamma'], tp['alpha'],
                 tp['sigma_e'], tp['eta'], tp['gamma0']]
    print("\n=== Parameter Recovery ===")
    for name, hat, true in zip(names, tG, true_vals):
        print(f"  {name:14s}: hat={hat:.6f}  true={true:.6f}  err={hat-true:+.6f}")

    # Delta and tilde_q correlation
    for i, md in enumerate(market_datas):
        mf = meta['meta_firms'][i]
        delta_true = tp['eta'] * np.log(np.maximum(mf['w'], 1e-300)) + mf['xi']
        ln_qbar_true = np.log(np.maximum(mf['qbar'], 1e-300))
        tq_true = (ln_qbar_true - tp['gamma0']) / tp['sigma_e']

        d_corr = float(np.corrcoef(result['deltas'][i], delta_true)[0, 1])
        tq_corr = float(np.corrcoef(result['tilde_qs'][i], tq_true)[0, 1])
        d_rmse = float(np.sqrt(np.mean((result['deltas'][i] - delta_true) ** 2)))
        tq_rmse = float(np.sqrt(np.mean((result['tilde_qs'][i] - tq_true) ** 2)))
        print(f"  Market {md.market_id}: delta_corr={d_corr:.4f} rmse={d_rmse:.4f}  "
              f"tq_corr={tq_corr:.4f} rmse={tq_rmse:.4f}")

    print(f"\n  g_bar = {result['g_bar']}")

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hybrid_auglag_estimates.json"

    out = {
        "solver": "augmented_lagrangian_hybrid",
        "M": M, "total_J": total_J,
        "converged": result['converged'],
        "n_outer_iters": result['n_outer_iters'],
        "objective": result['obj'],
        "nll": result['total_nll'],
        "theta_G": tG.tolist(),
        "g_bar": result['g_bar'].tolist(),
        "mu": result['mu'].tolist(),
        "nu": result['nu'].tolist(),
        "wall_sec": wall,
        "true_params": tp,
        "history": result['history'],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
