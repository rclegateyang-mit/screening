#!/usr/bin/env python3
"""Per-market computation for distributed augmented-Lagrangian hybrid solver.

Pure computation — no MPI.  Each MPI rank calls these functions on its
locally held ``HybridMarketData`` objects.  Data stays in-process for
the lifetime of the rank (received once via ``comm.scatter`` at startup).

JIT caching is keyed on J (number of firms) — same pattern as
``distributed_worker.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridMarketData:
    """Per-market data bundle (plain NumPy, picklable for MPI scatter).

    Extends the single-process version in ``run_hybrid_auglag.py`` by using
    ``np.ndarray`` instead of ``jnp.ndarray`` for MPI compatibility.
    """
    market_id: int
    J: int
    N: int
    v: np.ndarray           # (N,)
    choice_idx: np.ndarray  # (N,) int32
    D: np.ndarray           # (N, J)
    w: np.ndarray           # (J,)
    R: np.ndarray           # (J,) revenue
    L: np.ndarray           # (J,) labor counts
    z1: np.ndarray          # (J,)
    z2: np.ndarray          # (J,)
    z3: np.ndarray          # (J,)
    omega: float            # J_m / sum(J_j) market weight


@dataclass
class HybridMarketState:
    """Mutable per-market state held by worker across outer iterations."""
    delta: np.ndarray       # (J,)
    tilde_q: np.ndarray     # (J,)
    g_m: np.ndarray         # (4,) cached moment vector
    nll: float              # cached NLL


# ---------------------------------------------------------------------------
# JAX initialisation (call once per process)
# ---------------------------------------------------------------------------


def init_jax() -> None:
    """Set up JAX with float64 and single-threaded XLA."""
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                          "intra_op_parallelism_threads=1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import jax
    jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# JIT cache (same pattern as run_hybrid_auglag.py lines 93-169)
# ---------------------------------------------------------------------------

_jit_cache: Dict[int, dict] = {}


def _get_jitted_fns(J: int) -> dict:
    """Return JIT'd functions for the given J, creating if needed."""
    if J in _jit_cache:
        return _jit_cache[J]

    import jax
    import jax.numpy as jnp
    from screening.analysis.lib.model_components import (
        choice_probabilities,
        per_obs_nll,
        compute_tilde_Q_M,
        compute_gmm_moments_M,
    )

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

    _grad_nll_fn = jax.grad(
        lambda tG, *a: _forward_for_global(tG, *a)[0], argnums=0)
    _jac_g_fn = jax.jacobian(
        lambda tG, *a: _forward_for_global(tG, *a)[1], argnums=0)

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
# Core computation functions
# ---------------------------------------------------------------------------


def solve_market_subproblem(
    md: HybridMarketData,
    state: HybridMarketState,
    theta_G: np.ndarray,
    nu: np.ndarray,
    rho: float,
    c_m: np.ndarray,
    inner_maxiter: int = 200,
    inner_tol: float = 1e-6,
) -> Tuple[HybridMarketState, bool]:
    """Solve the market block subproblem (Step 2).

    Warm-starts from state.delta, state.tilde_q.
    Returns (updated_state, converged).
    """
    from scipy.optimize import minimize as sp_minimize
    import jax.numpy as jnp

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

    # Convert data to JAX arrays
    v_j = jnp.asarray(md.v, dtype=jnp.float64)
    choice_j = jnp.asarray(md.choice_idx, dtype=jnp.int32)
    D_j = jnp.asarray(md.D, dtype=jnp.float64)
    w_j = jnp.asarray(md.w, dtype=jnp.float64)
    R_j = jnp.asarray(md.R, dtype=jnp.float64)
    L_j = jnp.asarray(md.L, dtype=jnp.float64)
    z1_j = jnp.asarray(md.z1, dtype=jnp.float64)
    z2_j = jnp.asarray(md.z2, dtype=jnp.float64)
    z3_j = jnp.asarray(md.z3, dtype=jnp.float64)
    data_args = (v_j, choice_j, D_j, w_j, R_j, L_j, z1_j, z2_j, z3_j)

    # Data-driven tq re-init: place tq in active screening region using
    # matched-worker skill quantiles (same logic as distributed_worker)
    tq_override = np.empty(J, dtype=np.float64)
    choice_np = np.asarray(md.choice_idx, dtype=np.int32)
    v_np = np.asarray(md.v)
    for j in range(J):
        mask_j = choice_np == (j + 1)
        if np.any(mask_j):
            v_low = float(np.quantile(v_np[mask_j], 0.05))
            tq_override[j] = tg * v_low
        else:
            tq_override[j] = state.tilde_q[j]
    z0 = np.concatenate([state.delta, tq_override]).astype(np.float64)

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

    new_state = HybridMarketState(
        delta=delta_new,
        tilde_q=tq_new,
        g_m=np.asarray(g_m_new, dtype=np.float64),
        nll=float(nll_new),
    )
    return new_state, result.success


def forward_eval(
    md: HybridMarketData,
    state: HybridMarketState,
    theta_G: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Lightweight forward evaluation: returns (nll, g_m(4,))."""
    import jax.numpy as jnp

    fns = _get_jitted_fns(md.J)
    tau, tg, alpha, sigma_e, eta, gamma0 = theta_G

    v_j = jnp.asarray(md.v, dtype=jnp.float64)
    choice_j = jnp.asarray(md.choice_idx, dtype=jnp.int32)
    D_j = jnp.asarray(md.D, dtype=jnp.float64)
    w_j = jnp.asarray(md.w, dtype=jnp.float64)
    R_j = jnp.asarray(md.R, dtype=jnp.float64)
    L_j = jnp.asarray(md.L, dtype=jnp.float64)
    z1_j = jnp.asarray(md.z1, dtype=jnp.float64)
    z2_j = jnp.asarray(md.z2, dtype=jnp.float64)
    z3_j = jnp.asarray(md.z3, dtype=jnp.float64)

    nll_m, g_m = fns["forward"](
        jnp.float64(tau), jnp.float64(tg), jnp.float64(alpha),
        jnp.float64(sigma_e), jnp.float64(eta), jnp.float64(gamma0),
        jnp.asarray(state.delta), jnp.asarray(state.tilde_q),
        v_j, choice_j, D_j, w_j, R_j, L_j, z1_j, z2_j, z3_j,
    )
    return float(nll_m), np.asarray(g_m, dtype=np.float64)


def forward_with_jac_eval(
    md: HybridMarketData,
    state: HybridMarketState,
    theta_G_vec: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Forward + Jacobian evaluation for global solve.

    Returns (nll, g_m(4,), grad_nll(6,), jac_g(4,6)).
    """
    import jax.numpy as jnp

    fns = _get_jitted_fns(md.J)

    tG_jax = jnp.asarray(theta_G_vec, dtype=jnp.float64)
    delta_j = jnp.asarray(state.delta, dtype=jnp.float64)
    tq_j = jnp.asarray(state.tilde_q, dtype=jnp.float64)

    v_j = jnp.asarray(md.v, dtype=jnp.float64)
    choice_j = jnp.asarray(md.choice_idx, dtype=jnp.int32)
    D_j = jnp.asarray(md.D, dtype=jnp.float64)
    w_j = jnp.asarray(md.w, dtype=jnp.float64)
    R_j = jnp.asarray(md.R, dtype=jnp.float64)
    L_j = jnp.asarray(md.L, dtype=jnp.float64)
    z1_j = jnp.asarray(md.z1, dtype=jnp.float64)
    z2_j = jnp.asarray(md.z2, dtype=jnp.float64)
    z3_j = jnp.asarray(md.z3, dtype=jnp.float64)

    nll_m, g_m, grad_nll_m, jac_g_m = fns["forward_with_jac"](
        tG_jax, delta_j, tq_j,
        v_j, choice_j, D_j, w_j, R_j, L_j, z1_j, z2_j, z3_j,
    )

    return (
        float(nll_m),
        np.asarray(g_m, dtype=np.float64),
        np.asarray(grad_nll_m, dtype=np.float64),
        np.asarray(jac_g_m, dtype=np.float64),
    )
