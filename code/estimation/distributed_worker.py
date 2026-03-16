#!/usr/bin/env python3
"""Worker logic for distributed MLE: inner optimization + Hessian extraction.

Pure computation — no MPI, no process pools.  Each MPI rank calls
``solve_market`` on its locally held ``MarketData`` objects.  Data stays
in-process for the lifetime of the rank (received once via ``comm.scatter``
at startup).

JIT caching
-----------
Compiled functions are cached per J (number of firms) via ``_get_jitted_fns``.
J must be a compile-time constant because it controls parameter vector
slicing (``tM[:J]``, ``z_m[J:]``).  Markets with different N (number of
workers) reuse the same compiled function — JAX's internal trace cache
automatically handles shape variation in data arrays (X_m, D_m, etc.)
by recompiling and caching per unique (J, N) shape combination.

This means:
- First call for a new (J, N) pair triggers XLA compilation (~seconds).
- Subsequent calls with the same shapes are fast cache hits.
- Markets may have heterogeneous J and/or N across a single rank.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketData:
    """Per-market data bundle (plain NumPy, picklable for MPI scatter)."""
    market_id: int
    X_m: np.ndarray          # (N_per,)
    choice_m: np.ndarray     # (N_per,) int32
    D_m: np.ndarray          # (N_per, J_per)
    w_m: np.ndarray          # (J_per,)
    Y_m: np.ndarray          # (J_per,)
    labor_m: np.ndarray      # (J_per,)
    mu_e: float
    J_per: int
    N_per: int


def compute_naive_init(
    md: MarketData,
    alpha: float,
    gamma: float,
    tau0: float = 0.05,
    reg_delta: float = 1e-6,
    maxiter: int = 500,
) -> tuple[float, np.ndarray]:
    """Compute naive (tau, delta, qbar) initial guesses from one market's data.

    tau, delta: from MNL of choices on distance + firm FE (ignoring screening).
    qbar: from screening FOC using observed wages, revenue, and avg skill.

    Returns (tau_hat, theta_m_init) where theta_m_init = [delta_init, qbar_init].
    """
    from scipy.optimize import minimize

    J = md.J_per
    N = md.N_per
    D = md.D_m             # (N, J)
    choice = md.choice_m   # (N,) int32, 0=outside option

    # --- MNL for (tau, delta) ---
    counts_all = np.bincount(np.clip(choice, 0, J), minlength=J + 1)
    s0 = counts_all[0] / max(N, 1)
    s = counts_all[1:] / max(N, 1)
    eps = 1e-12
    delta0 = np.log(np.maximum(s, eps)) - np.log(np.maximum(s0, eps))
    theta0_mnl = np.concatenate(([float(tau0)], delta0))

    mask_in = choice > 0
    j_idx = (choice[mask_in] - 1).astype(int)
    d_chosen = D[mask_in, j_idx] if np.any(mask_in) else np.array([], dtype=float)
    counts_firms = np.bincount(j_idx, minlength=J) if np.any(mask_in) else np.zeros(J)

    def nll_and_grad(theta_vec):
        theta_vec = np.asarray(theta_vec, float).ravel()
        tau_v = float(theta_vec[0])
        delta_vals = theta_vec[1:]

        U = delta_vals[None, :] - tau_v * D
        a = np.maximum(0.0, np.max(U, axis=1))
        exp_U_shift = np.exp(U - a[:, None])
        denom_scaled = np.exp(-a) + exp_U_shift.sum(axis=1)
        log_denom = a + np.log(denom_scaled)
        P = np.exp(U - log_denom[:, None])

        sum_log_denom = np.sum(log_denom)
        sum_delta = float(delta_vals @ counts_firms)
        sum_d = float(np.sum(d_chosen))
        nll = (sum_log_denom - (sum_delta - tau_v * sum_d)
               + 0.5 * reg_delta * float(delta_vals @ delta_vals))

        g_delta = P.sum(axis=0) - counts_firms + reg_delta * delta_vals
        g_tau = -float(np.sum(P * D)) + sum_d
        g = np.concatenate(([g_tau], g_delta.astype(float)))
        return nll, g

    bounds = [(0.0, 1.0)] + [(-np.inf, np.inf)] * J
    res = minimize(nll_and_grad, theta0_mnl, method="L-BFGS-B", jac=True,
                   bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-10, "iprint": -1})
    tau_hat = float(res.x[0])
    delta_hat = res.x[1:].copy()

    # --- qbar from screening FOC ---
    # ln qbar_j = ln(1-alpha) + ln(w_j) - ln(Y_j / L_j) + avg_skill_j
    w_m = md.w_m
    Y_m = md.Y_m
    labor_m = md.labor_m
    X_m = md.X_m

    # Average skill of workers choosing each firm
    avg_skill = np.zeros(J, dtype=np.float64)
    for j in range(J):
        mask_j = choice == (j + 1)
        if np.any(mask_j):
            avg_skill[j] = np.mean(X_m[mask_j])

    # Compute qbar for firms with workers
    has_workers = labor_m > 0
    qbar_init = np.full(J, np.nan, dtype=np.float64)
    safe_labor = np.where(has_workers, labor_m, 1.0)
    log_qbar = (np.log(max(1.0 - alpha, 1e-12))
                + np.log(np.maximum(w_m, 1e-12))
                - np.log(np.maximum(Y_m / safe_labor, 1e-12))
                + np.log(np.maximum(avg_skill, 1e-12)))
    qbar_init[has_workers] = np.exp(log_qbar[has_workers])

    # Firms with zero workers: fall back to median of valid firms
    if not np.all(has_workers):
        valid_qbar = qbar_init[has_workers]
        fallback = float(np.median(valid_qbar)) if valid_qbar.size > 0 else 0.5
        qbar_init[~has_workers] = fallback

    qbar_init = np.maximum(qbar_init, 1e-10)

    theta_m_init = np.concatenate([delta_hat, qbar_init])
    return tau_hat, theta_m_init


@dataclass
class MarketResult:
    """Return value from each worker to the master."""
    market_id: int
    nll_m: float                # -ℓ_m at optimum
    grad_G_m: np.ndarray        # (5,) ∂(-ℓ_m)/∂θ_G
    schur_m: np.ndarray         # (5,5) Schur complement contribution
    theta_m_hat: np.ndarray     # (2*J_per,) optimized [δ_m, q̄_m]
    inner_converged: bool
    inner_iters: int


# ---------------------------------------------------------------------------
# JAX initialisation (call once per process, before any solve_market calls)
# ---------------------------------------------------------------------------


def init_jax():
    """Set up JAX with float64 and single-threaded XLA (one rank per core)."""
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                          "intra_op_parallelism_threads=1")
    # Limit LLVM compilation threads to avoid OOM when many ranks compile
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import jax
    jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Inner transforms for θ_m = [δ(J), q̄(J)]
# ---------------------------------------------------------------------------

QBAR_FLOOR = 1e-8


def _inner_fwd(z_m, J):
    """Map unconstrained z_m → (δ, q̄) with softplus on q̄."""
    import jax.numpy as jnp
    from jax import nn as jnn
    delta = z_m[:J]
    qbar = QBAR_FLOOR + jnn.softplus(z_m[J:])
    return jnp.concatenate([delta, qbar])


def _inner_inv(theta_m, J):
    """Map (δ, q̄) → unconstrained z_m."""
    import jax.numpy as jnp
    delta = theta_m[:J]
    qbar = theta_m[J:]
    qbar_shifted = jnp.maximum(qbar - QBAR_FLOOR, 1e-12)
    qbar_z = jnp.where(qbar_shifted < 20.0,
                        jnp.log(jnp.expm1(qbar_shifted)),
                        qbar_shifted + jnp.log1p(-jnp.exp(-qbar_shifted)))
    return jnp.concatenate([delta, qbar_z])


# ---------------------------------------------------------------------------
# JIT-compiled function cache
#
# Keyed on J (number of firms), which is a compile-time constant used for
# parameter vector slicing.  Variation in N (workers) is handled by JAX's
# internal trace cache: same compiled function, different cached traces
# per unique array shape.
# ---------------------------------------------------------------------------

_jit_cache: Dict[int, dict] = {}


def _market_nll_pure(tG, tM, X_m, choice_m, D_m, w_m, Y_m, labor_m, mu_e, J):
    """Pure-function market NLL: all inputs are explicit arguments.

    J is a compile-time constant (captured via closure in _get_jitted_fns).
    N (worker count) enters only through array shapes and is handled
    automatically by JAX's shape-polymorphic tracing.
    """
    import jax.numpy as jnp
    from code.estimation.jax_model import compute_penalty_components_jax

    tau, alpha = tG[0], tG[1]
    gamma, sigma_e = tG[2], tG[3]
    delta_m = tM[:J]
    qbar_m = tM[J:]
    theta_core = jnp.concatenate([jnp.array([tau, alpha]), delta_m, qbar_m])
    aux = {"D_nat": D_m, "mu_e": mu_e, "gamma": gamma, "sigma_e": sigma_e}
    _, per_obs_nll, _, _, _ = compute_penalty_components_jax(
        theta_core, X_m, choice_m, aux, w_m, Y_m, labor_m)
    return jnp.sum(per_obs_nll)


def _get_jitted_fns(J: int) -> dict:
    """Return dict of JIT-compiled functions for the given J, creating if needed.

    All functions take data arrays as explicit arguments (not via closure
    capture of data), so JAX can reuse compiled XLA code across markets.
    J is baked in at compile time; N variation is handled by JAX's internal
    shape-based trace cache (one XLA compilation per unique N, cached
    thereafter).

    The inner optimization uses scipy.optimize.minimize(L-BFGS-B) with a
    JIT-compiled value_and_grad callback.  This avoids jaxopt's XLA
    compilation issues (LLVM OOM with lax.while_loop) while keeping each
    function evaluation fast through JAX JIT caching.
    """
    if J in _jit_cache:
        return _jit_cache[J]

    import jax
    import jax.numpy as jnp

    # Fix J at trace time via closure
    def _nll(tG, tM, X_m, choice_m, D_m, w_m, Y_m, labor_m, mu_e):
        return _market_nll_pure(tG, tM, X_m, choice_m, D_m, w_m, Y_m, labor_m, mu_e, J)

    # Inner value-and-grad: z_m is optimized, rest are fixed args.
    # JIT-compiled so each scipy callback call is fast after first trace.
    def _inner_loss(z_m, tG, X_m, choice_m, D_m, w_m, Y_m, labor_m, mu_e):
        tM = _inner_fwd(z_m, J)
        return _nll(tG, tM, X_m, choice_m, D_m, w_m, Y_m, labor_m, mu_e)

    inner_vg = jax.jit(jax.value_and_grad(_inner_loss, argnums=0))

    # Gradient of NLL w.r.t. θ_G (first arg), with θ_m and data fixed
    grad_G_fn = jax.jit(jax.grad(_nll, argnums=0))

    # Hessian components for Schur complement (compiled lazily on first use)
    hess_GG_fn = jax.jit(jax.hessian(_nll, argnums=0))
    hess_Gm_fn = jax.jit(jax.jacobian(jax.grad(_nll, argnums=0), argnums=1))
    hess_mm_fn = jax.jit(jax.hessian(_nll, argnums=1))

    cache = {
        "inner_vg": inner_vg,
        "grad_G": grad_G_fn,
        "hess_GG": hess_GG_fn,
        "hess_Gm": hess_Gm_fn,
        "hess_mm": hess_mm_fn,
    }
    _jit_cache[J] = cache
    return cache


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


def solve_market(
    market_data: MarketData,
    theta_G: np.ndarray,
    theta_m_init: np.ndarray,
    inner_maxiter: int = 200,
    inner_tol: float = 1e-6,
    compute_hessian: bool = True,
) -> MarketResult:
    """Solve inner problem for one market and return aggregated derivatives.

    Uses scipy.optimize.minimize(L-BFGS-B) with a JIT-compiled JAX callback
    for value+gradient.  This avoids jaxopt's XLA compilation issues while
    keeping each function evaluation fast through JAX JIT caching.

    Cached JIT functions are keyed on J.  Markets with different J values
    get separate compiled functions; different N values within the same J
    share the function but trigger separate JAX trace caches.
    """
    import jax.numpy as jnp
    from scipy.optimize import minimize

    J = market_data.J_per
    fns = _get_jitted_fns(J)

    # Convert to JAX arrays (zero-copy when already contiguous float64)
    X_m = jnp.asarray(market_data.X_m, dtype=jnp.float64)
    choice_m = jnp.asarray(market_data.choice_m, dtype=jnp.int32)
    D_m = jnp.asarray(market_data.D_m, dtype=jnp.float64)
    w_m = jnp.asarray(market_data.w_m, dtype=jnp.float64)
    Y_m = jnp.asarray(market_data.Y_m, dtype=jnp.float64)
    labor_m = jnp.asarray(market_data.labor_m, dtype=jnp.float64)
    mu_e = jnp.asarray(market_data.mu_e, dtype=jnp.float64)
    theta_G_jax = jnp.asarray(theta_G, dtype=jnp.float64)
    data_args = (X_m, choice_m, D_m, w_m, Y_m, labor_m, mu_e)

    # ---- Solve inner problem: L-BFGS-B over z_m with θ_G fixed ----
    theta_m_init_jax = jnp.asarray(theta_m_init, dtype=jnp.float64)
    z_m_init = _inner_inv(theta_m_init_jax, J)

    inner_vg_fn = fns["inner_vg"]
    _iter_count = [0]

    def scipy_callback(z_flat):
        """Callback for scipy: returns (value, gradient) as numpy float64."""
        z_jax = jnp.asarray(z_flat, dtype=jnp.float64)
        val, grad = inner_vg_fn(z_jax, theta_G_jax, *data_args)
        _iter_count[0] += 1
        return float(val), np.asarray(grad, dtype=np.float64)

    z0 = np.asarray(z_m_init, dtype=np.float64)
    scipy_result = minimize(
        scipy_callback,
        z0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": inner_maxiter, "gtol": inner_tol, "ftol": 1e-15},
    )
    z_m_hat = jnp.asarray(scipy_result.x, dtype=jnp.float64)
    inner_converged = scipy_result.success
    inner_iters = _iter_count[0]
    theta_m_hat = _inner_fwd(z_m_hat, J)

    # ---- NLL at optimum ----
    nll_m = float(_market_nll_pure(
        theta_G_jax, theta_m_hat, *data_args, J))

    # ---- Gradient w.r.t. θ_G ----
    grad_G = fns["grad_G"](theta_G_jax, theta_m_hat, *data_args)
    grad_G_np = np.asarray(grad_G, dtype=np.float64)

    # ---- Schur complement for outer Hessian ----
    if compute_hessian:
        H_GG = fns["hess_GG"](theta_G_jax, theta_m_hat, *data_args)
        H_Gm = fns["hess_Gm"](theta_G_jax, theta_m_hat, *data_args)
        H_mm = fns["hess_mm"](theta_G_jax, theta_m_hat, *data_args)

        H_mm_reg = H_mm + 1e-6 * jnp.eye(2 * J)
        x = jnp.linalg.solve(H_mm_reg, H_Gm.T)
        schur_m = H_GG - H_Gm @ x
        schur_m = jnp.where(jnp.isfinite(schur_m), schur_m, H_GG)
        schur_m_np = np.asarray(schur_m, dtype=np.float64)
    else:
        schur_m_np = np.zeros((5, 5), dtype=np.float64)

    return MarketResult(
        market_id=market_data.market_id,
        nll_m=nll_m,
        grad_G_m=grad_G_np,
        schur_m=schur_m_np,
        theta_m_hat=np.asarray(theta_m_hat, dtype=np.float64),
        inner_converged=inner_converged,
        inner_iters=inner_iters,
    )
