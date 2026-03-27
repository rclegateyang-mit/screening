#!/usr/bin/env python3
"""Worker logic for distributed MLE: inner optimization + Hessian extraction.

Pure computation — no MPI, no process pools.  Each MPI rank calls
``solve_market`` on its locally held ``MarketData`` objects.  Data stays
in-process for the lifetime of the rank (received once via ``comm.scatter``
at startup).

Global parameters: θ_G = [τ, γ̃] (tau and tilde_gamma).
Per-market parameters: z_m = [δ(J), q̃(J)] (delta and tilde_q).
Fixed parameters: α, σ_e, γ₀ are held constant (set at initialization).

JIT caching
-----------
Compiled functions are cached per J (number of firms) via ``_get_jitted_fns``.
J must be a compile-time constant because it controls parameter vector
slicing.  Markets with different N reuse the same compiled function —
JAX's internal trace cache handles shape variation automatically.
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
    X_m: np.ndarray          # (N_per,) observable skill v
    choice_m: np.ndarray     # (N_per,) int32, 0=outside option
    D_m: np.ndarray          # (N_per, J_per) distances
    w_m: np.ndarray          # (J_per,) wages
    Y_m: np.ndarray          # (J_per,) revenue
    labor_m: np.ndarray      # (J_per,) employment counts
    gamma0: float            # skill intercept (fixed)
    alpha: float             # production elasticity (fixed)
    sigma_e: float           # skill noise std dev (fixed)
    J_per: int
    N_per: int
    z1_m: np.ndarray = None  # (J_per,) observed TFP shifter (instrument)
    z2_m: np.ndarray = None  # (J_per,) observed amenity shifter (instrument)


@dataclass
class MarketResult:
    """Return value from each worker to the master."""
    market_id: int
    nll_m: float                # -ℓ_m at optimum
    grad_G_m: np.ndarray        # (2,) ∂(-ℓ_m)/∂θ_G
    schur_m: np.ndarray         # (2,2) Schur complement contribution
    theta_m_hat: np.ndarray     # (2*J_per,) optimized [δ_m, q̃_m]
    inner_converged: bool
    inner_iters: int


# ---------------------------------------------------------------------------
# JAX initialisation (call once per process, before any solve_market calls)
# ---------------------------------------------------------------------------


def init_jax():
    """Set up JAX with float64 and single-threaded XLA (one rank per core)."""
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                          "intra_op_parallelism_threads=1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import jax
    jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# JIT-compiled function cache
# ---------------------------------------------------------------------------

_jit_cache: Dict[int, dict] = {}
_jit_cache_delta_only: Dict[int, dict] = {}
_jit_cache_profile_delta: Dict[int, dict] = {}


def _market_nll_pure(tG, tM, X_m, choice_m, D_m, J):
    """Pure-function market NLL using model_components.

    tG = [tau, tilde_gamma]
    tM = [delta(J), tilde_q(J)]
    """
    import jax.numpy as jnp
    from screening.analysis.lib.model_components import choice_probabilities, per_obs_nll

    tau = tG[0]
    tilde_gamma = tG[1]
    delta = tM[:J]
    tilde_q = tM[J:]

    P_nat = choice_probabilities(tau, tilde_gamma, delta, tilde_q, X_m, D_m)
    return jnp.sum(per_obs_nll(P_nat, choice_m))


def _market_nll_delta_only(tG, delta, tilde_q_fixed, X_m, choice_m, D_m):
    """Market NLL with tilde_q fixed — optimize only delta."""
    import jax.numpy as jnp
    from screening.analysis.lib.model_components import choice_probabilities, per_obs_nll

    tau = tG[0]
    tilde_gamma = tG[1]
    P_nat = choice_probabilities(tau, tilde_gamma, delta, tilde_q_fixed, X_m, D_m)
    return jnp.sum(per_obs_nll(P_nat, choice_m))


def _get_jitted_fns(J: int) -> dict:
    """Return dict of JIT-compiled functions for the given J."""
    if J in _jit_cache:
        return _jit_cache[J]

    import jax

    def _nll(tG, tM, X_m, choice_m, D_m):
        return _market_nll_pure(tG, tM, X_m, choice_m, D_m, J)

    # Inner value-and-grad: z_m is optimized, rest are fixed args.
    def _inner_loss(z_m, tG, X_m, choice_m, D_m):
        return _nll(tG, z_m, X_m, choice_m, D_m)

    inner_vg = jax.jit(jax.value_and_grad(_inner_loss, argnums=0))

    # Gradient of NLL w.r.t. θ_G (first arg), with θ_m and data fixed
    grad_G_fn = jax.jit(jax.grad(_nll, argnums=0))

    # Hessian components for Schur complement
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


def _get_jitted_fns_delta_only(J: int) -> dict:
    """Return JIT-compiled functions for delta-only optimization (tilde_q fixed)."""
    if J in _jit_cache_delta_only:
        return _jit_cache_delta_only[J]

    import jax

    # delta is optimized; tilde_q_fixed is a fixed argument
    def _inner_loss(delta, tG, tilde_q_fixed, X_m, choice_m, D_m):
        return _market_nll_delta_only(tG, delta, tilde_q_fixed, X_m, choice_m, D_m)

    inner_vg = jax.jit(jax.value_and_grad(_inner_loss, argnums=0))

    cache = {"inner_vg": inner_vg}
    _jit_cache_delta_only[J] = cache
    return cache


def _get_jitted_fns_profile_delta(J: int) -> dict:
    """Return JIT-compiled functions for profile-delta mode.

    Optimizes over tilde_q only; delta is profiled out via contraction mapping
    and treated with stop_gradient (NFP approach).
    """
    if J in _jit_cache_profile_delta:
        return _jit_cache_profile_delta[J]

    import jax

    def _profiled_nll(tilde_q, delta_sg, tG, X_m, choice_m, D_m):
        """NLL with delta treated as non-differentiable (stop_gradient)."""
        import jax.numpy as jnp
        delta = jax.lax.stop_gradient(delta_sg)
        tM = jnp.concatenate([delta, tilde_q])
        return _market_nll_pure(tG, tM, X_m, choice_m, D_m, J)

    # Shares function for contraction mapping
    def _shares(tG, delta, tilde_q, X_m, D_m):
        import jax.numpy as jnp
        from screening.analysis.lib.model_components import choice_probabilities
        tau, tilde_gamma = tG[0], tG[1]
        P_nat = choice_probabilities(tau, tilde_gamma, delta, tilde_q, X_m, D_m)
        return jnp.mean(P_nat[:, 1:], axis=0)

    profiled_vg = jax.jit(jax.value_and_grad(_profiled_nll, argnums=0))
    shares_fn = jax.jit(_shares)

    cache = {"profiled_vg": profiled_vg, "shares_fn": shares_fn}
    _jit_cache_profile_delta[J] = cache
    return cache


def _solve_delta_contraction(
    shares_fn, theta_G_jax, tilde_q, X_m, D_m, shares_emp, delta_init,
    tol: float = 1e-2, maxiter: int = 1000,
):
    """Solve for delta via BLP contraction given tilde_q (Python loop)."""
    import jax.numpy as jnp

    log_s_data = jnp.log(jnp.maximum(shares_emp, 1e-300))
    delta = delta_init

    for it in range(maxiter):
        s_model = shares_fn(theta_G_jax, delta, tilde_q, X_m, D_m)
        log_s_model = jnp.log(jnp.maximum(s_model, 1e-300))
        diff = log_s_data - log_s_model
        delta = delta + diff
        err = float(jnp.max(jnp.abs(diff)))
        if err <= tol:
            break

    return delta, err, it + 1


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
    tilde_q_fixed: np.ndarray | None = None,
    profile_delta: bool = False,
    contraction_maxiter: int = 2000,
    warm_start: bool = False,
) -> MarketResult:
    """Solve inner problem for one market and return aggregated derivatives.

    theta_G: (2,) [tau, tilde_gamma]
    theta_m_init: (2*J,) [delta(J), tilde_q(J)]
    tilde_q_fixed: if provided (J,), fix tilde_q and only optimize delta
    profile_delta: if True, profile out delta via BLP contraction and
                   optimize only over tilde_q
    warm_start: if True, skip data-driven tq override (use theta_m_init as-is)
    """
    import jax.numpy as jnp
    from scipy.optimize import minimize

    J = market_data.J_per
    delta_only = tilde_q_fixed is not None

    # Convert to JAX arrays
    X_m = jnp.asarray(market_data.X_m, dtype=jnp.float64)
    choice_m = jnp.asarray(market_data.choice_m, dtype=jnp.int32)
    D_m = jnp.asarray(market_data.D_m, dtype=jnp.float64)
    theta_G_jax = jnp.asarray(theta_G, dtype=jnp.float64)

    _iter_count = [0]

    # ---- Data-driven tq init override ----
    # The naive init often places tq in a flat region where screening doesn't
    # bind (gradient ≈ 0).  Recompute tq_init from matched-worker skill
    # quantiles: tq_j ≈ tilde_gamma * v_low_j, where v_low_j is a low
    # quantile of skills among workers who chose firm j.  This places tq
    # in the region where screening is active and the gradient is nonzero.
    if not delta_only and not warm_start:
        tilde_gamma = float(theta_G[1])
        choice_np = np.asarray(market_data.choice_m, dtype=np.int32)
        v_np = np.asarray(market_data.X_m, dtype=np.float64)
        tq_override = np.empty(J, dtype=np.float64)
        for j in range(J):
            mask_j = choice_np == (j + 1)
            if np.any(mask_j):
                v_low = np.quantile(v_np[mask_j], 0.05)
                tq_override[j] = tilde_gamma * v_low
            else:
                tq_override[j] = theta_m_init[J + j]  # keep naive init
        theta_m_init = np.concatenate([theta_m_init[:J], tq_override])

    if profile_delta:
        # ---- Profile-delta mode: optimize tilde_q, contraction for delta ----
        fns_pd = _get_jitted_fns_profile_delta(J)
        profiled_vg_fn = fns_pd["profiled_vg"]
        shares_fn = fns_pd["shares_fn"]

        tq_init = jnp.asarray(theta_m_init[J:], dtype=jnp.float64)
        delta_warm = [jnp.asarray(theta_m_init[:J], dtype=jnp.float64)]

        # Empirical shares (with floor for zero-share firms)
        counts = np.bincount(market_data.choice_m, minlength=J + 1).astype(np.float64)
        shares_emp = counts[1:] / market_data.N_per
        shares_emp = np.maximum(shares_emp, 0.5 / market_data.N_per)
        shares_emp_jax = jnp.asarray(shares_emp, dtype=jnp.float64)

        # Data-driven contraction tolerance: smallest meaningful share increment
        # is 1/N (one worker switching firms), so tol = 1/N in log-share space
        contraction_tol = 1.0 / market_data.N_per

        def scipy_callback(tq_flat):
            tq_jax = jnp.asarray(tq_flat, dtype=jnp.float64)

            # Contraction: solve delta(tq) from current warm-start
            delta_solved, c_err, c_iters = _solve_delta_contraction(
                shares_fn, theta_G_jax, tq_jax, X_m, D_m,
                shares_emp_jax, delta_warm[0],
                tol=contraction_tol, maxiter=contraction_maxiter,
            )
            delta_warm[0] = delta_solved  # warm-start next eval

            # NLL and grad w.r.t. tilde_q (delta treated as constant via stop_gradient)
            val, grad = profiled_vg_fn(tq_jax, delta_solved, theta_G_jax,
                                       X_m, choice_m, D_m)
            _iter_count[0] += 1
            return float(val), np.asarray(grad, dtype=np.float64)

        tq0 = np.asarray(tq_init, dtype=np.float64)
        scipy_result = minimize(
            scipy_callback, tq0,
            method="L-BFGS-B", jac=True,
            options={"maxiter": inner_maxiter, "gtol": inner_tol,
                     "ftol": 1e-15, "maxcor": 50},
        )
        tq_hat = jnp.asarray(scipy_result.x, dtype=jnp.float64)

        # Final contraction at optimal tilde_q
        delta_hat, _, _ = _solve_delta_contraction(
            shares_fn, theta_G_jax, tq_hat, X_m, D_m,
            shares_emp_jax, delta_warm[0],
            tol=contraction_tol, maxiter=contraction_maxiter,
        )
        theta_m_hat = jnp.concatenate([delta_hat, tq_hat])

    elif delta_only:
        # ---- Delta-only mode: optimize delta with tilde_q fixed ----
        fns_do = _get_jitted_fns_delta_only(J)
        tq_jax = jnp.asarray(tilde_q_fixed, dtype=jnp.float64)
        inner_vg_fn = fns_do["inner_vg"]

        delta_init = jnp.asarray(theta_m_init[:J], dtype=jnp.float64)

        def scipy_callback(d_flat):
            d_jax = jnp.asarray(d_flat, dtype=jnp.float64)
            val, grad = inner_vg_fn(d_jax, theta_G_jax, tq_jax, X_m, choice_m, D_m)
            _iter_count[0] += 1
            return float(val), np.asarray(grad, dtype=np.float64)

        d0 = np.asarray(delta_init, dtype=np.float64)
        scipy_result = minimize(
            scipy_callback, d0,
            method="L-BFGS-B", jac=True,
            options={"maxiter": inner_maxiter, "gtol": inner_tol,
                     "ftol": 1e-15, "maxcor": 50},
        )
        delta_hat = jnp.asarray(scipy_result.x, dtype=jnp.float64)
        theta_m_hat = jnp.concatenate([delta_hat, tq_jax])
    else:
        # ---- Standard mode: optimize [delta, tilde_q] jointly ----
        fns = _get_jitted_fns(J)
        data_args = (X_m, choice_m, D_m)
        z_m_init = jnp.asarray(theta_m_init, dtype=jnp.float64)
        inner_vg_fn = fns["inner_vg"]

        def scipy_callback(z_flat):
            z_jax = jnp.asarray(z_flat, dtype=jnp.float64)
            val, grad = inner_vg_fn(z_jax, theta_G_jax, *data_args)
            _iter_count[0] += 1
            return float(val), np.asarray(grad, dtype=np.float64)

        z0 = np.asarray(z_m_init, dtype=np.float64)
        scipy_result = minimize(
            scipy_callback, z0,
            method="L-BFGS-B", jac=True,
            options={"maxiter": inner_maxiter, "gtol": inner_tol,
                     "ftol": 1e-15, "maxcor": 50},
        )
        theta_m_hat = jnp.asarray(scipy_result.x, dtype=jnp.float64)

    inner_converged = scipy_result.success
    inner_iters = _iter_count[0]

    # ---- NLL at optimum ----
    data_args = (X_m, choice_m, D_m)
    nll_m = float(_market_nll_pure(
        theta_G_jax, theta_m_hat, *data_args, J))

    # ---- Gradient w.r.t. θ_G and Schur complement ----
    if delta_only:
        # Delta-only mode: globals are fixed, skip gradient/Hessian
        grad_G_np = np.zeros(2, dtype=np.float64)
        schur_m_np = np.zeros((2, 2), dtype=np.float64)
    else:
        fns = _get_jitted_fns(J)
        grad_G = fns["grad_G"](theta_G_jax, theta_m_hat, *data_args)
        grad_G_np = np.asarray(grad_G, dtype=np.float64)

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
            schur_m_np = np.zeros((2, 2), dtype=np.float64)

    return MarketResult(
        market_id=market_data.market_id,
        nll_m=nll_m,
        grad_G_m=grad_G_np,
        schur_m=schur_m_np,
        theta_m_hat=np.asarray(theta_m_hat, dtype=np.float64),
        inner_converged=inner_converged,
        inner_iters=inner_iters,
    )
