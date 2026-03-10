#!/usr/bin/env python3
"""
Worker Screening Equilibrium Solver (Levenberg–Marquardt)

This version solves the system of firm first-order conditions F(x)=0 in the log
variables x=[log w, log c] using a Levenberg–Marquardt solver with an autodiff
Jacobian (JAX). Behavioral elasticities have been removed; conduct modes now
cover:
    1) conduct_mode == 1: status-quo wages as in lines 298–299 of the previous
       implementation (log w_j = log(1-β) + log Y_j - log L_j + log(α/(α+1)))
    2) conduct_mode == 2: wage FOC with endogenous elasticities
       (log w_j = log Y_j − log L_j + log(1-β) − log(1+e_j) + log(e_j + s_j)),
       where e_j, s_j are labor/skill elasticities to own wage.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import least_squares, root

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    JAX_AVAILABLE = False

# Optional numba import (used only for diagnostics)
try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

try:
    from .. import get_data_dir
except ImportError:  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from code import get_data_dir  # type: ignore


@dataclass
class EquilibriumData:
    A: np.ndarray
    xi: np.ndarray
    loc_firms: np.ndarray
    support_points: np.ndarray
    support_weights: np.ndarray
    mu_s: float
    sigma_s: float
    alpha: float
    beta: float
    gamma: float
    N_workers: float = 1.0
    mu_x_skill: Optional[float] = None
    sigma_x_skill: Optional[float] = None
    mu_a_skill: Optional[float] = None
    sigma_a_skill: Optional[float] = None
    worker_mu_x: Optional[float] = None
    worker_mu_y: Optional[float] = None
    worker_sigma_x: Optional[float] = None
    worker_sigma_y: Optional[float] = None
    worker_rho: Optional[float] = None
    rho_x_skill_ell_x: float = 0.0
    rho_x_skill_ell_y: float = 0.0
    rho_x_skill_r: float = 0.0
    worker_loc_mode: Optional[str] = None
    worker_r_mu: Optional[float] = None
    worker_r_sigma: Optional[float] = None
    mu_s_loc: Optional[np.ndarray] = None
    sigma_s_loc: Optional[np.ndarray] = None


# =============================================================================
# Core math helpers
# =============================================================================


def _safe_log(x: Any, eps: float, xp: Any) -> Any:
    """Numerically safe log usable with numpy or jax.numpy."""
    return xp.log(xp.maximum(x, eps))


def _get_skill_moments(data: EquilibriumData) -> Tuple[Any, Any]:
    if data.mu_s_loc is not None and data.sigma_s_loc is not None:
        return data.mu_s_loc, data.sigma_s_loc
    return data.mu_s, data.sigma_s


def _compute_location_skill_moments(
    support_points: np.ndarray,
    support_weights: Optional[np.ndarray],
    params: Dict[str, Any],
    eps: float = 1e-12,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    rho_x_skill_ell_x = float(params.get("rho_x_skill_ell_x", 0.0))
    rho_x_skill_ell_y = float(params.get("rho_x_skill_ell_y", 0.0))
    rho_x_skill_r = float(params.get("rho_x_skill_r", 0.0))
    worker_loc_mode = params.get("worker_loc_mode", "cartesian")

    if abs(rho_x_skill_r) > 0.0:
        if abs(rho_x_skill_ell_x) > 0.0 or abs(rho_x_skill_ell_y) > 0.0:
            raise ValueError("Set either rho_x_skill_r or rho_x_skill_ell_x/ell_y, not both.")
        if not (-1.0 <= rho_x_skill_r <= 1.0):
            raise ValueError("rho_x_skill_r must be in [-1, 1].")

        mu_x_skill = params.get("mu_x_skill")
        sigma_x_skill = params.get("sigma_x_skill")
        mu_a_skill = params.get("mu_a_skill", 0.0)
        sigma_a_skill = params.get("sigma_a_skill", 0.0)
        worker_mu_x = params.get("worker_mu_x")
        worker_mu_y = params.get("worker_mu_y")

        missing = [
            name
            for name, value in {
                "mu_x_skill": mu_x_skill,
                "sigma_x_skill": sigma_x_skill,
                "worker_mu_x": worker_mu_x,
                "worker_mu_y": worker_mu_y,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "Skill-radius correlation requested but missing parameters: "
                + ", ".join(missing)
            )

        mu_x_skill = float(mu_x_skill)
        sigma_x_skill = float(sigma_x_skill)
        mu_a_skill = float(mu_a_skill)
        sigma_a_skill = float(sigma_a_skill)
        worker_mu_x = float(worker_mu_x)
        worker_mu_y = float(worker_mu_y)

        if sigma_x_skill <= 0:
            raise ValueError("sigma_x_skill must be positive for skill-radius correlation.")

        r_vals = np.sqrt(
            (support_points[:, 0] - worker_mu_x) ** 2
            + (support_points[:, 1] - worker_mu_y) ** 2
        )
        if support_weights is None:
            weights = np.full(r_vals.shape[0], 1.0 / r_vals.shape[0])
        else:
            weights = np.asarray(support_weights, dtype=float)
            weights = weights / np.maximum(weights.sum(), eps)
        r_mean = np.sum(weights * r_vals)
        r_var = np.sum(weights * (r_vals - r_mean) ** 2)
        r_std = np.sqrt(np.maximum(r_var, eps))
        if r_std <= eps:
            raise ValueError("Worker radius has near-zero variance; cannot apply rho_x_skill_r.")

        r_std_vals = (r_vals - r_mean) / r_std
        mu_x_cond = mu_x_skill + rho_x_skill_r * sigma_x_skill * r_std_vals
        sigma_x_cond = sigma_x_skill * np.sqrt(np.maximum(1.0 - rho_x_skill_r**2, eps))

        mu_s_loc = mu_x_cond + mu_a_skill
        sigma_s_loc = np.full_like(
            mu_s_loc, np.sqrt(sigma_x_cond**2 + sigma_a_skill**2), dtype=float
        )
        return mu_s_loc, sigma_s_loc

    if worker_loc_mode == "polar" and (abs(rho_x_skill_ell_x) > 0.0 or abs(rho_x_skill_ell_y) > 0.0):
        raise ValueError("worker_loc_mode=polar does not support rho_x_skill_ell_x/ell_y; use rho_x_skill_r.")

    if abs(rho_x_skill_ell_x) <= 0.0 and abs(rho_x_skill_ell_y) <= 0.0:
        return None, None

    if not (-1.0 <= rho_x_skill_ell_x <= 1.0 and -1.0 <= rho_x_skill_ell_y <= 1.0):
        raise ValueError("rho_x_skill_ell_x and rho_x_skill_ell_y must be in [-1, 1].")

    mu_x_skill = params.get("mu_x_skill")
    sigma_x_skill = params.get("sigma_x_skill")
    mu_a_skill = params.get("mu_a_skill", 0.0)
    sigma_a_skill = params.get("sigma_a_skill", 0.0)
    worker_mu_x = params.get("worker_mu_x")
    worker_mu_y = params.get("worker_mu_y")
    worker_sigma_x = params.get("worker_sigma_x")
    worker_sigma_y = params.get("worker_sigma_y")
    worker_rho = params.get("worker_rho", 0.0)

    missing = [
        name
        for name, value in {
            "mu_x_skill": mu_x_skill,
            "sigma_x_skill": sigma_x_skill,
            "worker_mu_x": worker_mu_x,
            "worker_mu_y": worker_mu_y,
            "worker_sigma_x": worker_sigma_x,
            "worker_sigma_y": worker_sigma_y,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Skill-location correlation requested but missing parameters: "
            + ", ".join(missing)
        )

    mu_x_skill = float(mu_x_skill)
    sigma_x_skill = float(sigma_x_skill)
    mu_a_skill = float(mu_a_skill)
    sigma_a_skill = float(sigma_a_skill)
    worker_mu_x = float(worker_mu_x)
    worker_mu_y = float(worker_mu_y)
    worker_sigma_x = float(worker_sigma_x)
    worker_sigma_y = float(worker_sigma_y)
    worker_rho = float(worker_rho)

    if worker_sigma_x <= 0 or worker_sigma_y <= 0 or sigma_x_skill <= 0:
        raise ValueError("Worker and skill standard deviations must be positive.")

    cov_xy = worker_rho * worker_sigma_x * worker_sigma_y
    Sigma_ell = np.array(
        [
            [worker_sigma_x**2, cov_xy],
            [cov_xy, worker_sigma_y**2],
        ],
        dtype=float,
    )
    det = np.linalg.det(Sigma_ell)
    if det <= eps:
        raise ValueError("Worker location covariance is not positive definite.")
    Sigma_ell_inv = np.linalg.inv(Sigma_ell)

    cov_x_ell = np.array(
        [
            rho_x_skill_ell_x * sigma_x_skill * worker_sigma_x,
            rho_x_skill_ell_y * sigma_x_skill * worker_sigma_y,
        ],
        dtype=float,
    )
    beta = cov_x_ell @ Sigma_ell_inv
    var_x_cond = sigma_x_skill**2 - cov_x_ell @ Sigma_ell_inv @ cov_x_ell.T
    if var_x_cond <= eps:
        raise ValueError(
            "Implied conditional variance for x_skill is non-positive; "
            "check correlation settings."
        )

    delta = support_points - np.array([worker_mu_x, worker_mu_y], dtype=float)
    mu_x_cond = mu_x_skill + delta @ beta.T

    mu_s_loc = mu_x_cond + mu_a_skill
    sigma_s_loc = np.sqrt(np.maximum(var_x_cond + sigma_a_skill**2, eps))
    sigma_s_loc = np.full_like(mu_s_loc, sigma_s_loc, dtype=float)
    return mu_s_loc, sigma_s_loc


def _truncated_normal_column_terms_backend(
    c_sorted: Any, mu_s: float, sigma_s: float, eps: float, xp: Any, ndtr_fn: Any
) -> Tuple[Any, Any]:
    """
    Backend-agnostic truncated normal interval masses and conditional means.

    Args:
        c_sorted: Sorted cutoff costs (J,)
        mu_s: Mean of skill distribution
        sigma_s: Std deviation of skill distribution
        eps: Numerical floor
        xp: numpy-like module (np or jnp)
        ndtr_fn: Normal CDF compatible with xp
    """
    J = int(c_sorted.shape[0])

    c_pad = xp.concatenate(
        [xp.array([-xp.inf]), c_sorted, xp.array([xp.inf])], axis=0  # type: ignore[attr-defined]
    )
    z = (c_pad - mu_s) / sigma_s

    Phi_inner = ndtr_fn(z[1:-1]) if J > 0 else xp.array([], dtype=c_sorted.dtype)
    Phi = xp.concatenate([xp.array([0.0]), Phi_inner, xp.array([1.0])], axis=0)

    phi_vals = xp.concatenate(
        [
            xp.array([0.0]),
            xp.exp(-0.5 * z[1:-1] ** 2) / xp.sqrt(2.0 * xp.pi),
            xp.array([0.0]),
        ],
        axis=0,
    )

    DeltaF = Phi[1:] - Phi[:-1]
    DeltaF = xp.maximum(DeltaF, eps)
    DeltaF = DeltaF / xp.sum(DeltaF)

    M = mu_s + sigma_s * (phi_vals[:-1] - phi_vals[1:]) / xp.maximum(DeltaF, eps)
    M = xp.clip(M, mu_s - 20.0 * sigma_s, mu_s + 20.0 * sigma_s)
    return DeltaF, M


def truncated_normal_column_terms(
    c_sorted: np.ndarray, mu_s: float, sigma_s: float, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper for numpy backend."""
    return _truncated_normal_column_terms_backend(
        c_sorted, mu_s, sigma_s, eps, np, special.ndtr
    )


def jax_aggregate_over_locations(
    support_points: Any,
    weights: Any,
    w: Any,
    c: Any,
    xi: Any,
    loc: Any,
    alpha: float,
    gamma: float,
    mu_s: float,
    sigma_s: float,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """JAX-based aggregation with vectorized computations (differentiable)."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for jax_aggregate_over_locations.")

    order_idx = jnp.argsort(c)
    inv_order = jnp.argsort(order_idx)
    c_sorted = c[order_idx]
    w_sorted = w[order_idx]
    xi_sorted = xi[order_idx]
    loc_sorted = loc[order_idx]

    mu_s_arr = jnp.asarray(mu_s)
    sigma_s_arr = jnp.asarray(sigma_s)
    if mu_s_arr.ndim == 0 and sigma_s_arr.ndim == 0:
        DeltaF, M_conditional = _truncated_normal_column_terms_backend(
            c_sorted, mu_s_arr, sigma_s_arr, eps, jnp, jsp.special.ndtr
        )
        p = DeltaF
        m_raw = DeltaF * M_conditional
        m_sum = jnp.sum(m_raw)
        scale = jnp.where(jnp.abs(m_sum) > 1e-12, mu_s_arr / jnp.maximum(m_sum, eps), 0.0)
        m = jnp.concatenate([m_raw[:1], m_raw[1:] * scale])
    else:
        if mu_s_arr.shape[0] != support_points.shape[0] or sigma_s_arr.shape[0] != support_points.shape[0]:
            raise ValueError(
                "mu_s and sigma_s must match support_points when using location-dependent skills."
            )

        def _loc_terms(mu_loc: Any, sigma_loc: Any) -> Tuple[Any, Any]:
            return _truncated_normal_column_terms_backend(
                c_sorted, mu_loc, sigma_loc, eps, jnp, jsp.special.ndtr
            )

        DeltaF, M_conditional = jax.vmap(_loc_terms)(mu_s_arr, sigma_s_arr)
        p = DeltaF
        m_raw = DeltaF * M_conditional
        m_sum = jnp.sum(m_raw, axis=1)
        scale = jnp.where(
            jnp.abs(m_sum) > 1e-12, mu_s_arr / jnp.maximum(m_sum, eps), 0.0
        )
        m = jnp.concatenate([m_raw[:, :1], m_raw[:, 1:] * scale[:, None]], axis=1)

    # Distances and inclusive values for all support points
    distances = jnp.linalg.norm(
        support_points[:, None, :] - loc_sorted[None, :, :], axis=2
    )  # (S, J)
    log_w_sorted = _safe_log(w_sorted, eps, jnp)
    v_firms = jnp.exp(-gamma * distances + alpha * log_w_sorted + xi_sorted)  # (S, J)
    v = jnp.concatenate([jnp.ones((support_points.shape[0], 1)), v_firms], axis=1)  # (S, J+1)

    denom = jnp.cumsum(v, axis=1)
    q = p / jnp.maximum(denom, eps)
    r = m / jnp.maximum(denom, eps)
    Q = jnp.flip(jnp.cumsum(jnp.flip(q, axis=1), axis=1), axis=1)
    R = jnp.flip(jnp.cumsum(jnp.flip(r, axis=1), axis=1), axis=1)

    L_loc = v * Q
    M_loc = v * R

    weights_col = weights[:, None]
    L_byc = jnp.sum(weights_col * L_loc, axis=0)
    M_byc = jnp.sum(weights_col * M_loc, axis=0)
    S_byc = jnp.where(L_byc > eps, M_byc / L_byc, jnp.zeros_like(M_byc))

    L_firms_byc = L_byc[1:]
    M_firms_byc = M_byc[1:]
    S_firms_byc = S_byc[1:]

    L_firms_nat = L_firms_byc[inv_order]
    M_firms_nat = M_firms_byc[inv_order]
    S_firms_nat = S_firms_byc[inv_order]

    return {
        "L_byc": L_byc,
        "M_byc": M_byc,
        "S_byc": S_byc,
        "L_firms_nat": L_firms_nat,
        "M_firms_nat": M_firms_nat,
        "S_firms_nat": S_firms_nat,
        "order_idx": order_idx,
        "inv_order": inv_order,
    }


def jax_aggregate_noscreening(
    support_points: Any,
    weights: Any,
    w: Any,
    xi: Any,
    loc: Any,
    alpha: float,
    gamma: float,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    JAX aggregation when all workers have identical skill (no cutoffs).
    Returns labor shares by firm; skill is homogeneous and handled separately.
    """
    distances = jnp.linalg.norm(
        support_points[:, None, :] - loc[None, :, :], axis=2
    )  # (S, J)
    log_w = _safe_log(w, eps, jnp)
    v = jnp.exp(-gamma * distances + alpha * log_w + xi)  # (S, J)
    denom = 1.0 + jnp.sum(v, axis=1, keepdims=True)
    shares = v / denom
    L_share = jnp.sum(weights[:, None] * shares, axis=0)
    return {"L_share": L_share}


def _inclusive_values_at_loc_backend(
    ell: Any,
    w_sorted: Any,
    xi_sorted: Any,
    loc_sorted: Any,
    alpha: float,
    gamma: float,
    xp: Any,
    eps: float,
) -> Any:
    distances = xp.linalg.norm(ell[None, :] - loc_sorted, axis=1)
    log_w = _safe_log(w_sorted, eps, xp)
    v_firms = xp.exp(-gamma * distances + alpha * log_w + xi_sorted)
    return xp.concatenate([xp.array([1.0], dtype=w_sorted.dtype), v_firms], axis=0)


def _conditional_LM_at_loc_backend(
    v: Any, p: Any, m: Any, xp: Any, eps: float
) -> Tuple[Any, Any]:
    denom = xp.cumsum(v)
    q = p / xp.maximum(denom, eps)
    r = m / xp.maximum(denom, eps)
    Q = xp.flip(xp.cumsum(xp.flip(q)))
    R = xp.flip(xp.cumsum(xp.flip(r)))
    L_loc = v * Q
    M_loc = v * R
    return L_loc, M_loc


def _aggregate_over_locations_backend(
    support_points: Any,
    weights: Any,
    w: Any,
    c: Any,
    xi: Any,
    loc: Any,
    alpha: float,
    gamma: float,
    mu_s: float,
    sigma_s: float,
    eps: float,
    xp: Any,
    ndtr_fn: Any,
) -> Dict[str, Any]:
    """Shared aggregator that works with numpy or jax.numpy backends."""
    J = int(w.shape[0])
    order_idx = xp.argsort(c)
    inv_order = xp.argsort(order_idx)
    c_sorted = c[order_idx]
    w_sorted = w[order_idx]
    xi_sorted = xi[order_idx]
    loc_sorted = loc[order_idx]

    L_byc = xp.zeros(J + 1, dtype=w.dtype)
    M_byc = xp.zeros(J + 1, dtype=w.dtype)

    mu_s_arr = xp.asarray(mu_s)
    sigma_s_arr = xp.asarray(sigma_s)
    if mu_s_arr.ndim == 0 and sigma_s_arr.ndim == 0:
        DeltaF, M_conditional = _truncated_normal_column_terms_backend(
            c_sorted, mu_s_arr, sigma_s_arr, eps, xp, ndtr_fn
        )
        p = DeltaF
        m_raw = DeltaF * M_conditional
        m_sum = xp.sum(m_raw)
        scale = xp.where(xp.abs(m_sum) > 1e-12, mu_s_arr / xp.maximum(m_sum, eps), 0.0)
        m = xp.concatenate([m_raw[:1], m_raw[1:] * scale])
        for s in range(int(support_points.shape[0])):
            ell = support_points[s]
            omega = weights[s]
            v = _inclusive_values_at_loc_backend(
                ell, w_sorted, xi_sorted, loc_sorted, alpha, gamma, xp, eps
            )
            L_loc, M_loc = _conditional_LM_at_loc_backend(v, p, m, xp, eps)
            L_byc = L_byc + omega * L_loc
            M_byc = M_byc + omega * M_loc
    else:
        if mu_s_arr.shape[0] != support_points.shape[0] or sigma_s_arr.shape[0] != support_points.shape[0]:
            raise ValueError(
                "mu_s and sigma_s must match support_points when using location-dependent skills."
            )
        for s in range(int(support_points.shape[0])):
            ell = support_points[s]
            omega = weights[s]
            DeltaF, M_conditional = _truncated_normal_column_terms_backend(
                c_sorted, mu_s_arr[s], sigma_s_arr[s], eps, xp, ndtr_fn
            )
            p = DeltaF
            m_raw = DeltaF * M_conditional
            m_sum = xp.sum(m_raw)
            scale = xp.where(
                xp.abs(m_sum) > 1e-12, mu_s_arr[s] / xp.maximum(m_sum, eps), 0.0
            )
            m = xp.concatenate([m_raw[:1], m_raw[1:] * scale])
            v = _inclusive_values_at_loc_backend(
                ell, w_sorted, xi_sorted, loc_sorted, alpha, gamma, xp, eps
            )
            L_loc, M_loc = _conditional_LM_at_loc_backend(v, p, m, xp, eps)
            L_byc = L_byc + omega * L_loc
            M_byc = M_byc + omega * M_loc

    S_byc = xp.where(L_byc > eps, M_byc / L_byc, xp.zeros_like(M_byc))

    L_firms_byc = L_byc[1:]
    M_firms_byc = M_byc[1:]
    S_firms_byc = S_byc[1:]

    L_firms_nat = L_firms_byc[inv_order]
    M_firms_nat = M_firms_byc[inv_order]
    S_firms_nat = S_firms_byc[inv_order]

    return {
        "L_byc": L_byc,
        "M_byc": M_byc,
        "S_byc": S_byc,
        "L_firms_nat": L_firms_nat,
        "M_firms_nat": M_firms_nat,
        "S_firms_nat": S_firms_nat,
        "order_idx": order_idx,
        "inv_order": inv_order,
    }


def aggregate_over_locations(
    support_points: np.ndarray,
    weights: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    xi: np.ndarray,
    loc: np.ndarray,
    alpha: float,
    gamma: float,
    mu_s: float,
    sigma_s: float,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Numpy wrapper for the backend aggregator."""
    return _aggregate_over_locations_backend(
        support_points,
        weights,
        w,
        c,
        xi,
        loc,
        alpha,
        gamma,
        mu_s,
        sigma_s,
        eps,
        np,
        special.ndtr,
    )


def aggregate_noscreening(
    support_points: np.ndarray,
    weights: np.ndarray,
    w: np.ndarray,
    xi: np.ndarray,
    loc: np.ndarray,
    alpha: float,
    gamma: float,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Numpy aggregation for the no-screening case (homogeneous skills)."""
    distances = np.linalg.norm(support_points[:, None, :] - loc[None, :, :], axis=2)
    log_w = _safe_log(w, eps, np)
    v = np.exp(-gamma * distances + alpha * log_w + xi)
    denom = 1.0 + np.sum(v, axis=1, keepdims=True)
    shares = v / denom
    L_share = np.sum(weights[:, None] * shares, axis=0)
    return {"L_share": L_share}


def compute_equilibrium_objects(
    logw: np.ndarray,
    logc: np.ndarray,
    data: EquilibriumData,
    conduct_mode: int,
    eps: float,
) -> Dict[str, Any]:
    """Evaluate equilibrium objects given (logw, logc) without updating them."""
    w = np.exp(logw)
    c = np.exp(logc)

    mu_s_arg, sigma_s_arg = _get_skill_moments(data)
    agg = aggregate_over_locations(
        data.support_points,
        data.support_weights,
        w,
        c,
        data.xi,
        data.loc_firms,
        data.alpha,
        data.gamma,
        mu_s_arg,
        sigma_s_arg,
        eps=eps,
    )

    L_byc = agg["L_byc"]
    S_byc = agg["S_byc"]
    order_idx = agg["order_idx"]
    inv_order = agg["inv_order"]

    L_firms_byc = L_byc[1:]
    S_firms_byc = S_byc[1:]
    L_levels_byc = data.N_workers * L_firms_byc
    A_byc = data.A[order_idx]

    LS_term = np.maximum(L_levels_byc * S_firms_byc, eps)
    Y_byc = A_byc * (LS_term ** (1 - data.beta))
    Y_nat = Y_byc[inv_order]

    rank = np.argsort(order_idx)

    return {
        "w": w,
        "c": c,
        "L_share_nat": agg["L_firms_nat"],
        "M_nat": agg["M_firms_nat"],
        "S_nat": agg["S_firms_nat"],
        "L_levels_nat": data.N_workers * agg["L_firms_nat"],
        "Y_nat": Y_nat,
        "order_idx": order_idx,
        "inv_order": inv_order,
        "rank": rank,
        "L_byc": L_byc,
        "S_byc": S_byc,
        "Y_byc": Y_byc,
    }


def wage_elasticities_jax(
    logw: Any, logc: Any, data: EquilibriumData, eps: float = 1e-12
) -> Tuple[Any, Any]:
    """
    Compute (e_j, s_j) = elasticities of labor and skill with respect to own wage.
    e_j = d ln L_j / d ln w_j, s_j = d ln S_j / d ln w_j, holding c fixed.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for wage_elasticities_jax.")

    logw = jnp.asarray(logw)
    logc = jnp.asarray(logc)
    mu_s_arg, sigma_s_arg = _get_skill_moments(data)

    def _logL_logS(lw: Any) -> Tuple[Any, Any]:
        agg = jax_aggregate_over_locations(
            jnp.asarray(data.support_points),
            jnp.asarray(data.support_weights),
            jnp.exp(lw),
            jnp.exp(logc),
            jnp.asarray(data.xi),
            jnp.asarray(data.loc_firms),
            data.alpha,
            data.gamma,
            mu_s_arg,
            sigma_s_arg,
            eps=eps,
        )
        L_share_nat = agg["L_firms_nat"]
        S_nat = agg["S_firms_nat"]
        L_levels = data.N_workers * L_share_nat
        logL = jnp.log(jnp.maximum(L_levels, eps))
        logS = jnp.log(jnp.maximum(S_nat, eps))
        return logL, logS

    def _logL_only(lw: Any) -> Any:
        return _logL_logS(lw)[0]

    def _logS_only(lw: Any) -> Any:
        return _logL_logS(lw)[1]

    J_logL = jax.jacrev(_logL_only)(logw)
    J_logS = jax.jacrev(_logS_only)(logw)

    e = jnp.diag(J_logL)
    s = jnp.diag(J_logS)
    return e, s


def cutoff_elasticities_jax(
    logw: Any, logc: Any, data: EquilibriumData, eps: float = 1e-12
) -> Tuple[Any, Any]:
    """
    Compute d ln L_j / d ln c_j and d ln S_j / d ln c_j holding wages fixed.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for cutoff_elasticities_jax.")

    logw = jnp.asarray(logw)
    logc = jnp.asarray(logc)
    mu_s_arg, sigma_s_arg = _get_skill_moments(data)

    def _logL_logS(lc: Any) -> Tuple[Any, Any]:
        agg = jax_aggregate_over_locations(
            jnp.asarray(data.support_points),
            jnp.asarray(data.support_weights),
            jnp.exp(logw),
            jnp.exp(lc),
            jnp.asarray(data.xi),
            jnp.asarray(data.loc_firms),
            data.alpha,
            data.gamma,
            mu_s_arg,
            sigma_s_arg,
            eps=eps,
        )
        L_share_nat = agg["L_firms_nat"]
        S_nat = agg["S_firms_nat"]
        L_levels = data.N_workers * L_share_nat
        logL = jnp.log(jnp.maximum(L_levels, eps))
        logS = jnp.log(jnp.maximum(S_nat, eps))
        return logL, logS

    def _logL_only(lc: Any) -> Any:
        return _logL_logS(lc)[0]

    def _logS_only(lc: Any) -> Any:
        return _logL_logS(lc)[1]

    J_logL = jax.jacrev(_logL_only)(logc)
    J_logS = jax.jacrev(_logS_only)(logc)

    e_c = jnp.diag(J_logL)
    s_c = jnp.diag(J_logS)
    return e_c, s_c


def compute_wage_elasticities(
    logw: np.ndarray, logc: np.ndarray, data: EquilibriumData, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numpy wrapper for wage elasticities using JAX autodiff.
    """
    e_jax, s_jax = wage_elasticities_jax(logw, logc, data, eps)
    return np.asarray(e_jax, dtype=float), np.asarray(s_jax, dtype=float)


def compute_cutoff_elasticities(
    logw: np.ndarray, logc: np.ndarray, data: EquilibriumData, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numpy wrapper for cutoff elasticities using JAX autodiff.
    """
    e_c_jax, s_c_jax = cutoff_elasticities_jax(logw, logc, data, eps)
    return np.asarray(e_c_jax, dtype=float), np.asarray(s_c_jax, dtype=float)


def wage_elasticities_noscreening_jax(
    logw: Any, data: EquilibriumData, eps: float = 1e-12
) -> Any:
    """Labor supply elasticity e_j = d ln L_j / d ln w_j under no-screening."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for wage_elasticities_noscreening_jax.")

    logw = jnp.asarray(logw)

    def _logL(lw: Any) -> Any:
        agg = jax_aggregate_noscreening(
            jnp.asarray(data.support_points),
            jnp.asarray(data.support_weights),
            jnp.exp(lw),
            jnp.asarray(data.xi),
            jnp.asarray(data.loc_firms),
            data.alpha,
            data.gamma,
            eps=eps,
        )
        L_levels = data.N_workers * agg["L_share"]
        return jnp.log(jnp.maximum(L_levels, eps))

    J_logL = jax.jacrev(_logL)(logw)
    e = jnp.diag(J_logL)
    return e


def compute_wage_elasticities_noscreening(
    logw: np.ndarray, data: EquilibriumData, eps: float = 1e-12
) -> np.ndarray:
    """Numpy wrapper for no-screening labor elasticities."""
    e_jax = wage_elasticities_noscreening_jax(logw, data, eps)
    return np.asarray(e_jax, dtype=float)


# =============================================================================
# Residuals and solver
# =============================================================================


def equilibrium_residual_jax(
    logx: Any, data: EquilibriumData, conduct_mode: int, eps: float
) -> Any:
    """JAX-compatible residual F(logx) for LM solver."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for the autodiff Jacobian.")

    logx = jnp.asarray(logx)
    J = data.A.shape[0]
    logw = logx[:J]
    logc = logx[J:]
    w = jnp.exp(logw)
    c = jnp.exp(logc)

    mu_s_arg, sigma_s_arg = _get_skill_moments(data)
    agg = jax_aggregate_over_locations(
        jnp.asarray(data.support_points),
        jnp.asarray(data.support_weights),
        w,
        c,
        jnp.asarray(data.xi),
        jnp.asarray(data.loc_firms),
        data.alpha,
        data.gamma,
        mu_s_arg,
        sigma_s_arg,
        eps=eps,
    )

    L_byc = agg["L_byc"]
    S_byc = agg["S_byc"]
    order_idx = agg["order_idx"]
    inv_order = agg["inv_order"]

    L_firms_byc = L_byc[1:]
    S_firms_byc = S_byc[1:]
    L_levels_byc = data.N_workers * L_firms_byc
    A_byc = jnp.asarray(data.A)[order_idx]
    LS_term = jnp.maximum(L_levels_byc * S_firms_byc, eps)
    Y_byc = A_byc * (LS_term ** (1 - data.beta))

    if conduct_mode == 1:
        logw_target_byc = (
            jnp.log(1 - data.beta)
            + _safe_log(Y_byc, eps, jnp)
            - _safe_log(L_levels_byc, eps, jnp)
            + jnp.log(data.alpha / (data.alpha + 1))
        )
    elif conduct_mode == 2:
        e_nat, s_nat = wage_elasticities_jax(logw, logc, data, eps)
        e_byc = e_nat[order_idx]
        s_byc = s_nat[order_idx]
        logw_target_byc = (
            _safe_log(Y_byc, eps, jnp)
            - _safe_log(L_levels_byc, eps, jnp)
            + jnp.log(1 - data.beta)
            - _safe_log(1.0 + e_byc, eps, jnp)
            + _safe_log(e_byc + s_byc, eps, jnp)
        )
    else:
        raise ValueError("conduct_mode must be 1 or 2.")
    logc_target_byc = logw_target_byc - jnp.log(A_byc) - jnp.log(1 - data.beta)
    logc_target_byc = logc_target_byc + (data.beta / (1 - data.beta)) * (
        _safe_log(Y_byc, eps, jnp) - jnp.log(A_byc)
    )

    logw_current_byc = logw[order_idx]
    logc_current_byc = logc[order_idx]

    resid_w = (logw_current_byc - logw_target_byc)[inv_order]
    resid_c = (logc_current_byc - logc_target_byc)[inv_order]
    return jnp.concatenate([resid_w, resid_c])


def equilibrium_residual_noscreening_jax(
    logw: Any, data: EquilibriumData, eps: float, mode: str = "homogeneous"
) -> Any:
    """Residuals for no-screening case (only wages)."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for the autodiff Jacobian.")

    logw = jnp.asarray(logw)
    w = jnp.exp(logw)
    mu_s_arg, sigma_s_arg = _get_skill_moments(data)

    if mode == "heterogeneous_acceptall":
        # Use heterogeneous skills with c = -inf (accept all workers)
        # This requires the full screening aggregation with very low cutoffs
        c_very_low = jnp.full_like(w, -1e10)  # Effectively -infinity
        
        agg = jax_aggregate_over_locations(
            jnp.asarray(data.support_points),
            jnp.asarray(data.support_weights),
            w,
            c_very_low,
            jnp.asarray(data.xi),
            jnp.asarray(data.loc_firms),
            data.alpha,
            data.gamma,
            mu_s_arg,
            sigma_s_arg,
            eps=eps,
        )
        L_share = agg["L_firms_nat"]
        S_nat = agg["S_firms_nat"]
        L_levels = data.N_workers * L_share
        
        # Compute elasticities using the screening aggregation with fixed cutoffs
        # We need elasticities with respect to wage holding c fixed at -inf
        logc_fixed = jnp.log(jnp.maximum(c_very_low, eps))
        e, s = wage_elasticities_jax(logw, logc_fixed, data, eps)
        
        LS_term = jnp.maximum(L_levels * S_nat, eps)
        Y = jnp.asarray(data.A) * (LS_term ** (1 - data.beta))
        
        # Use conduct_mode 2 wage FOC with elasticities
        logw_target = (
            _safe_log(Y, eps, jnp)
            - _safe_log(L_levels, eps, jnp)
            + jnp.log(1 - data.beta)
            - _safe_log(1.0 + e, eps, jnp)
            + _safe_log(e + s, eps, jnp)
        )
    else:
        # mode == "homogeneous": use population average skill
        agg = jax_aggregate_noscreening(
            jnp.asarray(data.support_points),
            jnp.asarray(data.support_weights),
            w,
            jnp.asarray(data.xi),
            jnp.asarray(data.loc_firms),
            data.alpha,
            data.gamma,
            eps=eps,
        )
        L_share = agg["L_share"]
        L_levels = data.N_workers * L_share
        # Homogeneous skill = mean skill μ_s
        LS_term = jnp.maximum(L_levels * data.mu_s, eps)
        Y = jnp.asarray(data.A) * (LS_term ** (1 - data.beta))

        e = wage_elasticities_noscreening_jax(logw, data, eps)
        # Skill elasticity is zero when skills are homogeneous
        s = jnp.zeros_like(e)
        logw_target = (
            _safe_log(Y, eps, jnp)
            - _safe_log(L_levels, eps, jnp)
            + jnp.log(1 - data.beta)
            - _safe_log(1.0 + e, eps, jnp)
            + _safe_log(e + s, eps, jnp)
        )
    
    return logw - logw_target


def compute_equilibrium_objects_noscreening(
    logw: np.ndarray, data: EquilibriumData, eps: float, mode: str = "homogeneous"
) -> Dict[str, Any]:
    """Compute equilibrium objects for no-screening."""
    w = np.exp(logw)
    
    if mode == "heterogeneous_acceptall":
        # Use heterogeneous skills with c = -inf (accept all workers)
        c_very_low = np.full_like(w, -1e10)
        logc_fixed = np.log(np.maximum(c_very_low, eps))

        mu_s_arg, sigma_s_arg = _get_skill_moments(data)
        agg = aggregate_over_locations(
            data.support_points,
            data.support_weights,
            w,
            c_very_low,
            data.xi,
            data.loc_firms,
            data.alpha,
            data.gamma,
            mu_s_arg,
            sigma_s_arg,
            eps=eps,
        )
        L_share = agg["L_firms_nat"]
        S_nat = agg["S_firms_nat"]
        L_levels = data.N_workers * L_share
        
        # Compute elasticities with c fixed at -inf
        e, s = compute_wage_elasticities(logw, logc_fixed, data, eps)
        
        LS_term = np.maximum(L_levels * S_nat, eps)
        Y = data.A * (LS_term ** (1 - data.beta))
        
        return {
            "w": w,
            "L": L_levels,
            "L_share": L_share,
            "S": S_nat,
            "Y": Y,
            "e": e,
            "s": s,
        }
    else:
        # mode == "homogeneous": use population average skill
        agg = aggregate_noscreening(
            data.support_points,
            data.support_weights,
            w,
            data.xi,
            data.loc_firms,
            data.alpha,
            data.gamma,
            eps=eps,
        )
        L_share = agg["L_share"]
        L_levels = data.N_workers * L_share
        LS_term = np.maximum(L_levels * data.mu_s, eps)
        Y = data.A * (LS_term ** (1 - data.beta))
        e = compute_wage_elasticities_noscreening(logw, data, eps)
        s = np.zeros_like(e)
        return {
            "w": w,
            "L": L_levels,
            "L_share": L_share,
            "S": np.full_like(w, data.mu_s, dtype=float),
            "Y": Y,
            "e": e,
            "s": s,
        }


def compute_profit_at_logs(
    logw: np.ndarray, logc: np.ndarray, data: EquilibriumData, eps: float = 1e-12
) -> np.ndarray:
    """Compute firm profits given log wage/cutoff vectors, holding others fixed."""
    w = np.exp(logw)
    c = np.exp(logc)
    mu_s_arg, sigma_s_arg = _get_skill_moments(data)
    agg = aggregate_over_locations(
        data.support_points,
        data.support_weights,
        w,
        c,
        data.xi,
        data.loc_firms,
        data.alpha,
        data.gamma,
        mu_s_arg,
        sigma_s_arg,
        eps=eps,
    )
    L_share_nat = np.maximum(agg["L_firms_nat"], eps)
    S_nat = np.maximum(agg["S_firms_nat"], eps)
    L_levels = data.N_workers * L_share_nat
    LS_term = np.maximum(L_levels * S_nat, eps)
    Y = data.A * (LS_term ** (1 - data.beta))
    profit = Y - w * L_levels
    return profit


def plot_profit_surface_fixed_others(
    logw_eq: np.ndarray,
    logc_eq: np.ndarray,
    data: EquilibriumData,
    j: int,
    out_path: Path,
    grid_n: int = 100,
    grid_log_span: float = 0.5,
    eps: float = 1e-12,
    br_logw: Optional[np.ndarray] = None,
    br_logc: Optional[np.ndarray] = None,
) -> None:
    """Plot firm j profit surface over (log w_j, log c_j) holding other firms fixed."""
    grid_n_int = int(grid_n)
    logw_grid = np.linspace(logw_eq[j] - grid_log_span, logw_eq[j] + grid_log_span, grid_n_int)
    logc_grid = np.linspace(logc_eq[j] - grid_log_span, logc_eq[j] + grid_log_span, grid_n_int)
    W, C = np.meshgrid(np.exp(logw_grid), np.exp(logc_grid))
    Pi = np.zeros_like(W)

    for i in range(grid_n_int):
        for k in range(grid_n_int):
            logw_mod = logw_eq.copy()
            logc_mod = logc_eq.copy()
            logw_mod[j] = logw_grid[k]
            logc_mod[j] = logc_grid[i]
            profits = compute_profit_at_logs(logw_mod, logc_mod, data, eps)
            Pi[i, k] = profits[j]

    plt.figure(figsize=(8, 6))
    pi_min = float(np.nanmin(Pi))
    pi_max = float(np.nanmax(Pi))
    span = max(pi_max - pi_min, 1e-12)
    base_levels = np.linspace(pi_min, pi_max, 30)
    top_levels = np.linspace(pi_max - 0.2 * span, pi_max, 20)
    levels = np.unique(np.concatenate([base_levels, top_levels]))
    contour = plt.contour(W, C, Pi, levels=levels, colors="k", linewidths=0.8)
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.2f")
    plt.plot(np.exp(logw_eq[j]), np.exp(logc_eq[j]), "ro", markersize=8, label="Equilibrium (FOC)")
    if br_logw is not None and br_logc is not None:
        plt.plot(np.exp(br_logw[j]), np.exp(br_logc[j]), "bx", markersize=10, label="Profit max (dA=0)")
    plt.xlabel("Wage $w_j$")
    plt.ylabel("Cutoff $c_j$")
    plt.title(f"Profit Contours (others fixed) for Firm {j+1}")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def solve_equilibrium_lm(
    data: EquilibriumData,
    conduct_mode: int = 1,
    max_iter: int = 5000,
    tol: float = 1e-10,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Solve F(logx)=0 using Levenberg–Marquardt with an autodiff Jacobian.
    """
    if conduct_mode not in (1, 2):
        raise ValueError("conduct_mode must be 1 or 2.")
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for the LM solver with autodiff Jacobian. Install jax to proceed."
        )

    J = data.A.shape[0]
    logw_init = np.log(1 - data.beta) + np.log(data.A) + np.log(data.alpha / (data.alpha + 1)) + 2.0
    # Initial guess for cutoffs: use firm-specific variation based on A and xi
    # This breaks symmetry and gives the solver a better starting point
    logc_init = np.log(np.maximum(data.mu_s, 1e-3)) + 0.3 * np.log(data.A) + 0.1 * data.xi
    x0 = np.concatenate([logw_init, logc_init])

    def residual_np(logx: np.ndarray) -> np.ndarray:
        return np.asarray(equilibrium_residual_jax(logx, data, conduct_mode, eps), dtype=float)

    jacobian = jax.jacfwd(lambda z: equilibrium_residual_jax(z, data, conduct_mode, eps))

    def jac_np(logx: np.ndarray) -> np.ndarray:
        return np.asarray(jacobian(jnp.asarray(logx)), dtype=float)

    start_time = time.time()
    
    # Print initial residual for diagnostics
    init_resid = residual_np(x0)
    print(f"Initial residual (max abs): {np.max(np.abs(init_resid)):.2e}")
    print(f"Initial residual (mean abs): {np.mean(np.abs(init_resid)):.2e}")
    
    result = least_squares(
        residual_np,
        x0,
        jac=jac_np,
        method="trf",  # Trust Region Reflective - more robust than LM
        xtol=tol,
        ftol=tol,
        gtol=tol,
        max_nfev=max_iter,
        verbose=0,
    )
    total_time = time.time() - start_time

    logw = result.x[:J]
    logc = result.x[J:]
    diagnostics = compute_equilibrium_objects(logw, logc, data, conduct_mode, eps)
    final_resid = residual_np(result.x)
    residual_max = float(np.max(np.abs(final_resid)))
    
    # Proper convergence check: residual must be small, not just solver status
    converged = result.success and (residual_max < tol)
    
    if not converged:
        resid_w = final_resid[:J]
        resid_c = final_resid[J:]
        print(f"\nWarning: Solver did not converge properly!")
        print(f"  Solver status: {result.status}, message: {result.message}")
        print(f"  Final residual (max abs): {residual_max:.2e} (should be < {tol:.1e})")
        print(f"  Wage residuals (max abs): {np.max(np.abs(resid_w)):.2e}")
        print(f"  Cutoff residuals (max abs): {np.max(np.abs(resid_c)):.2e}")
        print(f"  This means the equilibrium FOCs are NOT satisfied.")

    return {
        "w": diagnostics["w"],
        "c": diagnostics["c"],
        "logw": logw,
        "logc": logc,
        "L": diagnostics["L_levels_nat"],
        "S": diagnostics["S_nat"],
        "Y": diagnostics["Y_nat"],
        "rank": diagnostics["rank"],
        "diagnostics": diagnostics,
        "iters": result.nfev,
        "converged": converged,
        "status": result.status,
        "message": result.message,
        "residual": residual_max,
        "timing": {"total_time": total_time},
        "solver_result": result,
    }


def solve_equilibrium_root(
    data: EquilibriumData,
    conduct_mode: int = 1,
    max_iter: int = 5000,
    tol: float = 1e-10,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Solve F(logx)=0 using scipy.optimize.root (hybr) with an autodiff Jacobian.
    """
    if conduct_mode not in (1, 2):
        raise ValueError("conduct_mode must be 1 or 2.")
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for the root solver with autodiff Jacobian. Install jax to proceed."
        )

    J = data.A.shape[0]
    logw_init = np.log(1 - data.beta) + np.log(data.A) + np.log(data.alpha / (data.alpha + 1)) + 2.0
    logc_init = np.log(np.maximum(data.mu_s, 1e-3)) + 0.3 * np.log(data.A) + 0.1 * data.xi
    x0 = np.concatenate([logw_init, logc_init])

    def residual_np(logx: np.ndarray) -> np.ndarray:
        return np.asarray(equilibrium_residual_jax(logx, data, conduct_mode, eps), dtype=float)

    jacobian = jax.jacfwd(lambda z: equilibrium_residual_jax(z, data, conduct_mode, eps))

    def jac_np(logx: np.ndarray) -> np.ndarray:
        return np.asarray(jacobian(jnp.asarray(logx)), dtype=float)

    result = root(residual_np, x0, jac=jac_np, method="hybr", tol=tol, options={"maxfev": max_iter})

    logw = result.x[:J]
    logc = result.x[J:]
    diagnostics = compute_equilibrium_objects(logw, logc, data, conduct_mode, eps)
    residual_vals = residual_np(result.x)
    residual_max = float(np.max(np.abs(residual_vals)))
    converged = result.success and residual_max < tol

    if not converged:
        resid_w = residual_vals[:J]
        resid_c = residual_vals[J:]
        print(f"\nWarning: root solver did not converge properly!")
        print(f"  Solver status: {result.status}, message: {result.message}")
        print(f"  Final residual (max abs): {residual_max:.2e} (should be < {tol:.1e})")
        print(f"  Wage residuals (max abs): {np.max(np.abs(resid_w)):.2e}")
        print(f"  Cutoff residuals (max abs): {np.max(np.abs(resid_c)):.2e}")
        print(f"  This means the equilibrium FOCs may NOT be satisfied.")

    return {
        "w": diagnostics["w"],
        "c": diagnostics["c"],
        "logw": logw,
        "logc": logc,
        "L": diagnostics["L_levels_nat"],
        "S": diagnostics["S_nat"],
        "Y": diagnostics["Y_nat"],
        "rank": diagnostics["rank"],
        "diagnostics": diagnostics,
        "iters": result.nfev if hasattr(result, "nfev") else None,
        "converged": converged,
        "status": result.status,
        "message": result.message,
        "residual": residual_max,
        "solver_result": result,
    }


def solve_equilibrium_noscreening_lm(
    data: EquilibriumData,
    max_iter: int = 500,
    tol: float = 1e-8,
    eps: float = 1e-12,
    mode: str = "homogeneous",
) -> Dict[str, Any]:
    """Solve no-screening equilibrium (only wages) via LM with autodiff Jacobian."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for the LM solver with autodiff Jacobian. Install jax to proceed.")

    J = data.A.shape[0]
    logw_init = np.log(1 - data.beta) + np.log(data.A) + np.log(data.alpha / (data.alpha + 1)) + 2.0

    def residual_np(logw: np.ndarray) -> np.ndarray:
        return np.asarray(
            equilibrium_residual_noscreening_jax(logw, data, eps, mode=mode),
            dtype=float,
        )

    jacobian = jax.jacfwd(lambda z: equilibrium_residual_noscreening_jax(z, data, eps, mode=mode))

    def jac_np(logw: np.ndarray) -> np.ndarray:
        return np.asarray(jacobian(jnp.asarray(logw)), dtype=float)

    start_time = time.time()
    result = least_squares(
        residual_np,
        logw_init,
        jac=jac_np,
        method="lm",
        xtol=tol,
        ftol=tol,
        gtol=tol,
        max_nfev=max_iter,
    )
    total_time = time.time() - start_time

    logw = result.x
    diagnostics = compute_equilibrium_objects_noscreening(logw, data, eps, mode=mode)
    residual_max = float(np.max(np.abs(residual_np(result.x))))

    return {
        "w": diagnostics["w"],
        "L": diagnostics["L"],
        "L_share": diagnostics["L_share"],
        "S": diagnostics["S"],
        "Y": diagnostics["Y"],
        "e": diagnostics["e"],
        "s": diagnostics["s"],
        "logw": logw,
        "iters": result.nfev,
        "converged": result.success,
        "status": result.status,
        "message": result.message,
        "residual": residual_max,
        "timing": {"total_time": total_time},
    }


# =============================================================================
# I/O helpers
# =============================================================================


def read_firms_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read firm data from CSV.
    """
    df = pd.read_csv(path)

    required_cols = ["firm_id", "logA", "A", "xi", "comp", "x", "y"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.sort_values("firm_id").reset_index(drop=True)

    firm_id = df["firm_id"].values
    A = df["A"].values
    xi = df["xi"].values
    loc_firms = df[["x", "y"]].values
    comp = df["comp"].values

    if "logA" in df.columns:
        logA = df["logA"].values
        max_rel_error = np.max(np.abs(A - np.exp(logA)) / np.maximum(A, 1e-12))
        if max_rel_error > 1e-8:
            print(f"Warning: A and logA inconsistent, max relative error: {max_rel_error:.2e}")

    return A, xi, loc_firms, firm_id, comp


def read_support_points_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read support points from CSV.
    """
    df = pd.read_csv(path)

    required_cols = ["x", "y", "weight"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    support_points = df[["x", "y"]].values
    support_weights = df["weight"].values

    if not np.all(support_weights >= 0):
        raise ValueError("Support weights must be non-negative")

    weight_sum = support_weights.sum()
    if not np.isclose(weight_sum, 1.0, atol=1e-12):
        print(f"Warning: Normalizing support weights from {weight_sum:.12f} to 1.0")
        support_weights = support_weights / weight_sum

    return support_points, support_weights


def read_parameters_csv(path: str) -> Dict[str, Any]:
    """
    Read parameters from CSV.
    """
    df = pd.read_csv(path)

    required_cols = ["parameter", "value"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    params: Dict[str, Any] = {}
    for _, row in df.iterrows():
        name = row["parameter"]
        value = row["value"]
        try:
            if name in ["conduct_mode", "max_iter", "max_plots", "quad_n_x", "quad_n_y", "seed"]:
                params[name] = int(value)
            elif name in ["quad_normalize", "quad_write_csv"]:
                params[name] = bool(value)
            elif name in ["quad_kind", "worker_loc_mode"]:
                params[name] = str(value)
            else:
                params[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid value for parameter {name}: {value}")
    return params


def write_equilibrium_csv(
    path: str,
    firm_id: np.ndarray,
    comp: np.ndarray,
    A: np.ndarray,
    xi: np.ndarray,
    loc_firms: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    L: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    logw: np.ndarray,
    logc: np.ndarray,
    rank: np.ndarray,
    firm_id_original: Optional[np.ndarray] = None,
    L_wage_elasticity: Optional[np.ndarray] = None,
    S_wage_elasticity: Optional[np.ndarray] = None,
    L_cutoff_elasticity: Optional[np.ndarray] = None,
    S_cutoff_elasticity: Optional[np.ndarray] = None,
    L_tfp_elasticity: Optional[np.ndarray] = None,
    w_tfp_elasticity: Optional[np.ndarray] = None,
    c_tfp_elasticity: Optional[np.ndarray] = None,
) -> str:
    """
    Write equilibrium results to CSV.
    """
    data = {
        "firm_id": firm_id,
        "w": w,
        "c": c,
        "L": L,
        "S": S,
        "Y": Y,
        "logw": logw,
        "logc": logc,
        "rank": rank,
        "A": A,
        "xi": xi,
        "x": loc_firms[:, 0],
        "y": loc_firms[:, 1],
        "comp": comp,
    }
    if firm_id_original is not None:
        data["firm_id_original"] = firm_id_original
    if L_wage_elasticity is not None:
        data["elastic_L_w_w"] = L_wage_elasticity
    if S_wage_elasticity is not None:
        data["elastic_S_w_w"] = S_wage_elasticity
    if L_tfp_elasticity is not None:
        data["elastic_L_A_A"] = L_tfp_elasticity
    if w_tfp_elasticity is not None:
        data["elastic_w_A_A"] = w_tfp_elasticity
    if c_tfp_elasticity is not None:
        data["elastic_c_A_A"] = c_tfp_elasticity
    if L_cutoff_elasticity is not None:
        data["elastic_L_c_c"] = L_cutoff_elasticity
    if S_cutoff_elasticity is not None:
        data["elastic_S_c_c"] = S_cutoff_elasticity

    df = pd.DataFrame(data)
    if firm_id_original is not None:
        cols = ["firm_id", "firm_id_original"] + [
            col for col in df.columns if col not in {"firm_id", "firm_id_original"}
        ]
        df = df[cols]

    df.to_csv(path, index=False)
    return path


def write_equilibrium_csv_noscreening(
    path: str,
    firm_id: np.ndarray,
    comp: np.ndarray,
    A: np.ndarray,
    xi: np.ndarray,
    loc_firms: np.ndarray,
    w: np.ndarray,
    L: np.ndarray,
    L_share: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    logw: np.ndarray,
    e: np.ndarray,
    s: np.ndarray,
    L_tfp_elasticity: Optional[np.ndarray] = None,
    w_tfp_elasticity: Optional[np.ndarray] = None,
    L_wage_elasticity_fd: Optional[np.ndarray] = None,
    S_wage_elasticity_fd: Optional[np.ndarray] = None,
    firm_id_original: Optional[np.ndarray] = None,
) -> str:
    """Write equilibrium results for the no-screening model with labeled columns."""
    data = {
        "firm_id": firm_id,
        "w_noscreening": w,
        "logw_noscreening": logw,
        "L_noscreening": L,
        "L_share_noscreening": L_share,
        "S_noscreening": S,
        "Y_noscreening": Y,
        "elastic_L_w_noscreening": e,
        "elastic_S_w_noscreening": s,
        "A": A,
        "xi": xi,
        "x": loc_firms[:, 0],
        "y": loc_firms[:, 1],
        "comp": comp,
    }
    if L_tfp_elasticity is not None:
        data["elastic_L_A_A_noscreening"] = L_tfp_elasticity
    if w_tfp_elasticity is not None:
        data["elastic_w_A_A_noscreening"] = w_tfp_elasticity
    if L_wage_elasticity_fd is not None:
        data["elastic_L_w_noscreening_fd"] = L_wage_elasticity_fd
    if S_wage_elasticity_fd is not None:
        data["elastic_S_w_noscreening_fd"] = S_wage_elasticity_fd
    if firm_id_original is not None:
        data["firm_id_original"] = firm_id_original

    df = pd.DataFrame(data)
    if firm_id_original is not None:
        cols = ["firm_id", "firm_id_original"] + [
            col for col in df.columns if col not in {"firm_id", "firm_id_original"}
        ]
        df = df[cols]
    df.to_csv(path, index=False)
    return path


# =============================================================================
# Main execution
# =============================================================================


def main() -> int:
    data_dir = get_data_dir(create=True)
    parser = argparse.ArgumentParser(description="Worker Screening Equilibrium Solver (LM)")

    parser.add_argument(
        "--firms_path",
        type=str,
        default=str(data_dir / "firms.csv"),
        help="Path to firms CSV file",
    )
    parser.add_argument(
        "--support_path",
        type=str,
        default=str(data_dir / "support_points.csv"),
        help="Path to support points CSV file",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default=str(data_dir / "parameters_effective.csv"),
        help="Path to parameters CSV file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(data_dir),
        help="Output directory for equilibrium data (defaults to project data/ folder)",
    )

    parser.add_argument(
        "--conduct_mode",
        type=int,
        choices=[1, 2],
        help="1=status quo; 2=elasticity-based wage FOC",
    )
    parser.add_argument(
        "--noscreening",
        action="store_true",
        help="Solve equilibrium without screening (homogeneous skills, no cutoffs).",
    )
    parser.add_argument(
        "--noscreening_mode",
        type=str,
        choices=["homogeneous", "heterogeneous_acceptall"],
        default="homogeneous",
        help="Variant of the no-screening model: homogeneous skill (default) or heterogeneous skill with cutoffs forced to -inf (accept all).",
    )
    parser.add_argument(
        "--plot_profits_fixed",
        action="store_true",
        help="Plot profit surfaces for each firm over (w_j, c_j) holding others fixed.",
    )
    parser.add_argument("--profit_grid_n", type=int, help="Grid resolution for profit surfaces (default=40)")
    parser.add_argument(
        "--profit_grid_log_span",
        type=float,
        help="Log span around equilibrium (default=0.5) for profit surface plots.",
    )
    parser.add_argument(
        "--use_lsq",
        action="store_true",
        help="Use nonlinear least squares (trf) instead of the default root finder (hybr).",
    )
    parser.add_argument("--max_iter", type=int, help="Maximum LM iterations")
    parser.add_argument("--tol", type=float, help="Convergence tolerance")
    parser.add_argument("--eps", type=float, help="Numerical safety floor")
    parser.add_argument(
        "--drop_share_below",
        type=float,
        default=None,
        help="Drop firms whose worker share (L/N_workers) falls below this threshold; outside option retained implicitly.",
    )

    args = parser.parse_args()

    print("Worker Screening Equilibrium Solver (LM)")
    print("=" * 50)

    firms_path = Path(args.firms_path)
    support_path = Path(args.support_path)
    params_path = Path(args.params_path)
    out_dir = Path(args.out_dir)

    try:
        A, xi, loc_firms, firm_id, comp = read_firms_csv(firms_path)
        firm_id_original = firm_id.copy()
        support_points, support_weights = read_support_points_csv(support_path)
        params = read_parameters_csv(params_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required CSV files exist:")
        print(f"  - {firms_path}")
        print(f"  - {support_path}")
        print(f"  - {params_path}")
        return 1

    print(f"Loaded {len(A)} firms and {len(support_points)} support points")
    print(f"Support weight sum: {support_weights.sum():.12f}")

    mu_s = params["mu_s"]
    sigma_s = params["sigma_s"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]

    N_workers = float(params.get("N_workers", 1.0))
    if not np.isfinite(N_workers) or N_workers <= 0:
        raise ValueError(f"N_workers must be positive; got {N_workers}")

    support_counts = np.rint(support_weights * N_workers)
    positive_mask = support_counts > 0
    support_points = support_points[positive_mask]
    support_counts = support_counts[positive_mask]
    total_counts = support_counts.sum()
    if total_counts <= 0:
        raise ValueError("Rounded support weights sum to zero; adjust N_workers or quadrature configuration.")
    support_weights = support_counts / total_counts
    print(
        f"After rounding: {len(support_weights)} support points with positive mass (sum={support_weights.sum():.12f}, total_count={int(total_counts)})"
    )

    max_iter = args.max_iter if args.max_iter is not None else params.get("max_iter", 10000)
    tol = args.tol if args.tol is not None else params.get("tol", 1e-8)
    eps = args.eps if args.eps is not None else params.get("eps", 1e-12)
    conduct_mode = args.conduct_mode if args.conduct_mode is not None else params.get("conduct_mode", 1)
    if conduct_mode not in (1, 2):
        raise ValueError("conduct_mode must be 1 or 2 after removing behavioral elasticities.")

    mu_s_loc, sigma_s_loc = _compute_location_skill_moments(
        support_points, support_weights, params, eps=eps
    )
    if mu_s_loc is not None:
        print("[INFO] Using location-dependent skill moments.")

    skill_params = {
        "mu_x_skill": params.get("mu_x_skill"),
        "sigma_x_skill": params.get("sigma_x_skill"),
        "mu_a_skill": params.get("mu_a_skill"),
        "sigma_a_skill": params.get("sigma_a_skill"),
        "worker_mu_x": params.get("worker_mu_x"),
        "worker_mu_y": params.get("worker_mu_y"),
        "worker_sigma_x": params.get("worker_sigma_x"),
        "worker_sigma_y": params.get("worker_sigma_y"),
        "worker_rho": params.get("worker_rho"),
        "rho_x_skill_ell_x": float(params.get("rho_x_skill_ell_x", 0.0)),
        "rho_x_skill_ell_y": float(params.get("rho_x_skill_ell_y", 0.0)),
        "rho_x_skill_r": float(params.get("rho_x_skill_r", 0.0)),
        "worker_loc_mode": params.get("worker_loc_mode"),
        "worker_r_mu": params.get("worker_r_mu"),
        "worker_r_sigma": params.get("worker_r_sigma"),
        "mu_s_loc": mu_s_loc,
        "sigma_s_loc": sigma_s_loc,
    }

    data = EquilibriumData(
        A=A,
        xi=xi,
        loc_firms=loc_firms,
        support_points=support_points,
        support_weights=support_weights,
        mu_s=mu_s,
        sigma_s=sigma_s,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        N_workers=N_workers,
        **skill_params,
    )

    if args.noscreening:
        print(f"\nSolving no-screening equilibrium (mode={args.noscreening_mode})...")
        result = solve_equilibrium_noscreening_lm(
            data, max_iter=max_iter, tol=tol, eps=eps, mode=args.noscreening_mode
        )
        print(f"  Mode: {args.noscreening_mode}")

        print(f"\n=== EQUILIBRIUM RESULTS (no-screening) ===")
        print(f"Converged: {result['converged']}")
        print(f"Iterations: {result['iters']}")
        print(f"Final residual (max abs): {result['residual']:.2e}")

        L = result["L"]
        w = result["w"]
        Y = result["Y"]
        e = result["e"]
        L_share = result["L_share"]

        drop_threshold = args.drop_share_below
        drop_applied = False
        if drop_threshold is not None and drop_threshold > 0:
            shares = L / max(float(N_workers), 1e-12)
            mask = shares >= drop_threshold
            dropped = int(np.size(L) - np.count_nonzero(mask))
            if dropped >= np.size(L):
                print(
                    f"Warning: drop_share_below={drop_threshold:.2e} would remove all firms; skipping drop."
                )
            elif dropped > 0:
                print(f"Dropping {dropped} firms with worker share below {drop_threshold:.2e}.")
                A = A[mask]
                xi = xi[mask]
                loc_firms = loc_firms[mask]
                comp = comp[mask]
                firm_id_original = firm_id_original[mask]
                w = w[mask]
                L = L[mask]
                Y = Y[mask]
                e = e[mask]
                logw_masked = np.log(np.maximum(w, eps))
                data_masked = EquilibriumData(
                    A=A,
                    xi=xi,
                    loc_firms=loc_firms,
                    support_points=support_points,
                    support_weights=support_weights,
                    mu_s=mu_s,
                    sigma_s=sigma_s,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    N_workers=N_workers,
                    **skill_params,
                )
                diagnostics_refit = compute_equilibrium_objects_noscreening(
                    logw_masked, data_masked, eps, mode=args.noscreening_mode
                )
                w = diagnostics_refit["w"]
                L = diagnostics_refit["L"]
                Y = diagnostics_refit["Y"]
                e = diagnostics_refit["e"]
                firm_id = np.arange(1, L.size + 1, dtype=int)
                result["logw"] = logw_masked
                result["L"] = L
                result["w"] = w
                result["Y"] = Y
                result["e"] = e
                result["L_share"] = L / N_workers
                print(
                    f"  Remaining firms: {L.size}; min share={(L/N_workers).min():.3e}, max share={(L/N_workers).max():.3e}"
                )
                drop_applied = True
            else:
                result["L_share"] = L_share
        else:
            result["L_share"] = L_share
        L_share = result["L_share"]

        print("\nComputing numerical TFP elasticities for L_j and w_j (no-screening) ...")
        logw_equil = np.log(np.maximum(w, 1e-300))
        w_equil = np.exp(logw_equil)
        logA_equil = np.log(np.maximum(A, 1e-300))
        diff_steps_A = np.array([1e-8], dtype=float)
        diff_step_w = float(diff_steps_A[0])
        J_final = w.size
        L_tfp_elasticity_steps = np.full((diff_steps_A.size, J_final), np.nan, dtype=float)
        w_tfp_elasticity_steps = np.full((diff_steps_A.size, J_final), np.nan, dtype=float)
        L_clipped = np.zeros(J_final, dtype=bool)

        data_noscreen = EquilibriumData(
            A=A,
            xi=xi,
            loc_firms=loc_firms,
            support_points=support_points,
            support_weights=support_weights,
            mu_s=mu_s,
            sigma_s=sigma_s,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            N_workers=N_workers,
            **skill_params,
        )
        mu_s_arg, sigma_s_arg = _get_skill_moments(data_noscreen)

        def _aggregate_noscreening_for_w(w_candidate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if args.noscreening_mode == "heterogeneous_acceptall":
                c_very_low = np.full_like(w_candidate, -1e10)
                agg_local = aggregate_over_locations(
                    support_points,
                    support_weights,
                    w_candidate,
                    c_very_low,
                    xi,
                    loc_firms,
                    alpha,
                    gamma,
                    mu_s_arg,
                    sigma_s_arg,
                )
                L_share_local = np.maximum(agg_local["L_firms_nat"], eps)
                S_nat_local = np.maximum(agg_local["S_firms_nat"], eps)
            else:
                agg_local = aggregate_noscreening(
                    support_points,
                    support_weights,
                    w_candidate,
                    xi,
                    loc_firms,
                    alpha,
                    gamma,
                    eps=eps,
                )
                L_share_local = np.maximum(agg_local["L_share"], eps)
                S_nat_local = np.full_like(w_candidate, data_noscreen.mu_s, dtype=float)
            L_levels_local = N_workers * L_share_local
            return L_levels_local, S_nat_local

        e_fd = np.zeros(J_final, dtype=float)
        s_fd = np.zeros(J_final, dtype=float)
        for j in range(J_final):
            logw_plus = logw_equil.copy()
            logw_minus = logw_equil.copy()
            logw_plus[j] += diff_step_w
            logw_minus[j] -= diff_step_w
            w_plus = np.exp(logw_plus)
            w_minus = np.exp(logw_minus)
            L_plus, S_plus_all = _aggregate_noscreening_for_w(w_plus)
            L_minus, S_minus_all = _aggregate_noscreening_for_w(w_minus)
            L_plus_j = L_plus[j]
            L_minus_j = L_minus[j]
            S_plus_j = S_plus_all[j]
            S_minus_j = S_minus_all[j]
            e_fd[j] = (np.log(np.maximum(L_plus_j, eps)) - np.log(np.maximum(L_minus_j, eps))) / (
                2.0 * diff_step_w
            )
            s_fd[j] = (np.log(np.maximum(S_plus_j, eps)) - np.log(np.maximum(S_minus_j, eps))) / (
                2.0 * diff_step_w
            )

        def _profit_at_logw(
            logw_val: float, firm_idx: int, A_variant: np.ndarray
        ) -> float:
            w_candidate = w_equil.copy()
            w_candidate[firm_idx] = np.exp(logw_val)
            L_levels_local, S_nat_local = _aggregate_noscreening_for_w(w_candidate)
            LS_term_local = np.maximum(L_levels_local * S_nat_local, eps)
            Y_local = A_variant * (LS_term_local ** (1 - beta))
            return Y_local[firm_idx] - w_candidate[firm_idx] * L_levels_local[firm_idx]

        def optimize_w_best_response(A_variant: np.ndarray, firm_idx: int) -> float:
            x0 = np.array([logw_equil[firm_idx]], dtype=float)

            def objective(logw_vec: np.ndarray) -> float:
                return -_profit_at_logw(float(logw_vec[0]), firm_idx, A_variant)

            from scipy.optimize import minimize

            result_opt = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 1000},
            )
            if not result_opt.success:
                print(
                    f"Warning: profit optimizer for firm {firm_idx} returned {result_opt.message}; "
                    "using best available point."
                )
            return float(result_opt.x[0])

        for s_idx, step_A in enumerate(diff_steps_A):
            for j in range(J_final):
                A_plus = A.copy()
                A_plus[j] = np.exp(logA_equil[j] + step_A)
                logw_br_plus_j = optimize_w_best_response(A_plus, j)
                w_response_plus = w_equil.copy()
                w_response_plus[j] = np.exp(logw_br_plus_j)
                L_plus = _aggregate_noscreening_for_w(w_response_plus)[0][j]

                A_minus = A.copy()
                A_minus[j] = np.exp(logA_equil[j] - step_A)
                logw_br_minus_j = optimize_w_best_response(A_minus, j)
                w_response_minus = w_equil.copy()
                w_response_minus[j] = np.exp(logw_br_minus_j)
                L_minus = _aggregate_noscreening_for_w(w_response_minus)[0][j]

                w_tfp_elasticity_steps[s_idx, j] = (
                    logw_br_plus_j - logw_br_minus_j
                ) / (2.0 * step_A)
                L_tfp_elasticity_steps[s_idx, j] = (np.log(L_plus) - np.log(L_minus)) / (
                    2.0 * step_A
                )

                if L_plus <= 1.1e-12 and L_minus <= 1.1e-12:
                    L_clipped[j] = True

        w_tfp_elasticity = np.nanmedian(w_tfp_elasticity_steps, axis=0)
        L_tfp_elasticity = np.nanmedian(L_tfp_elasticity_steps, axis=0)
        tfp_labor_clipped = int(L_clipped.sum())
        tfp_w_zero = int(np.sum(np.abs(w_tfp_elasticity) <= 1e-10))

        result["L_tfp_elasticity"] = L_tfp_elasticity
        result["w_tfp_elasticity"] = w_tfp_elasticity
        result["L_wage_elasticity_fd"] = e_fd
        result["S_wage_elasticity_fd"] = s_fd
        print(f"  TFP step sizes used: {diff_steps_A}")
        print(
            f"  L elasticities w.r.t. A (min, max): ({L_tfp_elasticity.min():.4f}, {L_tfp_elasticity.max():.4f})"
        )
        print(
            f"  w elasticities w.r.t. A (min, max): ({w_tfp_elasticity.min():.4f}, {w_tfp_elasticity.max():.4f})"
        )
        print(
            f"  L elasticities wrt w (finite diff, step={diff_step_w:.1e}): ({e_fd.min():.4f}, {e_fd.max():.4f})"
        )
        print(
            f"  S elasticities wrt w (finite diff, step={diff_step_w:.1e}): ({s_fd.min():.4f}, {s_fd.max():.4f})"
        )
        if tfp_labor_clipped > 0:
            print(
                f"  Note: {tfp_labor_clipped} firms hit the 1e-12 employment floor at baseline or under the TFP bump; "
                "their labor elasticities are mechanically zero."
            )
        if tfp_w_zero > 0:
            print(
                f"  Note: {tfp_w_zero} firms show w elasticities ≈ 0 because their optimal wage does not move when "
                "TFP changes (typically due to negligible employment)."
            )

        print(f"\nSolution bounds (no-screening):")
        print(f"  L: [{np.min(L):.4f}, {np.max(L):.4f}]")
        print(f"  S: [{np.min(result['S']):.4f}, {np.max(result['S']):.4f}]")
        print(f"  w: [{np.min(w):.4f}, {np.max(w):.4f}]")
        print(f"  Y: [{np.min(Y):.4f}, {np.max(Y):.4f}]")
        print(f"  e (labor elasticities): [{np.min(e):.4f}, {np.max(e):.4f}]")
        if args.noscreening_mode == "heterogeneous_acceptall":
            print(f"  s (skill elasticities): [{np.min(result['s']):.4f}, {np.max(result['s']):.4f}]")

        print(f"\nWriting results...")
        out_dir.mkdir(parents=True, exist_ok=True)
        equilibrium_path = write_equilibrium_csv_noscreening(
            out_dir / "equilibrium_firms_noscreening.csv",
            firm_id,
            comp,
            A,
            xi,
            loc_firms,
            w,
            L,
            L_share,
            result["S"],
            Y,
            result["logw"],
            e,
            result["s"],
            L_tfp_elasticity=result.get("L_tfp_elasticity"),
            w_tfp_elasticity=result.get("w_tfp_elasticity"),
            L_wage_elasticity_fd=result.get("L_wage_elasticity_fd"),
            S_wage_elasticity_fd=result.get("S_wage_elasticity_fd"),
            firm_id_original=firm_id_original if drop_applied else None,
        )
        print(f"Equilibrium results written to: {equilibrium_path}")

        print(f"\n=== PACKAGE VERSIONS ===")
        print(f"numpy: {np.__version__}")
        print(f"pandas: {pd.__version__}")
        print(f"numba available: {NUMBA_AVAILABLE}")
        if NUMBA_AVAILABLE:
            print(f"numba: {numba.__version__}")
        return 0

    if args.use_lsq:
        print(f"\nSolving equilibrium (conduct_mode={conduct_mode}, method=trf)...")
        result = solve_equilibrium_lm(
            data, conduct_mode=conduct_mode, max_iter=max_iter, tol=tol, eps=eps
        )
    else:
        print(f"\nSolving equilibrium (conduct_mode={conduct_mode}, method=root-hybr)...")
        result = solve_equilibrium_root(
            data, conduct_mode=conduct_mode, max_iter=max_iter, tol=tol, eps=eps
        )

    print(f"\n=== EQUILIBRIUM RESULTS ===")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iters']}")
    print(f"Final residual (max abs): {result['residual']:.2e}")

    L = result["L"]
    S = result["S"]
    w = result["w"]
    c = result["c"]
    Y = result["Y"]
    rank = result["rank"]

    drop_threshold = args.drop_share_below
    drop_applied = False
    if drop_threshold is not None and drop_threshold > 0:
        shares = L / max(float(N_workers), 1e-12)
        mask = shares >= drop_threshold
        dropped = int(np.size(L) - np.count_nonzero(mask))
        if dropped >= np.size(L):
            print(
                f"Warning: drop_share_below={drop_threshold:.2e} would remove all firms; skipping drop."
            )
        elif dropped > 0:
            print(f"Dropping {dropped} firms with worker share below {drop_threshold:.2e}.")
            A = A[mask]
            xi = xi[mask]
            loc_firms = loc_firms[mask]
            comp = comp[mask]
            firm_id_original = firm_id_original[mask]

            logw_masked = np.log(np.maximum(w[mask], eps))
            logc_masked = np.log(np.maximum(c[mask], eps))

            data = EquilibriumData(
                A=A,
                xi=xi,
                loc_firms=loc_firms,
                support_points=support_points,
                support_weights=support_weights,
                mu_s=mu_s,
                sigma_s=sigma_s,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                N_workers=N_workers,
                **skill_params,
            )
            diagnostics_refit = compute_equilibrium_objects(
                logw_masked, logc_masked, data, conduct_mode, eps
            )

            w = np.exp(logw_masked)
            c = np.exp(logc_masked)
            L = diagnostics_refit["L_levels_nat"]
            S = diagnostics_refit["S_nat"]
            Y = diagnostics_refit["Y_nat"]
            rank = diagnostics_refit["rank"]

            firm_id = np.arange(1, L.size + 1, dtype=int)
            result["diagnostics"] = diagnostics_refit
            result["logw"] = logw_masked
            result["logc"] = logc_masked
            result["rank"] = rank
            result["L"] = L
            result["S"] = S
            result["w"] = w
            result["c"] = c
            result["Y"] = Y

            shares = L / N_workers
            print(
                f"  Remaining firms: {L.size}; min share={shares.min():.3e}, max share={shares.max():.3e}"
            )
            drop_applied = True

    data_current = EquilibriumData(
        A=A,
        xi=xi,
        loc_firms=loc_firms,
        support_points=support_points,
        support_weights=support_weights,
        mu_s=mu_s,
        sigma_s=sigma_s,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        N_workers=N_workers,
        **skill_params,
    )

    print("\nComputing wage elasticities for L_j and S_j via autodiff ...")
    logw_equil = np.log(np.maximum(w, 1e-300))
    logc_equil = np.log(np.maximum(c, 1e-300))
    w_equil = np.exp(logw_equil)
    c_equil = np.exp(logc_equil)
    L_wage_elasticity, S_wage_elasticity = compute_wage_elasticities(
        logw_equil, logc_equil, data_current, eps
    )
    L_cutoff_elasticity, S_cutoff_elasticity = compute_cutoff_elasticities(
        logw_equil, logc_equil, data_current, eps
    )
    J_final = w.size

    result["L_wage_elasticity"] = L_wage_elasticity
    result["S_wage_elasticity"] = S_wage_elasticity
    result["L_cutoff_elasticity"] = L_cutoff_elasticity
    result["S_cutoff_elasticity"] = S_cutoff_elasticity

    print(
        f"  L elasticities (min, max): ({L_wage_elasticity.min():.4f}, {L_wage_elasticity.max():.4f})"
    )
    print(
        f"  S elasticities (min, max): ({S_wage_elasticity.min():.4f}, {S_wage_elasticity.max():.4f})"
    )
    print(
        f"  L wrt cutoff (min, max): ({L_cutoff_elasticity.min():.4f}, {L_cutoff_elasticity.max():.4f})"
    )
    print(
        f"  S wrt cutoff (min, max): ({S_cutoff_elasticity.min():.4f}, {S_cutoff_elasticity.max():.4f})"
    )

    print("\nComputing numerical TFP elasticities for L_j, w_j, and c_j ...")
    diff_step = 1e-6
    diff_step_A = 1e-6
    logA_equil = np.log(np.maximum(A, 1e-300))
    L_tfp_elasticity = np.zeros(J_final, dtype=float)
    w_tfp_elasticity = np.zeros(J_final, dtype=float)
    c_tfp_elasticity = np.zeros(J_final, dtype=float)
    tfp_labor_clipped = 0
    tfp_w_zero = 0
    w_equil = np.exp(logw_equil)
    c_equil = np.exp(logc_equil)
    mu_s_arg, sigma_s_arg = _get_skill_moments(data_current)
    # Baseline profit-maximizing choices at current A (diagnostic)
    br_logw_base = np.zeros(J_final, dtype=float)
    br_logc_base = np.zeros(J_final, dtype=float)

    def _profit_at_logs(log_vec: np.ndarray, firm_idx: int, A_variant: np.ndarray) -> float:
        w_candidate = w_equil.copy()
        c_candidate = c_equil.copy()
        w_candidate[firm_idx] = np.exp(log_vec[0])
        c_candidate[firm_idx] = np.exp(log_vec[1])

        agg = aggregate_over_locations(
            support_points,
            support_weights,
            w_candidate,
            c_candidate,
            xi,
            loc_firms,
            alpha,
            gamma,
            mu_s_arg,
            sigma_s_arg,
        )

        order_idx_local = agg["order_idx"]
        inv_order_local = agg["inv_order"]
        L_byc_local = np.maximum(agg["L_byc"][1:], 0.0)
        S_byc_local = np.maximum(agg["S_byc"][1:], 1e-12)
        L_levels_byc = N_workers * L_byc_local
        A_byc_local = A_variant[order_idx_local]
        LS_term_local = np.maximum(L_levels_byc * S_byc_local, 1e-300)
        Y_byc_local = A_byc_local * (LS_term_local ** (1 - beta))
        Y_nat = Y_byc_local[inv_order_local]

        L_share_nat = np.maximum(agg["L_firms_nat"], 1e-12)
        L_levels_nat = N_workers * L_share_nat

        profit = Y_nat[firm_idx] - w_candidate[firm_idx] * L_levels_nat[firm_idx]
        return profit

    def optimize_profit_best_response(
        A_variant: np.ndarray, firm_idx: int
    ) -> Tuple[float, float]:
        x0 = np.array([logw_equil[firm_idx], logc_equil[firm_idx]], dtype=float)

        def objective(log_vec: np.ndarray) -> float:
            return -_profit_at_logs(log_vec, firm_idx, A_variant)

        from scipy.optimize import minimize

        result_opt = minimize(objective, x0, method="L-BFGS-B")
        if not result_opt.success:
            print(
                f"Warning: profit optimizer for firm {firm_idx} returned {result_opt.message}; "
                "using best available point."
            )
        return float(result_opt.x[0]), float(result_opt.x[1])

    for j in range(J_final):
        br_logw_base[j], br_logc_base[j] = optimize_profit_best_response(A, j)
    for j in range(J_final):
        A_plus = A.copy()
        A_plus[j] = np.exp(logA_equil[j] + diff_step_A)

        logw_br_plus_j, logc_br_plus_j = optimize_profit_best_response(A_plus, j)

        logw_response_plus = logw_equil.copy()
        logc_response_plus = logc_equil.copy()
        logw_response_plus[j] = logw_br_plus_j
        logc_response_plus[j] = logc_br_plus_j

        agg_plus = aggregate_over_locations(
            support_points,
            support_weights,
            np.exp(logw_response_plus),
            np.exp(logc_response_plus),
            xi,
            loc_firms,
            alpha,
            gamma,
            mu_s_arg,
            sigma_s_arg,
        )
        L_plus = np.maximum(
            np.asarray(agg_plus["L_firms_nat"], dtype=float)[j],
            1e-12,
        )

        A_minus = A.copy()
        A_minus[j] = np.exp(logA_equil[j] - diff_step_A)

        logw_br_minus_j, logc_br_minus_j = optimize_profit_best_response(A_minus, j)

        logw_response_minus = logw_equil.copy()
        logc_response_minus = logc_equil.copy()
        logw_response_minus[j] = logw_br_minus_j
        logc_response_minus[j] = logc_br_minus_j

        agg_minus = aggregate_over_locations(
            support_points,
            support_weights,
            np.exp(logw_response_minus),
            np.exp(logc_response_minus),
            xi,
            loc_firms,
            alpha,
            gamma,
            mu_s_arg,
            sigma_s_arg,
        )
        L_minus = np.maximum(
            np.asarray(agg_minus["L_firms_nat"], dtype=float)[j],
            1e-12,
        )

        w_tfp_elasticity[j] = (logw_br_plus_j - logw_br_minus_j) / (2.0 * diff_step_A)
        c_tfp_elasticity[j] = (logc_br_plus_j - logc_br_minus_j) / (2.0 * diff_step_A)
        L_tfp_elasticity[j] = (np.log(L_plus) - np.log(L_minus)) / (2.0 * diff_step_A)

        if L_plus <= 1.1e-12 and L_minus <= 1.1e-12:
            tfp_labor_clipped += 1
        if abs(w_tfp_elasticity[j]) <= 1e-10:
            tfp_w_zero += 1

    result["L_tfp_elasticity"] = L_tfp_elasticity
    result["w_tfp_elasticity"] = w_tfp_elasticity
    result["c_tfp_elasticity"] = c_tfp_elasticity
    print(
        f"  L elasticities w.r.t. A (min, max): ({L_tfp_elasticity.min():.4f}, {L_tfp_elasticity.max():.4f})"
    )
    print(
        f"  w elasticities w.r.t. A (min, max): ({w_tfp_elasticity.min():.4f}, {w_tfp_elasticity.max():.4f})"
    )
    print(
        f"  c elasticities w.r.t. A (min, max): ({c_tfp_elasticity.min():.4f}, {c_tfp_elasticity.max():.4f})"
    )
    if tfp_labor_clipped > 0:
        print(
            f"  Note: {tfp_labor_clipped} firms hit the 1e-12 employment floor at baseline or under the TFP bump; "
            "their labor elasticities are mechanically zero."
        )
    if tfp_w_zero > 0:
        print(
            f"  Note: {tfp_w_zero} firms show w elasticities ≈ 0 because their optimal wage does not move when "
            "TFP changes (typically due to negligible employment)."
        )

    print("\nSolution bounds:")
    print(f"  L: [{np.min(L):.4f}, {np.max(L):.4f}]")
    print(f"  S: [{np.min(S):.4f}, {np.max(S):.4f}]")
    print(f"  w: [{np.min(w):.4f}, {np.max(w):.4f}]")
    print(f"  c: [{np.min(c):.4f}, {np.max(c):.4f}]")
    print(f"  Y: [{np.min(Y):.4f}, {np.max(Y):.4f}]")

    c_sorted = np.sort(c)
    from scipy.stats import norm

    mass_above_min = 1.0 - norm.cdf((c_sorted[0] - mu_s) / sigma_s)
    weight_sum = float(support_weights.sum())
    print(f"\n[CHECK] Σ L (counts) = {L.sum():.6f}  vs expected ≈ {N_workers * mass_above_min * weight_sum:.6f}")
    print(f"[INFO] N_workers = {N_workers:.0f}, employment rate = {L.sum()/N_workers:.6%}")

    print(f"\nWriting results...")
    out_dir.mkdir(parents=True, exist_ok=True)

    equilibrium_path = write_equilibrium_csv(
        out_dir / "equilibrium_firms.csv",
        firm_id,
        comp,
        A,
        xi,
        loc_firms,
        w,
        c,
        L,
        S,
        Y,
        result["logw"],
        result["logc"],
        result["rank"],
        firm_id_original=firm_id_original if drop_applied else None,
        L_wage_elasticity=L_wage_elasticity,
        S_wage_elasticity=S_wage_elasticity,
        L_cutoff_elasticity=L_cutoff_elasticity,
        S_cutoff_elasticity=S_cutoff_elasticity,
        L_tfp_elasticity=L_tfp_elasticity,
        w_tfp_elasticity=w_tfp_elasticity,
        c_tfp_elasticity=c_tfp_elasticity,
    )
    print(f"Equilibrium results written to: {equilibrium_path}")

    if args.plot_profits_fixed:
        grid_n = args.profit_grid_n if args.profit_grid_n is not None else 40
        grid_span = args.profit_grid_log_span if args.profit_grid_log_span is not None else 0.5
        profit_dir = out_dir / "profit_surfaces_fixed"
        data_plot = EquilibriumData(
            A=A,
            xi=xi,
            loc_firms=loc_firms,
            support_points=support_points,
            support_weights=support_weights,
            mu_s=mu_s,
            sigma_s=sigma_s,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            N_workers=N_workers,
            **skill_params,
        )
        for j_idx, fid in enumerate(firm_id):
            out_path = profit_dir / f"firm_{fid}_profit_surface.png"
            plot_profit_surface_fixed_others(
                result["logw"],
                result["logc"],
                data_plot,
                j_idx,
                out_path,
                grid_n=grid_n,
                grid_log_span=grid_span,
                eps=eps,
                br_logw=br_logw_base,
                br_logc=br_logc_base,
            )
        print(f"Profit surface plots written to: {profit_dir}")

    print(f"\n=== PACKAGE VERSIONS ===")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"numba available: {NUMBA_AVAILABLE}")
    if NUMBA_AVAILABLE:
        print(f"numba: {numba.__version__}")

    return 0


if __name__ == "__main__":
    exit(main())
