#!/usr/bin/env python3
"""
JAX implementation of discrete-choice probabilities used in the τ,δ,q̄ model.

- Pure JAX (no NumPy). Vectorized across workers (N) and firms (J).
- Outside option is column 0 in the returned matrix P (N, J+1).
- Numerical stability: clip probabilities only at the end, then renormalize rows.

Log-additive skill specification:
    ln q_i = gamma0 + gamma1 * v_i + e_i
    v_i ~ N(mu_v, sigma_v^2)     observable (stored as x_skill in CSV)
    e_i ~ N(0, sigma_e^2)        unobservable

Screening: worker i eligible for firm j iff ln q_i >= ln_qbar_j.
Cutoffs ln_qbar are stored/optimised in log space (unconstrained).

Inputs
------
- theta: 1D array with length 1 + 2J in natural firm order
         theta = [tau, delta_nat(1..J), ln_qbar_nat(1..J)]
- X:     Worker covariates; here X is the observable skill v with shape (N,)
- aux:   Dict pytree with required precomputed arrays/scalars:
         - 'D_nat': (N,J) worker→firm distances in natural order
         - 'gamma0': scalar intercept in ln q = gamma0 + gamma1*v + e
         - 'gamma1': scalar loading on observable skill v (replaces 'gamma')
         - 'sigma_e': scalar σ_e (>0)
         - 'firm_ids': (J,) natural firm ids (1..J). Only used for documentation/matching.

Semantics follow helpers.compute_choice_probabilities.
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr  # Normal CDF


def enable_x64() -> None:
    """Enable float64 precision for JAX computations."""
    jax.config.update("jax_enable_x64", True)


def _order_maps_jax(qbar_nat: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Return order_idx (natural→by-qbar), inv_order (by-qbar→natural), qbar_sorted."""
    order_idx = jnp.argsort(qbar_nat)  # (J,)
    # inv_order[pos] = natural index for by-qbar position 'pos'
    inv_order = jnp.argsort(order_idx)
    qbar_sorted = qbar_nat[order_idx]
    return {"order_idx": order_idx, "inv_order": inv_order, "qbar_sorted": qbar_sorted}


def _choice_probabilities_from_cutoffs(
    tau: jnp.ndarray,
    delta_nat: jnp.ndarray,
    qbar_nat: jnp.ndarray,
    X: jnp.ndarray,
    aux: Dict,
) -> jnp.ndarray:
    """Shared core that evaluates probabilities given (tau, delta, qbar)."""

    D_nat: jnp.ndarray = aux["D_nat"]  # (N, J)
    gamma0 = jnp.asarray(aux.get("gamma0", aux.get("mu_e", 0.0)), dtype=jnp.float64)
    gamma1 = jnp.asarray(aux.get("gamma1", aux.get("gamma", 1.0)), dtype=jnp.float64)
    sigma_e = jnp.asarray(aux.get("sigma_e", 1.0), dtype=jnp.float64)

    N = D_nat.shape[0]
    J = D_nat.shape[1]

    tau = jnp.asarray(tau)
    delta_nat = jnp.asarray(delta_nat).reshape((J,))
    ln_qbar_nat = jnp.asarray(qbar_nat).reshape((J,))

    maps = _order_maps_jax(ln_qbar_nat)
    order_idx = maps["order_idx"]
    inv_order = maps["inv_order"]
    ln_qbar_sorted = maps["qbar_sorted"]

    delta_sorted = delta_nat[order_idx]
    D_sorted = D_nat[:, order_idx]

    v_out = jnp.ones((N, 1), dtype=tau.dtype)
    v_in = jnp.exp(-tau * D_sorted + delta_sorted[None, :])
    v_all = jnp.concatenate((v_out, v_in), axis=1)

    denom = jnp.cumsum(v_all, axis=1)

    big = jnp.array(1e6, dtype=ln_qbar_sorted.dtype)
    ln_qbar_pad = jnp.concatenate(
        (jnp.array([-big], dtype=ln_qbar_sorted.dtype), ln_qbar_sorted, jnp.array([big], dtype=ln_qbar_sorted.dtype))
    )

    x_skill = jnp.asarray(X).reshape((N,))
    mu_lnq_i = gamma0 + gamma1 * x_skill  # per-worker mean of ln q

    z_hi = (ln_qbar_pad[1:][None, :] - mu_lnq_i[:, None]) / sigma_e
    z_lo = (ln_qbar_pad[:-1][None, :] - mu_lnq_i[:, None]) / sigma_e
    p_x = ndtr(z_hi) - ndtr(z_lo)

    p_x = jnp.clip(p_x, 0.0, 1.0)
    p_row_sum = jnp.sum(p_x, axis=1, keepdims=True)
    p_x = p_x / jnp.maximum(p_row_sum, 1e-300)

    q = p_x / jnp.maximum(denom, 1e-300)
    Q = jnp.flip(jnp.cumsum(jnp.flip(q, axis=1), axis=1), axis=1)

    P_byc = v_all * Q

    idx_nat = jnp.concatenate((jnp.array([0], dtype=jnp.int32), inv_order.astype(jnp.int32) + 1))
    P_nat = P_byc[:, idx_nat]

    P_nat = jnp.clip(P_nat, 1e-300, 1.0)
    row_sum = jnp.sum(P_nat, axis=1, keepdims=True)
    P_nat = P_nat / jnp.maximum(row_sum, 1e-300)

    return P_nat


@jax.jit
def compute_choice_probabilities_jax(theta: jnp.ndarray, X: jnp.ndarray, aux: Dict) -> jnp.ndarray:
    """
    Compute choice probabilities when theta packs firm cutoffs qbar in natural order.

    Args:
      theta: (1+2J,) with [tau, delta_nat(1..J), qbar_nat(1..J)]
      X:     (N,) worker skills vector
      aux:   dict with at least {D_nat, gamma0, gamma1, sigma_e}

    Returns:
      P_nat: (N, J+1) probabilities in natural firm order; column 0 = outside.
    """

    D_nat: jnp.ndarray = aux["D_nat"]
    J = D_nat.shape[1]

    theta = jnp.asarray(theta).reshape(-1)
    tau = theta[0]
    delta_nat = theta[1 : 1 + J]
    qbar_nat = theta[1 + J : 1 + 2 * J]

    return _choice_probabilities_from_cutoffs(tau, delta_nat, qbar_nat, X, aux)


@jax.jit
def compute_choice_probabilities_tau_qbar_delta_jax(theta: jnp.ndarray, X: jnp.ndarray, aux: Dict) -> jnp.ndarray:
    """Compute probabilities when the parameter vector packs (tau, delta, qbar).

    Args:
      theta: (1+2J,) with [tau, delta_nat(1..J), qbar_nat(1..J)]
      X:     (N,) worker skill draws
      aux:   dict with at least {D_nat, gamma0, gamma1, sigma_e}; other keys ignored

    Returns:
      P_nat: (N, J+1) probabilities in natural firm order; column 0 = outside.
    """

    return compute_choice_probabilities_jax(theta, X, aux)


def compute_penalty_components_jax(
    theta: jnp.ndarray,
    X: jnp.ndarray,
    choice_idx: jnp.ndarray,
    aux: Dict,
    w_nat: jnp.ndarray,
    Y_nat: jnp.ndarray,
    L_data: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (P_nat, per_obs_nll, m_vec, L_data, Q_nat) for penalized objectives."""

    D_nat: jnp.ndarray = aux["D_nat"]
    gamma0 = jnp.asarray(aux.get("gamma0", aux.get("mu_e", 0.0)), dtype=jnp.float64)
    gamma1 = jnp.asarray(aux.get("gamma1", aux.get("gamma", 1.0)), dtype=jnp.float64)
    sigma_e = jnp.asarray(aux.get("sigma_e", 1.0), dtype=jnp.float64)

    theta = jnp.asarray(theta).reshape(-1)
    X = jnp.asarray(X).reshape(-1)
    choice_idx = jnp.asarray(choice_idx, dtype=jnp.int32).reshape(-1)
    w_nat = jnp.asarray(w_nat).reshape(-1)
    Y_nat = jnp.asarray(Y_nat).reshape(-1)
    L_data = jnp.asarray(L_data).reshape(-1)

    N = X.shape[0]
    J = w_nat.shape[0]

    tau = theta[0]
    alpha = theta[1]
    delta_nat = theta[2 : 2 + J]
    ln_qbar_nat = theta[2 + J : 2 + 2 * J]

    maps = _order_maps_jax(ln_qbar_nat)
    order_idx = maps["order_idx"]
    inv_order = maps["inv_order"]
    ln_qbar_sorted = maps["qbar_sorted"]

    delta_sorted = delta_nat[order_idx]
    D_sorted = D_nat[:, order_idx]

    v_out = jnp.ones((N, 1), dtype=theta.dtype)
    v_in_sorted = jnp.exp(-tau * D_sorted + delta_sorted[None, :])
    v_all = jnp.concatenate((v_out, v_in_sorted), axis=1)

    denom = jnp.cumsum(v_all, axis=1)
    denom = jnp.maximum(denom, 1e-300)

    big = jnp.array(1e6, dtype=ln_qbar_sorted.dtype)
    ln_qbar_pad = jnp.concatenate(((-big)[None], ln_qbar_sorted, big[None]))

    mu_lnq_i = gamma0 + gamma1 * X  # (N,) per-worker mean of ln q
    z_hi = (ln_qbar_pad[1:] - mu_lnq_i[:, None]) / sigma_e
    z_lo = (ln_qbar_pad[:-1] - mu_lnq_i[:, None]) / sigma_e

    p_x = ndtr(z_hi) - ndtr(z_lo)
    p_x = jnp.clip(p_x, 0.0, 1.0)
    p_row_sum = jnp.sum(p_x, axis=1, keepdims=True)
    p_x = p_x / jnp.maximum(p_row_sum, 1e-12)

    q = p_x / denom
    suffix_sum = jnp.flip(jnp.cumsum(jnp.flip(q, axis=1), axis=1), axis=1)

    P_byc = v_all * suffix_sum
    idx_nat = jnp.concatenate((jnp.array([0], dtype=jnp.int32), inv_order.astype(jnp.int32) + 1))
    P_nat = P_byc[:, idx_nat]
    P_nat = jnp.clip(P_nat, 1e-300, 1.0)
    P_nat = P_nat / jnp.maximum(jnp.sum(P_nat, axis=1, keepdims=True), 1e-300)

    probs_chosen = jnp.take_along_axis(P_nat, choice_idx[:, None], axis=1).squeeze(axis=1)
    per_obs_nll = -jnp.log(jnp.clip(probs_chosen, 1e-300, 1.0))

    # --- E[q | matched to j] via truncated lognormal formula ---
    # E[exp(X) | a < X < b] where X ~ N(mu, sigma^2)
    #   = exp(mu + sigma^2/2) * [Phi(z_hi - sigma) - Phi(z_lo - sigma)] / [Phi(z_hi) - Phi(z_lo)]
    z_hi_shifted = z_hi - sigma_e
    z_lo_shifted = z_lo - sigma_e
    exp_factor = jnp.exp(mu_lnq_i[:, None] + 0.5 * sigma_e**2)  # (N, J+1)
    p_x_shifted = ndtr(z_hi_shifted) - ndtr(z_lo_shifted)  # (N, J+1)
    p_x_shifted = jnp.maximum(p_x_shifted, 0.0)
    # E_q_in_interval[i, k] = E[q_i | ln q_i in interval_k] * P(ln q_i in interval_k)
    E_q_in_interval = exp_factor * p_x_shifted  # (N, J+1) — unnormalized

    # Compute joint probability and skill contribution per firm (sorted order)
    joint_prob = v_in_sorted[:, :, None] * p_x[:, None, :] / denom[:, None, :]
    interval_idx = jnp.arange(J + 1)
    firm_thresholds = (jnp.arange(J) + 1)[:, None]
    eligibility_mask = (interval_idx[None, :] >= firm_thresholds).astype(theta.dtype)
    joint_prob = joint_prob * eligibility_mask[None, :, :]

    # E[q | matched to j] contribution: sum over eligible intervals
    weighted_q_mass = (
        v_in_sorted[:, :, None]
        * E_q_in_interval[:, None, :]
        / denom[:, None, :]
        * eligibility_mask[None, :, :]
    )
    skill_contrib_sorted = jnp.sum(weighted_q_mass, axis=2)
    joint_prob_sum = jnp.sum(joint_prob, axis=2)

    skill_sum_sorted = jnp.sum(skill_contrib_sorted, axis=0)
    L_model_sorted = jnp.sum(joint_prob_sum, axis=0)

    skill_sum_nat = jnp.zeros(J, dtype=theta.dtype).at[order_idx].set(skill_sum_sorted)
    L_model_nat = jnp.zeros(J, dtype=theta.dtype).at[order_idx].set(L_model_sorted)
    Q_nat = skill_sum_nat / jnp.maximum(L_model_nat, 1e-300)

    def safe_vals(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(arr, 1e-300)

    safe_alpha = jnp.clip(alpha, 1e-6, 1.0 - 1e-6)
    # ln_qbar is already in log space — no need for jnp.log(qbar)
    m_vec = (
        ln_qbar_nat
        + jnp.log1p(-safe_alpha)
        - jnp.log(safe_vals(w_nat))
        + jnp.log(safe_vals(Y_nat))
        - jnp.log(safe_vals(L_data))
        - jnp.log(safe_vals(Q_nat))
    )

    return P_nat, per_obs_nll, m_vec, L_data, Q_nat


def compute_penalty_components_with_params_jax(
    theta: jnp.ndarray,
    X: jnp.ndarray,
    choice_idx: jnp.ndarray,
    aux: Dict,
    w_nat: jnp.ndarray,
    Y_nat: jnp.ndarray,
    L_data: jnp.ndarray,
    gamma1: jnp.ndarray,
    sigma_e: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    aux_dynamic = dict(aux)
    aux_dynamic["gamma1"] = jnp.asarray(gamma1, dtype=jnp.float64)
    aux_dynamic["sigma_e"] = jnp.asarray(sigma_e, dtype=jnp.float64)
    return compute_penalty_components_jax(theta, X, choice_idx, aux_dynamic, w_nat, Y_nat, L_data)


__all__ = [
    "enable_x64",
    "choice_probabilities_from_cutoffs",
    "compute_choice_probabilities_jax",
    "compute_choice_probabilities_tau_qbar_delta_jax",
    "compute_penalty_components_jax",
    "compute_penalty_components_with_params_jax",
]

# Public alias for the core probability function (used by blp_contraction).
choice_probabilities_from_cutoffs = _choice_probabilities_from_cutoffs
