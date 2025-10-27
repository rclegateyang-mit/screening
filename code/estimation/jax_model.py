#!/usr/bin/env python3
"""
JAX implementation of discrete-choice probabilities used in the γ,V,c model.

- Pure JAX (no NumPy). Vectorized across workers (N) and firms (J).
- Outside option is column 0 in the returned matrix P (N, J+1).
- Numerical stability: clip probabilities only at the end, then renormalize rows.

Inputs
------
- theta: 1D array with length 1 + 2J in natural firm order
         theta = [gamma, V_nat(1..J), c_nat(1..J)]
- X:     Worker covariates; here we expect X to be the skill vector with shape (N,)
         (If you have richer features, pass the skill as X[:, 0] or adapt as needed.)
- aux:   Dict pytree with required precomputed arrays/scalars:
         - 'D_nat': (N,J) worker→firm distances in natural order
         - 'phi': scalar φ
         - 'mu_a': scalar μ_a
         - 'sigma_a': scalar σ_a (>0)
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


def _order_maps_jax(c_nat: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Return order_idx (natural→by-c), inv_order (by-c→natural), c_sorted."""
    order_idx = jnp.argsort(c_nat)  # (J,)
    # inv_order[pos] = natural index for by-c position 'pos'
    inv_order = jnp.argsort(order_idx)
    c_sorted = c_nat[order_idx]
    return {"order_idx": order_idx, "inv_order": inv_order, "c_sorted": c_sorted}


def _choice_probabilities_from_cutoffs(
    gamma: jnp.ndarray,
    V_nat: jnp.ndarray,
    c_nat: jnp.ndarray,
    X: jnp.ndarray,
    aux: Dict,
) -> jnp.ndarray:
    """Shared core that evaluates probabilities given (gamma, V, c)."""

    D_nat: jnp.ndarray = aux["D_nat"]  # (N, J)
    phi = jnp.asarray(aux.get("phi", 1.0), dtype=jnp.float64)
    mu_a = jnp.asarray(aux.get("mu_a", 0.0), dtype=jnp.float64)
    sigma_a = jnp.asarray(aux.get("sigma_a", 1.0), dtype=jnp.float64)

    N = D_nat.shape[0]
    J = D_nat.shape[1]

    gamma = jnp.asarray(gamma)
    V_nat = jnp.asarray(V_nat).reshape((J,))
    c_nat = jnp.asarray(c_nat).reshape((J,))

    maps = _order_maps_jax(c_nat)
    order_idx = maps["order_idx"]
    inv_order = maps["inv_order"]
    c_sorted = maps["c_sorted"]

    V_sorted = V_nat[order_idx]
    D_sorted = D_nat[:, order_idx]

    v_out = jnp.ones((N, 1), dtype=gamma.dtype)
    v_in = jnp.exp(-gamma * D_sorted + V_sorted[None, :])
    v_all = jnp.concatenate((v_out, v_in), axis=1)

    denom = jnp.cumsum(v_all, axis=1)

    big = jnp.array(1e6, dtype=c_sorted.dtype)
    c_pad = jnp.concatenate(
        (jnp.array([-big], dtype=c_sorted.dtype), c_sorted, jnp.array([big], dtype=c_sorted.dtype))
    )

    x_skill = jnp.asarray(X).reshape((N,))
    s = phi * x_skill + mu_a

    z_hi = (c_pad[1:][None, :] - s[:, None]) / sigma_a
    z_lo = (c_pad[:-1][None, :] - s[:, None]) / sigma_a
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
    Compute choice probabilities when theta packs firm cutoffs c in natural order.

    Args:
      theta: (1+2J,) with [gamma, V_nat(1..J), c_nat(1..J)]
      X:     (N,) worker skills vector
      aux:   dict with at least {D_nat, phi, mu_a, sigma_a}

    Returns:
      P_nat: (N, J+1) probabilities in natural firm order; column 0 = outside.
    """

    D_nat: jnp.ndarray = aux["D_nat"]
    J = D_nat.shape[1]

    theta = jnp.asarray(theta).reshape(-1)
    gamma = theta[0]
    V_nat = theta[1 : 1 + J]
    c_nat = theta[1 + J : 1 + 2 * J]

    return _choice_probabilities_from_cutoffs(gamma, V_nat, c_nat, X, aux)


@jax.jit
def compute_choice_probabilities_gamma_c_V_jax(theta: jnp.ndarray, X: jnp.ndarray, aux: Dict) -> jnp.ndarray:
    """Compute probabilities when the parameter vector packs (gamma, V, c).

    Args:
      theta: (1+2J,) with [gamma, V_nat(1..J), c_nat(1..J)]
      X:     (N,) worker skill draws
      aux:   dict with at least {D_nat, phi, mu_a, sigma_a}; other keys ignored

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
    """Return (P_nat, per_obs_nll, m_vec, L_data, S_nat) for penalized objectives."""

    D_nat: jnp.ndarray = aux["D_nat"]
    phi = jnp.asarray(aux.get("phi", 1.0), dtype=jnp.float64)
    mu_a = jnp.asarray(aux.get("mu_a", 0.0), dtype=jnp.float64)
    sigma_a = jnp.asarray(aux.get("sigma_a", 1.0), dtype=jnp.float64)

    theta = jnp.asarray(theta).reshape(-1)
    X = jnp.asarray(X).reshape(-1)
    choice_idx = jnp.asarray(choice_idx, dtype=jnp.int32).reshape(-1)
    w_nat = jnp.asarray(w_nat).reshape(-1)
    Y_nat = jnp.asarray(Y_nat).reshape(-1)
    L_data = jnp.asarray(L_data).reshape(-1)

    N = X.shape[0]
    J = w_nat.shape[0]

    gamma = theta[0]
    beta = theta[1]
    V_nat = theta[2 : 2 + J]
    c_nat = theta[2 + J : 2 + 2 * J]

    maps = _order_maps_jax(c_nat)
    order_idx = maps["order_idx"]
    inv_order = maps["inv_order"]
    c_sorted = maps["c_sorted"]

    V_sorted = V_nat[order_idx]
    D_sorted = D_nat[:, order_idx]

    v_out = jnp.ones((N, 1), dtype=theta.dtype)
    v_in_sorted = jnp.exp(-gamma * D_sorted + V_sorted[None, :])
    v_all = jnp.concatenate((v_out, v_in_sorted), axis=1)

    denom = jnp.cumsum(v_all, axis=1)
    denom = jnp.maximum(denom, 1e-300)

    big = jnp.array(1e6, dtype=c_sorted.dtype)
    c_pad = jnp.concatenate(((-big)[None], c_sorted, big[None]))

    phi_x = phi * X
    z_hi = (c_pad[1:] - phi_x[:, None] - mu_a) / sigma_a
    z_lo = (c_pad[:-1] - phi_x[:, None] - mu_a) / sigma_a

    p_x = ndtr(z_hi) - ndtr(z_lo)
    p_x = jnp.clip(p_x, 0.0, 1.0)
    p_row_sum = jnp.sum(p_x, axis=1, keepdims=True)
    p_x = p_x / jnp.maximum(p_row_sum, 1e-12)

    q = p_x / denom
    suffix_sum = jnp.flip(jnp.cumsum(jnp.flip(q, axis=1), axis=1), axis=1)

    P_byc = v_all * suffix_sum
    P_sorted = P_byc[:, 1:]
    idx_nat = jnp.concatenate((jnp.array([0], dtype=jnp.int32), inv_order.astype(jnp.int32) + 1))
    P_nat = P_byc[:, idx_nat]
    P_nat = jnp.clip(P_nat, 1e-300, 1.0)
    P_nat = P_nat / jnp.maximum(jnp.sum(P_nat, axis=1, keepdims=True), 1e-300)

    probs_chosen = jnp.take_along_axis(P_nat, choice_idx[:, None], axis=1).squeeze(axis=1)
    per_obs_nll = -jnp.log(jnp.clip(probs_chosen, 1e-300, 1.0))

    def std_normal_pdf(z: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-0.5 * jnp.square(z)) / jnp.sqrt(2.0 * jnp.pi)

    pdf_lo = std_normal_pdf(z_lo)
    pdf_hi = std_normal_pdf(z_hi)

    ability_weight_term = mu_a * p_x + sigma_a * (pdf_lo - pdf_hi)

    joint_prob = v_in_sorted[:, :, None] * p_x[:, None, :] / denom[:, None, :]
    interval_idx = jnp.arange(J + 1)
    firm_thresholds = (jnp.arange(J) + 1)[:, None]
    eligibility_mask = (interval_idx[None, :] >= firm_thresholds).astype(theta.dtype)
    joint_prob = joint_prob * eligibility_mask[None, :, :]

    weighted_mass = (
        v_in_sorted[:, :, None]
        * ability_weight_term[:, None, :]
        / denom[:, None, :]
        * eligibility_mask[None, :, :]
    )
    ability_weighted_sorted = jnp.sum(weighted_mass, axis=2)
    joint_prob_sum = jnp.sum(joint_prob, axis=2)
    skill_contrib_sorted = phi_x[:, None] * joint_prob_sum + ability_weighted_sorted

    skill_sum_sorted = jnp.sum(skill_contrib_sorted, axis=0)
    L_model_sorted = jnp.sum(joint_prob_sum, axis=0)

    skill_sum_nat = jnp.zeros(J, dtype=theta.dtype).at[order_idx].set(skill_sum_sorted)
    L_model_nat = jnp.zeros(J, dtype=theta.dtype).at[order_idx].set(L_model_sorted)
    S_nat = skill_sum_nat / jnp.maximum(L_model_nat, 1e-300)

    def safe_vals(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(arr, 1e-300)

    safe_beta = jnp.clip(beta, 1e-6, 1.0 - 1e-6)
    m_vec = (
        jnp.log(safe_vals(c_nat))
        + jnp.log1p(-safe_beta)
        - jnp.log(safe_vals(w_nat))
        + jnp.log(safe_vals(Y_nat))
        - jnp.log(safe_vals(L_data))
        - jnp.log(safe_vals(S_nat))
    )

    return P_nat, per_obs_nll, m_vec, L_data, S_nat


def compute_penalty_components_with_params_jax(
    theta: jnp.ndarray,
    X: jnp.ndarray,
    choice_idx: jnp.ndarray,
    aux: Dict,
    w_nat: jnp.ndarray,
    Y_nat: jnp.ndarray,
    L_data: jnp.ndarray,
    phi: jnp.ndarray,
    sigma_a: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    aux_dynamic = dict(aux)
    aux_dynamic["phi"] = jnp.asarray(phi, dtype=jnp.float64)
    aux_dynamic["sigma_a"] = jnp.asarray(sigma_a, dtype=jnp.float64)
    return compute_penalty_components_jax(theta, X, choice_idx, aux_dynamic, w_nat, Y_nat, L_data)


__all__ = [
    "enable_x64",
    "compute_choice_probabilities_jax",
    "compute_choice_probabilities_gamma_c_V_jax",
    "compute_penalty_components_jax",
    "compute_penalty_components_with_params_jax",
]
