#!/usr/bin/env python3
"""Spec M model components in standardized parametrization.

All choice probabilities and moments use standardized parameters:
    tilde_gamma = gamma1 / sigma_e
    tilde_q_j   = (ln_qbar_j - gamma0) / sigma_e

The micro likelihood depends ONLY on (tau, tilde_gamma, delta, tilde_q).
The macro GMM moments additionally depend on (gamma0, sigma_e, alpha, eta).

See estimation.tex for derivations.
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr  # Normal CDF


# ---------------------------------------------------------------------------
# Helper: ordering maps
# ---------------------------------------------------------------------------

def _order_maps(tilde_q: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Return sorting maps for firms ordered by ascending tilde_q.

    Args:
        tilde_q: (J,) standardized cutoffs in natural firm order.

    Returns:
        Dict with keys:
            order_idx:    (J,) natural→sorted index mapping
            inv_order:    (J,) sorted→natural position mapping
            tilde_q_sorted: (J,) cutoffs in ascending order
    """
    order_idx = jnp.argsort(tilde_q)
    inv_order = jnp.argsort(order_idx)
    tilde_q_sorted = tilde_q[order_idx]
    return {
        "order_idx": order_idx,
        "inv_order": inv_order,
        "tilde_q_sorted": tilde_q_sorted,
    }


# ---------------------------------------------------------------------------
# Function 1: choice_probabilities
# ---------------------------------------------------------------------------

def choice_probabilities(
    tau: jnp.ndarray,
    tilde_gamma: jnp.ndarray,
    delta: jnp.ndarray,
    tilde_q: jnp.ndarray,
    v: jnp.ndarray,
    D: jnp.ndarray,
) -> jnp.ndarray:
    """Compute choice probabilities in standardized space.

    Implements the ordered-threshold logit from estimation.tex eq (6)-(7),
    using standardized z-values: t_j(v_i) = tilde_q_j - tilde_gamma * v_i.

    Args:
        tau:         scalar, commuting disutility (>0)
        tilde_gamma: scalar, standardized skill loading = gamma1/sigma_e
        delta:       (J,) mean utilities in natural firm order
        tilde_q:     (J,) standardized cutoffs in natural firm order
        v:           (N,) observable skill
        D:           (N, J) worker-firm distances in natural firm order

    Returns:
        P_nat: (N, J+1) choice probabilities; column 0 = outside option.
    """
    tau = jnp.asarray(tau, dtype=jnp.float64)
    tilde_gamma = jnp.asarray(tilde_gamma, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64).reshape(-1)
    tilde_q = jnp.asarray(tilde_q, dtype=jnp.float64).reshape(-1)
    v = jnp.asarray(v, dtype=jnp.float64).reshape(-1)
    D = jnp.asarray(D, dtype=jnp.float64)

    N = v.shape[0]
    J = delta.shape[0]

    # Sort firms by ascending tilde_q
    maps = _order_maps(tilde_q)
    order_idx = maps["order_idx"]
    inv_order = maps["inv_order"]
    tilde_q_sorted = maps["tilde_q_sorted"]

    delta_sorted = delta[order_idx]
    D_sorted = D[:, order_idx]

    # Logit utilities: outside option = 1, inside = exp(delta_j - tau*d_ij)
    v_out = jnp.ones((N, 1), dtype=jnp.float64)
    v_in = jnp.exp(delta_sorted[None, :] - tau * D_sorted)
    v_all = jnp.concatenate((v_out, v_in), axis=1)  # (N, J+1)

    # Cumulative denominator
    denom = jnp.cumsum(v_all, axis=1)  # (N, J+1)

    # Padded cutoffs: [-inf, tilde_q_sorted, +inf]
    big = jnp.array(1e6, dtype=jnp.float64)
    tilde_q_pad = jnp.concatenate(
        (jnp.array([-big]), tilde_q_sorted, jnp.array([big]))
    )  # (J+2,)

    # z-values in standardized space (no gamma0, no sigma_e division)
    z_hi = tilde_q_pad[1:][None, :] - tilde_gamma * v[:, None]  # (N, J+1)
    z_lo = tilde_q_pad[:-1][None, :] - tilde_gamma * v[:, None]  # (N, J+1)

    # Interval probabilities
    p_x = ndtr(z_hi) - ndtr(z_lo)  # (N, J+1)
    p_x = jnp.clip(p_x, 0.0, 1.0)
    p_row_sum = jnp.sum(p_x, axis=1, keepdims=True)
    p_x = p_x / jnp.maximum(p_row_sum, 1e-300)

    # Suffix sum: Q[i,k] = sum_{m>=k} p_x[i,m] / denom[i,m]
    q = p_x / jnp.maximum(denom, 1e-300)
    suffix_sum = jnp.flip(jnp.cumsum(jnp.flip(q, axis=1), axis=1), axis=1)

    # Choice probabilities in sorted order
    P_byc = v_all * suffix_sum  # (N, J+1)

    # Reorder to natural firm order (col 0 = outside stays at 0)
    idx_nat = jnp.concatenate((
        jnp.array([0], dtype=jnp.int32),
        inv_order.astype(jnp.int32) + 1,
    ))
    P_nat = P_byc[:, idx_nat]

    # Final clip and normalize
    P_nat = jnp.clip(P_nat, 1e-300, 1.0)
    row_sum = jnp.sum(P_nat, axis=1, keepdims=True)
    P_nat = P_nat / jnp.maximum(row_sum, 1e-300)

    return P_nat


# ---------------------------------------------------------------------------
# Function 2: per_obs_nll
# ---------------------------------------------------------------------------

def per_obs_nll(
    P_nat: jnp.ndarray,
    choice_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Per-observation negative log-likelihood.

    Args:
        P_nat:      (N, J+1) choice probabilities (col 0 = outside).
        choice_idx: (N,) int32, observed choice index (0 = outside, 1..J = firms).

    Returns:
        nll: (N,) negative log-likelihood per worker.
    """
    choice_idx = jnp.asarray(choice_idx, dtype=jnp.int32).reshape(-1)
    probs_chosen = jnp.take_along_axis(
        P_nat, choice_idx[:, None], axis=1
    ).squeeze(axis=1)
    return -jnp.log(jnp.clip(probs_chosen, 1e-300, 1.0))


# ---------------------------------------------------------------------------
# Function 3: compute_tilde_Q_M
# ---------------------------------------------------------------------------

def compute_tilde_Q_M(
    sigma_e: jnp.ndarray,
    tilde_gamma: jnp.ndarray,
    tau: jnp.ndarray,
    delta: jnp.ndarray,
    tilde_q: jnp.ndarray,
    v: jnp.ndarray,
    D: jnp.ndarray,
    choice_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Compute standardized average skill per firm under Spec M.

    Implements estimation.tex eq (11):
        tilde_Q_j^M(sigma_e) = (1/L_j) sum_{i: y_i=j}
            exp(sigma_e * tilde_gamma * v_i)
            * E[exp(sigma_e * tilde_e_i) | y_i=j, v_i, d_i]

    The posterior expectation uses the joint-probability-tensor approach
    (same as jax_model.compute_penalty_components_jax lines 232-265).

    Structural recovery: Q_j = exp(gamma0) * tilde_Q_j^M(sigma_e).

    Args:
        sigma_e:     scalar, structural parameter
        tilde_gamma: scalar, standardized skill loading
        tau:         scalar, commuting disutility
        delta:       (J,) mean utilities in natural order
        tilde_q:     (J,) standardized cutoffs in natural order
        v:           (N,) observable skill
        D:           (N, J) distances in natural order
        choice_idx:  (N,) int32, observed match (0=outside, 1..J=firms)

    Returns:
        tilde_Q: (J,) standardized average skill in natural firm order.
    """
    sigma_e = jnp.asarray(sigma_e, dtype=jnp.float64)
    tilde_gamma = jnp.asarray(tilde_gamma, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64).reshape(-1)
    tilde_q = jnp.asarray(tilde_q, dtype=jnp.float64).reshape(-1)
    v = jnp.asarray(v, dtype=jnp.float64).reshape(-1)
    D = jnp.asarray(D, dtype=jnp.float64)
    choice_idx = jnp.asarray(choice_idx, dtype=jnp.int32).reshape(-1)

    N = v.shape[0]
    J = delta.shape[0]

    # Sort by ascending tilde_q
    maps = _order_maps(tilde_q)
    order_idx = maps["order_idx"]
    tilde_q_sorted = maps["tilde_q_sorted"]

    delta_sorted = delta[order_idx]
    D_sorted = D[:, order_idx]

    # Logit utilities (sorted order)
    v_in_sorted = jnp.exp(delta_sorted[None, :] - tau * D_sorted)  # (N, J)
    v_out = jnp.ones((N, 1), dtype=jnp.float64)
    v_all = jnp.concatenate((v_out, v_in_sorted), axis=1)  # (N, J+1)

    # Cumulative denominator
    denom = jnp.cumsum(v_all, axis=1)  # (N, J+1)
    denom = jnp.maximum(denom, 1e-300)

    # Padded cutoffs
    big = jnp.array(1e6, dtype=jnp.float64)
    tilde_q_pad = jnp.concatenate(
        (jnp.array([-big]), tilde_q_sorted, jnp.array([big]))
    )

    # z-values (standardized — same as choice_probabilities)
    z_hi = tilde_q_pad[1:][None, :] - tilde_gamma * v[:, None]  # (N, J+1)
    z_lo = tilde_q_pad[:-1][None, :] - tilde_gamma * v[:, None]  # (N, J+1)

    # Interval probabilities
    p_x = ndtr(z_hi) - ndtr(z_lo)  # (N, J+1)
    p_x = jnp.clip(p_x, 0.0, 1.0)
    p_row_sum = jnp.sum(p_x, axis=1, keepdims=True)
    p_x = p_x / jnp.maximum(p_row_sum, 1e-12)

    # --- Shifted z-values for E[exp(sigma_e * tilde_e) | interval] ---
    # E[exp(sigma_e * e) | e in [a,b]] where e ~ N(0,1)
    #   = exp(sigma_e^2/2) * [Phi(b - sigma_e) - Phi(a - sigma_e)]
    #                        / [Phi(b) - Phi(a)]
    # We compute the unnormalized numerator (includes interval probability):
    #   exp(sigma_e * tilde_gamma * v_i + sigma_e^2/2) * p_x_shifted
    z_hi_shifted = z_hi - sigma_e  # (N, J+1)
    z_lo_shifted = z_lo - sigma_e  # (N, J+1)
    p_x_shifted = ndtr(z_hi_shifted) - ndtr(z_lo_shifted)  # (N, J+1)
    p_x_shifted = jnp.maximum(p_x_shifted, 0.0)

    # Per-worker exponential factor: exp(sigma_e * tilde_gamma * v_i + sigma_e^2/2)
    exp_factor = jnp.exp(
        sigma_e * tilde_gamma * v[:, None] + 0.5 * sigma_e ** 2
    )  # (N, J+1) broadcast

    # E_q_in_interval[i, k] = exp_factor * p_x_shifted (unnormalized)
    E_q_in_interval = exp_factor * p_x_shifted  # (N, J+1)

    # Joint probability tensor: firm j accessible only in intervals k >= rank(j)
    # joint_prob[i, j_sorted, k] = v_in[i,j] * p_x[i,k] / denom[i,k] * mask[j,k]
    interval_idx = jnp.arange(J + 1)
    firm_thresholds = (jnp.arange(J) + 1)[:, None]  # (J, 1)
    eligibility_mask = (interval_idx[None, :] >= firm_thresholds).astype(
        jnp.float64
    )  # (J, J+1)

    # Weighted skill mass per (worker, firm_sorted, interval)
    weighted_q_mass = (
        v_in_sorted[:, :, None]          # (N, J, 1)
        * E_q_in_interval[:, None, :]    # (N, 1, J+1)
        / denom[:, None, :]             # (N, 1, J+1)
        * eligibility_mask[None, :, :]   # (1, J, J+1)
    )  # (N, J, J+1)

    # Joint prob mass (for denominator / model labor counts)
    joint_prob = (
        v_in_sorted[:, :, None]
        * p_x[:, None, :]
        / denom[:, None, :]
        * eligibility_mask[None, :, :]
    )  # (N, J, J+1)

    # Sum over intervals → (N, J), then sum over workers → (J,)
    skill_contrib_sorted = jnp.sum(weighted_q_mass, axis=2)  # (N, J)
    joint_prob_sum = jnp.sum(joint_prob, axis=2)  # (N, J)

    skill_sum_sorted = jnp.sum(skill_contrib_sorted, axis=0)  # (J,)
    L_model_sorted = jnp.sum(joint_prob_sum, axis=0)  # (J,)

    # Reorder to natural firm order
    skill_sum_nat = jnp.zeros(J, dtype=jnp.float64).at[order_idx].set(
        skill_sum_sorted
    )
    L_model_nat = jnp.zeros(J, dtype=jnp.float64).at[order_idx].set(
        L_model_sorted
    )

    tilde_Q = skill_sum_nat / jnp.maximum(L_model_nat, 1e-300)
    return tilde_Q


# ---------------------------------------------------------------------------
# Function 4: compute_gmm_moments_M
# ---------------------------------------------------------------------------

def compute_gmm_moments_M(
    delta: jnp.ndarray,
    tilde_q: jnp.ndarray,
    tilde_Q: jnp.ndarray,
    gamma0: jnp.ndarray,
    sigma_e: jnp.ndarray,
    alpha: jnp.ndarray,
    eta: jnp.ndarray,
    w: jnp.ndarray,
    R: jnp.ndarray,
    L: jnp.ndarray,
    z1: jnp.ndarray,
    z2: jnp.ndarray,
    z3: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the 4 sample moments from Spec M (estimation.tex eqs 9-12).

    Args:
        delta:   (J,) mean utilities
        tilde_q: (J,) standardized cutoffs
        tilde_Q: (J,) standardized average skill (from compute_tilde_Q_M)
        gamma0:  scalar, skill intercept
        sigma_e: scalar, unobserved skill dispersion
        alpha:   scalar, output elasticity
        eta:     scalar, wage sensitivity
        w:       (J,) wages
        R:       (J,) revenue (= Y in simulation data)
        L:       (J,) labor counts
        z1:      (J,) instrument for m1 (TFP shifter)
        z2:      (J,) instrument for m2 (amenity shifter)
        z3:      (J,) instrument for m3 (e.g. tilde_q or ones)

    Returns:
        m_vec: (4,) stacked moment vector [m1, m2, m3, m4].
    """
    gamma0 = jnp.asarray(gamma0, dtype=jnp.float64)
    sigma_e = jnp.asarray(sigma_e, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    eta = jnp.asarray(eta, dtype=jnp.float64)

    ln_w = jnp.log(jnp.maximum(w, 1e-300))
    ln_R = jnp.log(jnp.maximum(R, 1e-300))
    ln_L = jnp.log(jnp.maximum(L, 1e-300))
    ln_tQ = jnp.log(jnp.maximum(tilde_Q, 1e-300))

    # m1 (preference, identifies eta): E[xi | z1] = 0
    #   xi_j = delta_j - eta * ln w_j
    m1_j = delta - eta * ln_w
    m1 = jnp.mean(z1 * m1_j)

    # m2 (screening FOC, identifies alpha): E[-ln A | z2] = 0
    #   -ln A_j = ln(1-alpha) + (1-alpha)*gamma0 + sigma_e*tilde_q_j
    #             - alpha*ln_tQ_j - alpha*ln_L_j - ln_w_j
    safe_alpha = jnp.clip(alpha, 1e-6, 1.0 - 1e-6)
    m2_j = (
        jnp.log1p(-safe_alpha)
        + (1.0 - safe_alpha) * gamma0
        + sigma_e * tilde_q
        - safe_alpha * ln_tQ
        - safe_alpha * ln_L
        - ln_w
    )
    m2 = jnp.mean(z2 * m2_j)

    # m3 (A-free/revenue, identifies sigma_e; gamma0 cancels)
    #   = sigma_e*tilde_q - ln_tQ - ln_w - ln_L + ln(1-alpha) + ln_R
    m3_j = (
        sigma_e * tilde_q
        - ln_tQ
        - ln_w
        - ln_L
        + jnp.log1p(-safe_alpha)
        + ln_R
    )
    m3 = jnp.mean(z3 * m3_j)

    # m4 (TFP normalization, identifies gamma0): m2 with z=1
    m4 = jnp.mean(m2_j)

    return jnp.array([m1, m2, m3, m4])


# ---------------------------------------------------------------------------
# Function 5: compute_penalty
# ---------------------------------------------------------------------------

def compute_penalty(
    m_vec: jnp.ndarray,
    W: jnp.ndarray,
) -> jnp.ndarray:
    """Quadratic GMM penalty: 0.5 * m' W m.

    Args:
        m_vec: (K,) moment vector.
        W:     (K, K) positive-definite weighting matrix.

    Returns:
        Scalar penalty value.
    """
    return 0.5 * m_vec @ W @ m_vec


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "choice_probabilities",
    "per_obs_nll",
    "compute_tilde_Q_M",
    "compute_gmm_moments_M",
    "compute_penalty",
]
