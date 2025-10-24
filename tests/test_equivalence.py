import os
import math
import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


from code.helpers import (
    compute_worker_firm_distances,
    compute_choice_probabilities as compute_choice_probabilities_np,
    compute_cutoffs,
    chamberlain_instruments_numeric,
)
from code.jax_model import compute_choice_probabilities_jax, enable_x64
from code.g_features import chamberlain_instruments_jax
from code.moments import criterion as criterion_jax


def _clip_and_renorm(P):
    P = np.clip(P, 1e-300, 1.0)
    rs = P.sum(axis=1, keepdims=True)
    return P / np.maximum(rs, 1e-300)


def _c_to_a(c, w, Y, beta):
    eps = 1e-12
    c = np.asarray(c, dtype=float)
    w = np.asarray(w, dtype=float)
    Y = np.asarray(Y, dtype=float)
    beta = float(beta)
    log_w_term = np.log(np.maximum(w, eps)) - np.log(1.0 - beta)
    log_Y_term = np.log(np.maximum(Y, eps))
    log_c_term = np.log(np.maximum(c, eps))
    log_A = (1.0 - beta) * log_w_term + beta * log_Y_term - (1.0 - beta) * log_c_term
    return np.exp(log_A)


def _build_small_synthetic(N=5, J=3, seed=0):
    rng = np.random.default_rng(seed)
    w = np.exp(rng.normal(size=J)) + 0.5
    xi = rng.normal(scale=0.1, size=J)
    Y = np.exp(rng.normal(size=J)) + 0.5
    A_nat = np.exp(rng.normal(size=J)) + 0.5
    firm_ids = np.arange(1, J + 1)
    alpha = 1.3
    beta = 0.2
    phi = 1.0
    mu_a = 0.0
    sigma_a = 1.5
    gamma = 0.07
    V_nat = alpha * np.log(np.maximum(w, 1e-300)) + xi

    # Worker data
    x_skill = rng.normal(loc=10.0, scale=2.0, size=N)
    ell_x = rng.normal(size=N)
    ell_y = rng.normal(size=N)
    loc_firms = rng.normal(size=(J, 2))

    D_nat = compute_worker_firm_distances(ell_x, ell_y, loc_firms)

    c_nat = compute_cutoffs(w, Y, A_nat, beta)

    aux = {
        'D_nat': jnp.asarray(D_nat, dtype=jnp.float64),
        'phi': float(phi),
        'mu_a': float(mu_a),
        'sigma_a': float(sigma_a),
        'firm_ids': jnp.asarray(firm_ids, dtype=jnp.int32),
    }

    return {
        'N': N,
        'J': J,
        'alpha': alpha,
        'beta': beta,
        'phi': phi,
        'mu_a': mu_a,
        'sigma_a': sigma_a,
        'gamma': gamma,
        'V_nat': V_nat,
        'A_nat': A_nat,
        'c_nat': c_nat,
        'w': w,
        'Y': Y,
        'firm_ids': firm_ids,
        'x_skill': x_skill,
        'D_nat': D_nat,
        'aux': aux,
    }


@pytest.mark.slow
def test_probabilities_equivalence_small():
    enable_x64()
    data = _build_small_synthetic(N=5, J=3, seed=1)
    N = data['N']; J = data['J']

    # NumPy reference
    P_np = compute_choice_probabilities_np(
        data['gamma'],
        data['V_nat'],
        data['D_nat'],
        data['w'], data['Y'], data['A_nat'],
        c=np.ones(J),  # unused internally
        x_skill=data['x_skill'],
        firm_ids=data['firm_ids'],
        beta=data['beta'], phi=data['phi'], mu_a=data['mu_a'], sigma_a=data['sigma_a'],
        V_nat=data['V_nat'],
    )

    # JAX
    theta = jnp.asarray(np.concatenate(([data['gamma']], data['V_nat'], data['c_nat'])), dtype=jnp.float64)
    X = jnp.asarray(data['x_skill'], dtype=jnp.float64)
    P_jax = np.asarray(compute_choice_probabilities_jax(theta, X, data['aux']))

    # Clip+renorm then compare
    P_np_c = _clip_and_renorm(P_np)
    P_jax_c = _clip_and_renorm(P_jax)

    assert P_np_c.shape == (N, J + 1)
    assert P_jax_c.shape == (N, J + 1)
    # Row sums sanity
    assert np.allclose(P_jax_c.sum(axis=1), 1.0, atol=1e-12)

    diff = np.max(np.abs(P_np_c - P_jax_c))
    assert diff <= 1e-9, f"max|ΔP|={diff}"


def test_chamberlain_instruments_match_numeric():
    enable_x64()
    data = _build_small_synthetic(N=6, J=3, seed=7)
    N = data['N']; J = data['J']

    theta_np = np.concatenate(([data['gamma']], data['V_nat'], data['c_nat'])).astype(np.float64)
    theta_j = jnp.asarray(theta_np, dtype=jnp.float64)
    X = jnp.asarray(data['x_skill'], dtype=jnp.float64)

    def prob_eval_jax(theta_arr: jnp.ndarray) -> jnp.ndarray:
        return compute_choice_probabilities_jax(theta_arr, X, data['aux'])

    G_jax = np.asarray(chamberlain_instruments_jax(theta_j, prob_eval_jax, check_rowsum=True))

    def prob_eval_np(theta_arr: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta_arr, dtype=np.float64).ravel()
        gamma = theta_arr[0]
        V_nat = theta_arr[1:1+J]
        c_nat = theta_arr[1+J:1+2*J]
        A_nat = _c_to_a(c_nat, data['w'], data['Y'], data['beta'])
        return compute_choice_probabilities_np(
            gamma,
            data['V_nat'],
            data['D_nat'],
            data['w'],
            data['Y'],
            A_nat,
            c=np.ones(J),
            x_skill=data['x_skill'],
            firm_ids=data['firm_ids'],
            beta=data['beta'],
            phi=data['phi'],
            mu_a=data['mu_a'],
            sigma_a=data['sigma_a'],
            V_nat=V_nat,
        )

    G_num = chamberlain_instruments_numeric(theta_np, prob_eval_np)

    assert G_jax.shape == (N, J, theta_np.size)
    assert G_num.shape == G_jax.shape
    assert np.allclose(G_jax, G_num, atol=1e-6)


@pytest.mark.slow
def test_criterion_grad_matches_fd():
    enable_x64()
    data = _build_small_synthetic(N=8, J=4, seed=3)
    N = data['N']; J = data['J']

    theta = np.concatenate(([data['gamma']], data['V_nat'], data['c_nat'])).astype(np.float64)
    theta_j = jnp.asarray(theta, dtype=jnp.float64)

    X = jnp.asarray(data['x_skill'], dtype=jnp.float64)

    def prob_eval_jax(theta_arr: jnp.ndarray) -> jnp.ndarray:
        return compute_choice_probabilities_jax(theta_arr, X, data['aux'])

    G_feat = chamberlain_instruments_jax(theta_j, prob_eval_jax, check_rowsum=True)

    # Random choices Y (one-hot with outside at 0)
    rng = np.random.default_rng(5)
    chosen = rng.integers(low=0, high=J + 1, size=N)
    Y = np.zeros((N, J + 1))
    Y[np.arange(N), chosen] = 1.0

    # JAX gradient
    Y_j = jnp.asarray(Y, dtype=jnp.float64)
    gfun = jax.grad(lambda th: criterion_jax(th, X, Y_j, G_feat, data['aux']))
    g_jax = np.asarray(gfun(theta_j))

    # Finite-difference gradient using NumPy reference criterion
    def crit_np(theta_np: np.ndarray) -> float:
        c_nat = theta_np[1+J:]
        A_nat = _c_to_a(c_nat, data['w'], data['Y'], data['beta'])
        P_np = compute_choice_probabilities_np(
            theta_np[0],
            data['V_nat'],
            data['D_nat'],
            data['w'], data['Y'], A_nat,
            c=np.ones(J),
            x_skill=data['x_skill'],
            firm_ids=data['firm_ids'],
            beta=data['beta'], phi=data['phi'], mu_a=data['mu_a'], sigma_a=data['sigma_a'],
            V_nat=theta_np[1:1+J],
        )
        R = Y[:, 1:] - P_np[:, 1:]
        # G_feat dense value
        Gd = np.asarray(G_feat)
        m = np.tensordot(R, Gd, axes=([0, 1], [0, 1]))
        return float(m @ m)

    eps = 1e-6
    idxs = [0, 1, 1 + J // 2, 1 + J + (J // 2)]  # sample across gamma, V, A
    for k in idxs:
        e = np.zeros_like(theta)
        e[k] = 1.0
        f_p = crit_np(theta + eps * e)
        f_m = crit_np(theta - eps * e)
        g_fd = (f_p - f_m) / (2 * eps)
        assert np.isfinite(g_jax[k])
        assert abs(g_jax[k] - g_fd) <= 1e-6, f"k={k}, jax={g_jax[k]:.3e}, fd={g_fd:.3e}"
