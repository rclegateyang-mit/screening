#!/usr/bin/env python3
"""Verification tests for model_components.py against jax_model.py.

Tests:
1. Numerical equivalence of choice probabilities (standardized vs structural).
2. Numerical equivalence of tilde_Q_M vs Q_nat from jax_model.
3. Moment sanity checks at true DGP parameters.
4. Gradient checks (no NaN through choice_probabilities and compute_tilde_Q_M).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from screening.analysis.lib.model_components import (
    choice_probabilities,
    per_obs_nll,
    compute_tilde_Q_M,
    compute_gmm_moments_M,
    compute_penalty,
)
from screening.analysis.lib.jax_model import (
    _choice_probabilities_from_cutoffs,
    compute_penalty_components_jax,
)


def _make_test_data(N=200, J=10, seed=42):
    """Generate synthetic test data."""
    rng = np.random.default_rng(seed)

    # True structural params
    tau = 0.4
    gamma0 = 0.76
    gamma1 = 0.94
    sigma_e = 0.135
    alpha = 0.2
    eta = 7.0

    # Standardized params
    tilde_gamma = gamma1 / sigma_e

    # Firm data
    delta = rng.normal(1.0, 0.5, size=J)
    ln_qbar = rng.normal(gamma0, 0.3, size=J)
    tilde_q = (ln_qbar - gamma0) / sigma_e

    # Worker data
    v = rng.normal(12.0, 0.25, size=N)
    D = rng.uniform(0.0, 2.0, size=(N, J))

    # Generate choices (simple: assign to highest-utility firm with screening)
    choice_idx = rng.integers(0, J + 1, size=N).astype(np.int32)

    # Firm observables for moments
    w = rng.uniform(5.0, 15.0, size=J)
    R = rng.uniform(100.0, 500.0, size=J)
    L = rng.uniform(10.0, 100.0, size=J)
    z1 = rng.normal(0.0, 1.0, size=J)
    z2 = rng.normal(0.0, 1.0, size=J)
    z3 = rng.normal(0.0, 1.0, size=J)

    return {
        "tau": tau, "gamma0": gamma0, "gamma1": gamma1,
        "sigma_e": sigma_e, "alpha": alpha, "eta": eta,
        "tilde_gamma": tilde_gamma, "tilde_q": tilde_q,
        "delta": delta, "ln_qbar": ln_qbar,
        "v": v, "D": D, "choice_idx": choice_idx,
        "w": w, "R": R, "L": L,
        "z1": z1, "z2": z2, "z3": z3,
    }


def test_choice_probabilities_equivalence():
    """Verify new choice_probabilities matches _choice_probabilities_from_cutoffs."""
    d = _make_test_data()

    # New code (standardized)
    P_new = choice_probabilities(
        tau=jnp.array(d["tau"]),
        tilde_gamma=jnp.array(d["tilde_gamma"]),
        delta=jnp.array(d["delta"]),
        tilde_q=jnp.array(d["tilde_q"]),
        v=jnp.array(d["v"]),
        D=jnp.array(d["D"]),
    )

    # Old code (structural)
    aux = {
        "D_nat": jnp.array(d["D"]),
        "gamma0": d["gamma0"],
        "gamma1": d["gamma1"],
        "sigma_e": d["sigma_e"],
    }
    P_old = _choice_probabilities_from_cutoffs(
        tau=jnp.array(d["tau"]),
        delta_nat=jnp.array(d["delta"]),
        qbar_nat=jnp.array(d["ln_qbar"]),
        X=jnp.array(d["v"]),
        aux=aux,
    )

    max_diff = float(jnp.max(jnp.abs(P_new - P_old)))
    print(f"  Choice prob max |diff|: {max_diff:.2e}")
    assert max_diff < 1e-12, f"Choice probs differ by {max_diff}"
    print("  PASSED")


def test_per_obs_nll():
    """Verify per_obs_nll produces correct values."""
    d = _make_test_data()

    P = choice_probabilities(
        tau=jnp.array(d["tau"]),
        tilde_gamma=jnp.array(d["tilde_gamma"]),
        delta=jnp.array(d["delta"]),
        tilde_q=jnp.array(d["tilde_q"]),
        v=jnp.array(d["v"]),
        D=jnp.array(d["D"]),
    )

    nll = per_obs_nll(P, jnp.array(d["choice_idx"]))
    assert nll.shape == (d["v"].shape[0],), f"Shape mismatch: {nll.shape}"
    assert jnp.all(jnp.isfinite(nll)), "NaN/Inf in NLL"
    assert jnp.all(nll >= 0), "Negative NLL values"

    # Cross-check: manual computation
    choice_idx = jnp.array(d["choice_idx"])
    probs_chosen = jnp.take_along_axis(P, choice_idx[:, None], axis=1).squeeze(1)
    nll_manual = -jnp.log(jnp.clip(probs_chosen, 1e-300, 1.0))
    max_diff = float(jnp.max(jnp.abs(nll - nll_manual)))
    assert max_diff < 1e-14, f"NLL manual check diff: {max_diff}"
    print("  PASSED")


def test_tilde_Q_M_equivalence():
    """Verify exp(gamma0) * tilde_Q_M matches Q_nat from jax_model."""
    d = _make_test_data()
    J = len(d["delta"])

    # Old code: compute_penalty_components_jax
    # theta = [tau, alpha, delta(J), ln_qbar(J)]
    theta_old = jnp.concatenate([
        jnp.array([d["tau"], d["alpha"]]),
        jnp.array(d["delta"]),
        jnp.array(d["ln_qbar"]),
    ])
    aux = {
        "D_nat": jnp.array(d["D"]),
        "gamma0": d["gamma0"],
        "gamma1": d["gamma1"],
        "sigma_e": d["sigma_e"],
    }
    _, _, _, _, Q_old = compute_penalty_components_jax(
        theta_old,
        jnp.array(d["v"]),
        jnp.array(d["choice_idx"]),
        aux,
        jnp.array(d["w"]),
        jnp.array(d["R"]),
        jnp.array(d["L"]),
    )

    # New code
    tilde_Q = compute_tilde_Q_M(
        sigma_e=jnp.array(d["sigma_e"]),
        tilde_gamma=jnp.array(d["tilde_gamma"]),
        tau=jnp.array(d["tau"]),
        delta=jnp.array(d["delta"]),
        tilde_q=jnp.array(d["tilde_q"]),
        v=jnp.array(d["v"]),
        D=jnp.array(d["D"]),
        choice_idx=jnp.array(d["choice_idx"]),
    )

    # Structural recovery: Q_j = exp(gamma0) * tilde_Q_j
    Q_new = jnp.exp(d["gamma0"]) * tilde_Q

    max_diff = float(jnp.max(jnp.abs(Q_new - Q_old)))
    rel_diff = float(jnp.max(jnp.abs(Q_new - Q_old) / jnp.maximum(jnp.abs(Q_old), 1e-300)))
    print(f"  Q max |diff|: {max_diff:.2e}, max |rel diff|: {rel_diff:.2e}")
    assert max_diff < 1e-10, f"Q values differ by {max_diff}"
    print("  PASSED")


def test_gmm_moments_shapes():
    """Verify moment computation produces correct shapes and finite values."""
    d = _make_test_data()

    tilde_Q = compute_tilde_Q_M(
        sigma_e=jnp.array(d["sigma_e"]),
        tilde_gamma=jnp.array(d["tilde_gamma"]),
        tau=jnp.array(d["tau"]),
        delta=jnp.array(d["delta"]),
        tilde_q=jnp.array(d["tilde_q"]),
        v=jnp.array(d["v"]),
        D=jnp.array(d["D"]),
        choice_idx=jnp.array(d["choice_idx"]),
    )

    m_vec = compute_gmm_moments_M(
        delta=jnp.array(d["delta"]),
        tilde_q=jnp.array(d["tilde_q"]),
        tilde_Q=tilde_Q,
        gamma0=jnp.array(d["gamma0"]),
        sigma_e=jnp.array(d["sigma_e"]),
        alpha=jnp.array(d["alpha"]),
        eta=jnp.array(d["eta"]),
        w=jnp.array(d["w"]),
        R=jnp.array(d["R"]),
        L=jnp.array(d["L"]),
        z1=jnp.array(d["z1"]),
        z2=jnp.array(d["z2"]),
        z3=jnp.array(d["z3"]),
    )

    assert m_vec.shape == (4,), f"Shape mismatch: {m_vec.shape}"
    assert jnp.all(jnp.isfinite(m_vec)), f"Non-finite moments: {m_vec}"
    print(f"  Moments: {m_vec}")
    print("  PASSED")


def test_compute_penalty():
    """Verify penalty computation."""
    m = jnp.array([0.1, -0.2, 0.05, 0.3])
    W = jnp.eye(4)
    p = compute_penalty(m, W)
    expected = 0.5 * jnp.sum(m ** 2)
    assert abs(float(p - expected)) < 1e-14
    print("  PASSED")


def test_gradient_choice_probs():
    """Verify jax.grad through choice_probabilities produces no NaN."""
    d = _make_test_data(N=50, J=5)

    def total_nll(tau, tg, delta, tq):
        P = choice_probabilities(tau, tg, delta, tq,
                                 jnp.array(d["v"]), jnp.array(d["D"]))
        return jnp.sum(per_obs_nll(P, jnp.array(d["choice_idx"])))

    grads = jax.grad(total_nll, argnums=(0, 1, 2, 3))(
        jnp.array(d["tau"]),
        jnp.array(d["tilde_gamma"]),
        jnp.array(d["delta"][:5]),
        jnp.array(d["tilde_q"][:5]),
    )

    names = ["tau", "tilde_gamma", "delta", "tilde_q"]
    for name, g in zip(names, grads):
        assert jnp.all(jnp.isfinite(g)), f"NaN/Inf in grad w.r.t. {name}: {g}"
        print(f"  grad/{name}: finite, norm={float(jnp.linalg.norm(g)):.4e}")
    print("  PASSED")


def test_gradient_tilde_Q():
    """Verify jax.grad through compute_tilde_Q_M produces no NaN."""
    d = _make_test_data(N=50, J=5)

    def sum_tQ(sigma_e, tg, tau, delta, tq):
        tQ = compute_tilde_Q_M(sigma_e, tg, tau, delta, tq,
                                jnp.array(d["v"]),
                                jnp.array(d["D"][:, :5]),
                                jnp.array(d["choice_idx"]))
        return jnp.sum(tQ)

    grads = jax.grad(sum_tQ, argnums=(0, 1, 2, 3, 4))(
        jnp.array(d["sigma_e"]),
        jnp.array(d["tilde_gamma"]),
        jnp.array(d["tau"]),
        jnp.array(d["delta"][:5]),
        jnp.array(d["tilde_q"][:5]),
    )

    names = ["sigma_e", "tilde_gamma", "tau", "delta", "tilde_q"]
    for name, g in zip(names, grads):
        assert jnp.all(jnp.isfinite(g)), f"NaN/Inf in grad w.r.t. {name}: {g}"
        print(f"  grad/{name}: finite, norm={float(jnp.linalg.norm(g)):.4e}")
    print("  PASSED")


if __name__ == "__main__":
    tests = [
        ("Choice prob equivalence", test_choice_probabilities_equivalence),
        ("Per-obs NLL", test_per_obs_nll),
        ("tilde_Q_M equivalence", test_tilde_Q_M_equivalence),
        ("GMM moments shapes", test_gmm_moments_shapes),
        ("Penalty computation", test_compute_penalty),
        ("Gradient: choice probs", test_gradient_choice_probs),
        ("Gradient: tilde_Q_M", test_gradient_tilde_Q),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)
