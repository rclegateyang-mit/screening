#!/usr/bin/env python3
"""Accuracy verification for the augmented-Lagrangian hybrid MLE+GMM solver.

Uses data_v2/ which has z1/z2 instruments.
True params: tau=0.4, gamma0=0.5, gamma1=0.94, sigma_e=0.135, alpha=0.2, eta=7.0.

Run::
    cd code
    python -m tests.test_hybrid_auglag
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# JAX init
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                      "intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
jax.config.update("jax_enable_x64", True)

from screening.analysis.auglag.run_pooled import (
    load_hybrid_data,
    init_from_true,
    init_naive,
    compute_all_moments,
    run_auglag,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
FIRMS = os.path.join(BASE, "data_v2", "clean", "equilibrium_firms.csv")
WORKERS = os.path.join(BASE, "data_v2", "build", "workers_dataset.csv")
PARAMS = os.path.join(BASE, "data_v2", "raw", "parameters_effective.csv")

NAMES = ['tau', 'tilde_gamma', 'alpha', 'sigma_e', 'eta', 'gamma0']


def _true_vec(tp):
    return np.array([tp['tau'], tp['tilde_gamma'], tp['alpha'],
                     tp['sigma_e'], tp['eta'], tp['gamma0']])


def _print_recovery(theta_G, tp, deltas, tilde_qs, meta, market_datas, g_bar):
    true_vec = _true_vec(tp)
    print("\n  Parameter recovery:")
    for name, hat, true in zip(NAMES, theta_G, true_vec):
        print(f"    {name:14s}: hat={hat:.6f}  true={true:.6f}  err={hat-true:+.6f}")

    for i, md in enumerate(market_datas):
        mf = meta['meta_firms'][i]
        delta_true = tp['eta'] * np.log(np.maximum(mf['w'], 1e-300)) + mf['xi']
        ln_qbar_true = np.log(np.maximum(mf['qbar'], 1e-300))
        tq_true = (ln_qbar_true - tp['gamma0']) / tp['sigma_e']
        d_corr = float(np.corrcoef(deltas[i], delta_true)[0, 1])
        tq_corr = float(np.corrcoef(tilde_qs[i], tq_true)[0, 1])
        print(f"    Market {md.market_id}: delta_corr={d_corr:.4f}  tq_corr={tq_corr:.4f}")

    print(f"    g_bar = {g_bar}")
    print(f"    |g_bar| = {np.linalg.norm(g_bar):.4e}")


# ---------------------------------------------------------------------------
# Test 1: True-init, M=2
# ---------------------------------------------------------------------------

def test_true_init_M2():
    print("\n" + "=" * 60)
    print("TEST 1: True-init, M=2")
    print("=" * 60)

    market_datas, meta = load_hybrid_data(FIRMS, WORKERS, PARAMS, M_subset=2)
    tp = meta['true_params']
    print(f"  M={meta['M']}, J per market: {[md.J for md in market_datas]}")

    theta_G, deltas, tilde_qs = init_from_true(market_datas, meta)
    print(f"  theta_G_init = {theta_G}")

    W = np.eye(4, dtype=np.float64)

    t0 = time.perf_counter()
    result = run_auglag(
        market_datas, W, theta_G, deltas, tilde_qs,
        max_outer_iter=50,
        inner_maxiter=200,
        inner_tol=1e-6,
        global_maxiter=100,
        global_tol=1e-5,
        verbose=True,
    )
    wall = time.perf_counter() - t0

    print(f"\n  Converged: {result['converged']}")
    print(f"  Iterations: {result['n_outer_iters']}")
    print(f"  Wall time: {wall:.1f}s")
    print(f"  Objective: {result['obj']:.4f}")

    _print_recovery(result['theta_G'], tp, result['deltas'], result['tilde_qs'],
                    meta, market_datas, result['g_bar'])

    # Assertions
    true_vec = _true_vec(tp)
    errors = np.abs(result['theta_G'] - true_vec)
    print(f"\n  Max global param error: {errors.max():.6f}")
    return result


# ---------------------------------------------------------------------------
# Test 2: Perturbed init, M=2
# ---------------------------------------------------------------------------

def test_perturbed_init_M2():
    print("\n" + "=" * 60)
    print("TEST 2: Perturbed init, M=2")
    print("=" * 60)

    market_datas, meta = load_hybrid_data(FIRMS, WORKERS, PARAMS, M_subset=2)
    tp = meta['true_params']

    theta_G_true, deltas_true, tilde_qs_true = init_from_true(market_datas, meta)

    # Perturb by 10-20%
    theta_G_perturbed = theta_G_true.copy()
    theta_G_perturbed[0] = 0.35   # tau: 0.4 -> 0.35
    theta_G_perturbed[2] = 0.25   # alpha: 0.2 -> 0.25
    theta_G_perturbed[3] = 0.15   # sigma_e: 0.135 -> 0.15
    theta_G_perturbed[4] = 7.5    # eta: 7.0 -> 7.5
    theta_G_perturbed[5] = 0.55   # gamma0: 0.5 -> 0.55

    # Perturb deltas/tilde_qs slightly
    deltas_perturbed = [d * 1.05 + 0.1 for d in deltas_true]
    tilde_qs_perturbed = [q * 0.95 - 0.5 for q in tilde_qs_true]

    print(f"  theta_G_init (perturbed) = {theta_G_perturbed}")
    print(f"  theta_G_true             = {theta_G_true}")

    W = np.eye(4, dtype=np.float64)

    t0 = time.perf_counter()
    result = run_auglag(
        market_datas, W, theta_G_perturbed, deltas_perturbed, tilde_qs_perturbed,
        max_outer_iter=100,
        inner_maxiter=200,
        inner_tol=1e-6,
        global_maxiter=100,
        global_tol=1e-5,
        verbose=True,
    )
    wall = time.perf_counter() - t0

    print(f"\n  Converged: {result['converged']}")
    print(f"  Iterations: {result['n_outer_iters']}")
    print(f"  Wall time: {wall:.1f}s")

    _print_recovery(result['theta_G'], tp, result['deltas'], result['tilde_qs'],
                    meta, market_datas, result['g_bar'])

    # Check tau recovery
    tau_err = abs(result['theta_G'][0] - tp['tau'])
    print(f"\n  |tau_hat - tau_true| = {tau_err:.4f}  (target < 0.05)")

    # Check delta correlation
    for i, md in enumerate(market_datas):
        mf = meta['meta_firms'][i]
        delta_true = tp['eta'] * np.log(np.maximum(mf['w'], 1e-300)) + mf['xi']
        d_corr = float(np.corrcoef(result['deltas'][i], delta_true)[0, 1])
        print(f"  Market {md.market_id} delta_corr = {d_corr:.4f}  (target > 0.95)")

    return result


# ---------------------------------------------------------------------------
# Test 3: Naive init, M=5
# ---------------------------------------------------------------------------

def test_naive_init_M5():
    print("\n" + "=" * 60)
    print("TEST 3: Naive init, M=5")
    print("=" * 60)

    market_datas, meta = load_hybrid_data(FIRMS, WORKERS, PARAMS, M_subset=5)
    tp = meta['true_params']
    print(f"  M={meta['M']}, J per market: {[md.J for md in market_datas]}")

    theta_G, deltas, tilde_qs = init_naive(market_datas, meta)
    print(f"  theta_G_init (naive) = {theta_G}")
    print(f"  theta_G_true         = {_true_vec(tp)}")

    W = np.eye(4, dtype=np.float64)

    t0 = time.perf_counter()
    result = run_auglag(
        market_datas, W, theta_G, deltas, tilde_qs,
        max_outer_iter=100,
        inner_maxiter=200,
        inner_tol=1e-6,
        global_maxiter=100,
        global_tol=1e-5,
        verbose=True,
    )
    wall = time.perf_counter() - t0

    print(f"\n  Converged: {result['converged']}")
    print(f"  Iterations: {result['n_outer_iters']}")
    print(f"  Wall time: {wall:.1f}s")

    _print_recovery(result['theta_G'], tp, result['deltas'], result['tilde_qs'],
                    meta, market_datas, result['g_bar'])

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Hybrid AugLag Accuracy Tests")
    print(f"Data: {FIRMS}")

    # Run tests sequentially
    r1 = test_true_init_M2()
    r2 = test_perturbed_init_M2()
    r3 = test_naive_init_M5()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
