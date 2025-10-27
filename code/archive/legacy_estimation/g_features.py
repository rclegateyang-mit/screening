#!/usr/bin/env python3
"""
Chamberlain optimal instruments implemented with JAX autodiff utilities.

This module provides a differentiable analogue of
``helpers.chamberlain_instruments_numeric`` by computing the Jacobian of the
log-odds map using JAX instead of finite differences. Given a probability
evaluator ``prob_evaluator`` that returns row-stochastic matrices
``P(theta)`` with outside option in column 0, the Chamberlain instruments are

    G[n, j, k] = ∂/∂θ_k { log P_{n, j+1}(theta) − log P_{n, 0}(theta) }.

Exports
-------
- ``chamberlain_instruments_jax(theta, prob_evaluator, *, eps=1e-300,
   mode="auto", chunk_size=128, check_rowsum=False)`` → ``G`` with shape
   ``(N, J, K)``.
- ``to_bcoo(G, density_threshold=0.15)`` to optionally convert dense arrays to
   sparse ``BCOO`` tensors.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax import nn as jnn


Array = jax.Array
ProbEvaluator = Callable[[Array], Array]


def _log_odds(theta: Array, prob_evaluator: ProbEvaluator, eps: float) -> Array:
    """Return vec(log P[:,1:] − log P[:,0]) for probabilities P(theta)."""
    P = prob_evaluator(theta)
    P = jnp.asarray(P, dtype=jnp.float64)
    if P.ndim != 2:
        raise ValueError(f"prob_evaluator must return a 2D array; got shape {P.shape}")
    if P.shape[1] < 2:
        raise ValueError(f"prob_evaluator must return at least two columns; got {P.shape[1]}")

    log_P_firms = jnp.log(jnp.clip(P[:, 1:], eps, 1.0))
    log_P_outside = jnp.log(jnp.clip(P[:, 0:1], eps, 1.0))
    log_odds = log_P_firms - log_P_outside
    return log_odds.reshape((-1,))


def _jacobian_full(F: Callable[[Array], Array], theta: Array) -> Array:
    """Compute Jacobian using ``jax.jacfwd`` (cost ~K forward passes)."""
    return jax.jacfwd(F)(theta)


def _jacobian_chunked(
    F: Callable[[Array], Array],
    theta: Array,
    K: int,
    *,
    chunk_size: int = 128,
) -> Array:
    """Compute Jacobian by batching JVPs over slices of the identity basis."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")

    def jvp_one(v: Array) -> Array:
        _, dF = jax.jvp(F, (theta,), (v,))
        return dF

    jvp_many = jax.jit(jax.vmap(jvp_one, in_axes=(0,)))

    blocks = []
    for start in range(0, K, chunk_size):
        stop = min(start + chunk_size, K)
        idx = jnp.arange(start, stop)
        basis_block = jnn.one_hot(idx, K, dtype=theta.dtype)
        dF_block = jvp_many(basis_block)
        blocks.append(dF_block)

    if not blocks:
        return jnp.zeros((F(theta).size, 0), dtype=theta.dtype)

    J = jnp.concatenate(blocks, axis=0).T
    return J


def _jacobian_finite_diff(
    F: Callable[[Array], Array],
    theta: Array,
    rel_step: float,
    abs_step: float,
) -> Array:
    theta = jnp.asarray(theta, dtype=jnp.float64)
    K = theta.size
    step = jnp.maximum(abs_step, rel_step * jnp.maximum(1.0, jnp.abs(theta)))
    eye = jnp.eye(K, dtype=theta.dtype)

    def fd_column(e_k, delta):
        fp = F(theta + delta * e_k)
        fm = F(theta - delta * e_k)
        return (fp - fm) / (2.0 * delta)

    cols = jax.vmap(fd_column)(eye, step)
    return cols.T


def chamberlain_instruments_jax(
    theta: Array,
    prob_evaluator: ProbEvaluator,
    *,
    eps: float = 1e-300,
    mode: str = "auto",
    chunk_size: int = 128,
    check_rowsum: bool = False,
    fd_rel_step: float = 1e-5,
    fd_abs_step: float = 1e-6,
) -> Array:
    """Compute Chamberlain optimal instruments using automatic differentiation.

    Parameters
    ----------
    theta : array_like, shape (K,)
        Parameter vector supplied to ``prob_evaluator``.
    prob_evaluator : callable
        Pure JAX function returning ``P`` with shape ``(N, J+1)``. Column 0 must
        correspond to the outside option and each row should sum to one prior to
        clipping.
    eps : float, default 1e-300
        Lower bound applied before logarithms for numerical stability.
    mode : {"auto", "full", "chunked", "fd"}, default "auto"
        Jacobian evaluation strategy. ``"full"`` uses ``jax.jacfwd``; ``"chunked"``
        runs batched JVPs with ``chunk_size`` columns at a time; ``"fd"`` uses
        symmetric finite differences. ``"auto"`` picks ``"full"`` if ``K <= 256``
        else ``"chunked"``.
    chunk_size : int, default 128
        Number of columns processed per batch in ``"chunked"`` mode.
    check_rowsum : bool, default False
        If set, verify that the raw probabilities sum to one along each row.
    fd_rel_step : float, default 1e-5
        Relative step size for finite differences (``mode="fd"``).
    fd_abs_step : float, default 1e-6
        Absolute step size floor for finite differences (``mode="fd"``).

    Returns
    -------
    G : jax.Array, shape (N, J, K)
        Chamberlain optimal instruments evaluated at ``theta``.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64).ravel()
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1-D after ravel; got shape {theta.shape}")

    P0 = prob_evaluator(theta)
    P0 = jnp.asarray(P0, dtype=jnp.float64)
    if P0.ndim != 2:
        raise ValueError(f"prob_evaluator must return a 2D array; got shape {P0.shape}")
    if P0.shape[1] < 2:
        raise ValueError(f"prob_evaluator must return at least two columns; got {P0.shape[1]}")

    if check_rowsum:
        row_sum = jnp.sum(P0, axis=1)
        if not jnp.allclose(row_sum, 1.0, atol=1e-10):
            min_rs = float(jnp.min(row_sum))
            max_rs = float(jnp.max(row_sum))
            raise ValueError(f"Row sums must be ≈ 1; observed range [{min_rs:.2e}, {max_rs:.2e}]")

    N = P0.shape[0]
    J = P0.shape[1] - 1
    K = int(theta.size)

    F = jax.jit(lambda th: _log_odds(th, prob_evaluator, eps))

    mode_sel = mode.lower()
    if mode_sel == "auto":
        mode_sel = "full" if K <= 256 else "chunked"

    if mode_sel == "full":
        JF = _jacobian_full(F, theta)
    elif mode_sel == "chunked":
        JF = _jacobian_chunked(F, theta, K, chunk_size=chunk_size)
    elif mode_sel == "fd":
        JF = _jacobian_finite_diff(F, theta, rel_step=fd_rel_step, abs_step=fd_abs_step)
    else:
        raise ValueError("mode must be 'auto', 'full', 'chunked', or 'fd'")

    if JF.shape[0] != N * J or JF.shape[1] != K:
        raise ValueError(
            f"Jacobian has unexpected shape {JF.shape}; expected ({N * J}, {K})"
        )

    G = JF.reshape((N, J, K))
    return G


def to_bcoo(G: Array, density_threshold: float = 0.15):
    """Convert dense Chamberlain instruments to ``BCOO`` if sufficiently sparse."""
    if not isinstance(G, jax.Array):
        G = jnp.asarray(G)

    nnz = jnp.count_nonzero(G)
    total = G.size
    density = float(nnz) / max(int(total), 1)
    if density <= density_threshold:
        return BCOO.fromdense(G)
    return G


__all__ = [
    "chamberlain_instruments_jax",
    "to_bcoo",
]
