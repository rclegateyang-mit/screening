#!/usr/bin/env python3
"""
JAX moment functions for Chamberlain-style GMM.

Functions
---------
- residuals(theta, X, Y, aux) -> R: (N,J) = Y[:,1:] - P(theta,X)[:,1:]
- moment(theta, X, Y, G_feat, aux) -> m: (p,) = tensordot over (N,J)
- criterion(theta, X, Y, G_feat, aux, W=None) -> scalar: m' m or m' W m

Notes
-----
- All computations are in JAX (jnp) and JIT-compatible.
- The probability kernel is imported from jax_model.compute_choice_probabilities_jax.
- Numerical stability is handled inside the probability kernel; here we operate
  on residuals and features only.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

try:
    # Prefer package-relative import when available
    from .jax_model import compute_choice_probabilities_jax
except Exception:  # pragma: no cover - fallback for direct module usage
    from jax_model import compute_choice_probabilities_jax


def residuals(theta: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, aux) -> jnp.ndarray:
    """Compute residuals R = Y[:,1:] − P(theta,X)[:,1:] with shape (N,J).

    Args:
      theta: (1+2J,) parameter vector packing (γ, V_nat(1..J), c_nat(1..J))
      X:     (N,) worker skill (or first column of features)
      Y:     (N,J+1) one-hot choices with outside at col 0
      aux:   dict pytree with precomputed arrays (see jax_model)
    """
    P = compute_choice_probabilities_jax(theta, X, aux)  # (N, J+1)
    R = Y[:, 1:] - P[:, 1:]
    return R


@jax.jit
def moment(theta: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, G_feat: jnp.ndarray, aux) -> jnp.ndarray:
    """Compute moment vector m (p,) as tensordot over (N,J).

    m = Σ_{n=1..N} Σ_{j=1..J} R[n,j] · G_feat[n,j,:]
    """
    R = residuals(theta, X, Y, aux)          # (N,J)
    m = jnp.tensordot(R, G_feat, axes=([0, 1], [0, 1]))  # (p,)
    return m


@jax.jit
def criterion(
    theta: jnp.ndarray,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    G_feat: jnp.ndarray,
    aux,
    W: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Quadratic GMM objective: m' m if W is None else m' W m.

    If W is provided, it must be (p,p) SPD; we symmetrize and rely on Cholesky
    to fail if not SPD.
    """
    m = moment(theta, X, Y, G_feat, aux)  # (p,)
    if W is None:
        return jnp.dot(m, m)
    Ws = 0.5 * (W + W.T)
    # Cholesky acts as an SPD assertion (raises if not SPD during execution)
    _ = jnp.linalg.cholesky(Ws)
    return m @ (Ws @ m)


__all__ = [
    "residuals",
    "moment",
    "criterion",
]
