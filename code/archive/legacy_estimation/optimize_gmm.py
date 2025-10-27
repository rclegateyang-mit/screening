#!/usr/bin/env python3
"""
Optimization wrappers using jaxopt solvers wired to the GMM moment residual.

Exports
-------
- solve_gn(theta0, X, Y, G_feat, aux, maxiter=500, tol=1e-6, lb=None, ub=None)
- solve_lm(theta0, X, Y, G_feat, aux, maxiter=500, tol=1e-6, lb=None, ub=None)

Both functions minimize ||m(theta)||^2 where m is the moment vector produced by
code.moments.moment. Optional box bounds are enforced via smooth reparameterization
using sigmoid/softplus so the solver runs on an unconstrained variable z.

Notes
-----
- Heavy functions are JIT-compiled on first call.
- This module assumes JAX and jaxopt are available in the environment.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.nn import sigmoid, softplus

from jaxopt import GaussNewton, LevenbergMarquardt

try:
    from .moments import moment
    from .jax_model import enable_x64
except Exception:  # pragma: no cover - fallback for direct module usage
    from moments import moment
    from jax_model import enable_x64


# ------------------------------
# Reparameterization utilities
# ------------------------------

def _inv_softplus(t: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable inverse of softplus for t>0: log(exp(t) - 1).
    Uses two branches for stability.
    """
    t = jnp.asarray(t)
    small = jnp.log(jnp.expm1(t))            # stable for small t
    large = t + jnp.log1p(-jnp.exp(-t))      # stable for large t
    return jnp.where(t < 20.0, small, large)


def _logit(y: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Stable logit: log(y) - log(1-y) with clipping."""
    y = jnp.clip(y, eps, 1.0 - eps)
    return jnp.log(y) - jnp.log1p(-y)


def make_reparam(lb: Optional[jnp.ndarray], ub: Optional[jnp.ndarray]):
    """Create forward/backward transforms between unconstrained z and bounded theta.

    Supports elementwise combinations of free, [lb, +inf), (-inf, ub], and [lb, ub].
    """
    if lb is None and ub is None:
        def fwd(z):
            return z
        def inv(theta):
            return theta
        return fwd, inv

    if lb is None:
        lb = -jnp.inf * jnp.ones_like(ub)
    if ub is None:
        ub = jnp.inf * jnp.ones_like(lb)
    lb = jnp.asarray(lb, dtype=jnp.float64)
    ub = jnp.asarray(ub, dtype=jnp.float64)

    lb_f = jnp.isfinite(lb)
    ub_f = jnp.isfinite(ub)
    both = lb_f & ub_f
    only_lb = lb_f & (~ub_f)
    only_ub = (~lb_f) & ub_f

    def fwd(z: jnp.ndarray) -> jnp.ndarray:
        # z -> theta
        z = jnp.asarray(z, dtype=jnp.float64)
        lb_both = jnp.where(both, lb, 0.0)
        span = jnp.where(both, ub - lb, 0.0)
        y_both = lb_both + span * sigmoid(z)

        lb_only = jnp.where(only_lb, lb, 0.0)
        y_lb = lb_only + softplus(z)

        ub_only = jnp.where(only_ub, ub, 0.0)
        y_ub = ub_only - softplus(z)

        theta = jnp.where(
            both,
            y_both,
            jnp.where(only_lb, y_lb, jnp.where(only_ub, y_ub, z)),
        )
        return theta

    def inv(theta: jnp.ndarray) -> jnp.ndarray:
        # theta -> z
        theta = jnp.asarray(theta, dtype=jnp.float64)
        span = jnp.where(both, ub - lb, 1.0)
        # Guard against division by zero and ensure argument stays in (0,1)
        ratio = jnp.where(
            both,
            (theta - lb) / jnp.maximum(span, 1e-300),
            0.5,
        )
        z_both = _logit(jnp.clip(ratio, 1e-12, 1 - 1e-12))

        diff_lb = jnp.where(only_lb, theta - lb, 1.0)
        z_lb = _inv_softplus(jnp.maximum(diff_lb, 1e-12))

        diff_ub = jnp.where(only_ub, ub - theta, 1.0)
        z_ub = _inv_softplus(jnp.maximum(diff_ub, 1e-12))

        z = jnp.where(
            both,
            z_both,
            jnp.where(only_lb, z_lb, jnp.where(only_ub, z_ub, theta)),
        )
        return z

    return fwd, inv


# ------------------------------
# Solver wrappers
# ------------------------------

def _build_residual_theta(X, Y, G_feat, aux, moment_fn=None):
    """JIT-compiled residual function r(theta)."""
    if moment_fn is None:
        @jax.jit
        def r_theta(theta: jnp.ndarray) -> jnp.ndarray:
            return moment(theta, X, Y, G_feat, aux)
        return r_theta

    return jax.jit(moment_fn)


def _build_residual_z(fwd, X, Y, G_feat, aux, moment_fn=None):
    """JIT-compiled residual function r(z) with theta=fwd(z)."""
    r_theta = _build_residual_theta(X, Y, G_feat, aux, moment_fn=moment_fn)

    @jax.jit
    def r_z(z: jnp.ndarray) -> jnp.ndarray:
        return r_theta(fwd(z))
    return r_z


def _build_criterion_theta(X, Y, G_feat, aux, moment_fn=None):
    r_theta = _build_residual_theta(X, Y, G_feat, aux, moment_fn=moment_fn)

    @jax.jit
    def f_theta(theta: jnp.ndarray) -> jnp.ndarray:
        r = r_theta(theta)
        return jnp.dot(r, r)
    return f_theta


def solve_gn(
    theta0: jnp.ndarray,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    G_feat,
    aux,
    *,
    maxiter: int = 500,
    tol: float = 1e-6,
    lb: Optional[jnp.ndarray] = None,
    ub: Optional[jnp.ndarray] = None,
    iterate_callback=None,
    moment_fn=None,
):
    """Gauss-Newton solve for min_theta ||moment(theta)||^2 with optional box bounds.

    Additional stability controls can be supplied through ``solver_kwargs``
    (forwarded to :class:`jaxopt.LevenbergMarquardt`). For example, ``damping``
    raises/lowers the initial λ in LM (larger ⇒ smaller, GN-like steps), while
    ``initial_trust_radius`` shrinks the trust region around each iterate.

    Returns dict: {theta_hat, obj, nit, grad_norm}
    """
    enable_x64()
    theta0 = jnp.asarray(theta0, dtype=jnp.float64)

    fwd, inv = make_reparam(lb, ub)
    z0 = inv(theta0)

    r_z = _build_residual_z(fwd, X, Y, G_feat, aux, moment_fn=moment_fn)
    solver = GaussNewton(residual_fun=r_z, maxiter=int(maxiter), tol=tol)

    f_theta = _build_criterion_theta(X, Y, G_feat, aux, moment_fn=moment_fn)
    grad_f = jax.jit(jax.grad(f_theta))

    # Run solver without callback first
    res = solver.run(z0)
    z_hat = res.params
    theta_hat = fwd(z_hat)

    obj = float(f_theta(theta_hat))
    g = grad_f(theta_hat)
    grad_norm = float(jnp.linalg.norm(g))

    nit = None
    try:
        nit = int(res.state.iter_num)
    except Exception:
        try:
            nit = int(res.state.iterations)
        except Exception:
            nit = None

    return {
        "theta_hat": jnp.asarray(theta_hat),
        "obj": obj,
        "nit": nit,
        "grad_norm": grad_norm,
    }


def solve_lm(
    theta0: jnp.ndarray,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    G_feat,
    aux,
    *,
    maxiter: int = 500,
    tol: float = 1e-6,
    lb: Optional[jnp.ndarray] = None,
    ub: Optional[jnp.ndarray] = None,
    iterate_callback=None,
    solver_kwargs: Optional[dict] = None,
    moment_fn=None,
):
    """Levenberg-Marquardt solve with residual_fun=moment and optional box bounds.

    Returns dict: {theta_hat, obj, nit, grad_norm}
    """
    enable_x64()
    theta0 = jnp.asarray(theta0, dtype=jnp.float64)

    fwd, inv = make_reparam(lb, ub)
    z0 = inv(theta0)

    r_z = _build_residual_z(fwd, X, Y, G_feat, aux, moment_fn=moment_fn)
    if solver_kwargs is None:
        solver_kwargs = {}
    solver = LevenbergMarquardt(
        residual_fun=r_z,
        maxiter=int(maxiter),
        tol=tol,
        **solver_kwargs,
    )

    f_theta = _build_criterion_theta(X, Y, G_feat, aux, moment_fn=moment_fn)
    grad_f = jax.jit(jax.grad(f_theta))

    # Run solver without callback first
    res = solver.run(z0)
    z_hat = res.params
    theta_hat = fwd(z_hat)

    obj = float(f_theta(theta_hat))
    g = grad_f(theta_hat)
    grad_norm = float(jnp.linalg.norm(g))

    nit = None
    try:
        nit = int(res.state.iter_num)
    except Exception:
        try:
            nit = int(res.state.iterations)
        except Exception:
            nit = None

    return {
        "theta_hat": jnp.asarray(theta_hat),
        "obj": obj,
        "nit": nit,
        "grad_norm": grad_norm,
    }


__all__ = [
    "solve_gn",
    "solve_lm",
]


# ------------------------------
# Newton-CG (Gauss–Newton HVP) solver
# ------------------------------

def _build_grad_and_value(r_z):
    """Return jitted (g(z), f(z), r(z)) for f(z)=||r(z)||^2 and g=∇f.

    Uses reverse-mode on r to compute g = 2 J^T r.
    """

    @jax.jit
    def gv(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r, pullback = jax.vjp(r_z, z)
        g = 2.0 * pullback(r)[0]
        f = jnp.dot(r, r)
        return g, f, r

    return gv


def _build_hvp(r_z):
    """Return jitted hvp(z, v) ≈ 2 J(z)^T J(z) v (Gauss–Newton)."""

    @jax.jit
    def hvp(z: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # JVP to form J v
        _, Jv = jax.jvp(r_z, (z,), (v,))
        # VJP to form J^T (J v)
        _, pullback = jax.vjp(r_z, z)
        return 2.0 * pullback(Jv)[0]

    return hvp


def _tau_to_boundary(p: jnp.ndarray, d: jnp.ndarray, delta: float) -> float:
    """Compute step length tau to hit trust-region boundary along direction d.

    Solve ||p + tau d||^2 = delta^2 for tau > 0.
    """
    a = float(jnp.dot(d, d))
    b = float(jnp.dot(p, d))
    c = float(jnp.dot(p, p) - (delta * delta))
    # tau = (-b + sqrt(b^2 - a*c)) / a  (positive root)
    disc = max(0.0, b * b - a * c)
    tau = (-b + disc ** 0.5) / max(a, 1e-30)
    return float(max(tau, 0.0))


def _truncated_cg(hvp, g: jnp.ndarray, delta: float, maxiter: int = 50, tol: float = 1e-6):
    """Truncated CG to approximately solve H p = -g with trust-region radius delta.

    Returns (p, boundary_hit, iters).
    """
    p = jnp.zeros_like(g)
    r = -g.copy()
    d = r.copy()
    rr = float(jnp.dot(r, r))
    cg_tol = max(tol, min(0.5, rr ** 0.5) * rr ** 0.5)
    boundary_hit = False

    for k in range(int(maxiter)):
        Hd = hvp(p * 0.0 + 0.0 * d + 0.0 * g + 0.0, d)  # ensure same dtype; hvp(z,v) will ignore z here if implemented differently
        dHd = float(jnp.dot(d, Hd))
        if dHd <= 0.0:
            tau = _tau_to_boundary(p, d, delta)
            p = p + tau * d
            boundary_hit = True
            return p, boundary_hit, k + 1

        alpha = rr / max(dHd, 1e-30)
        p_next = p + alpha * d
        if float(jnp.linalg.norm(p_next)) >= delta:
            tau = _tau_to_boundary(p, d, delta)
            p = p + tau * d
            boundary_hit = True
            return p, boundary_hit, k + 1

        r = r - alpha * Hd
        rr_new = float(jnp.dot(r, r))
        if rr_new ** 0.5 <= cg_tol:
            p = p_next
            return p, boundary_hit, k + 1
        beta = rr_new / max(rr, 1e-30)
        d = r + beta * d
        p = p_next
        rr = rr_new

    return p, boundary_hit, int(maxiter)


def solve_newton_cg(
    theta0: jnp.ndarray,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    G_feat: jnp.ndarray,
    aux,
    *,
    maxiter: int = 200,
    tol: float = 1e-6,
    lb: Optional[jnp.ndarray] = None,
    ub: Optional[jnp.ndarray] = None,
    cg_maxiter: int = 50,
    trust_radius: float = 1.0,
):
    """Newton-CG with Gauss–Newton HVPs, trust region, and backtracking.

    Returns dict: {theta_hat, obj, nit, grad_norm, n_cg}
    """
    enable_x64()
    theta0 = jnp.asarray(theta0, dtype=jnp.float64)

    fwd, inv = make_reparam(lb, ub)
    z = inv(theta0)

    # Residual and value/grad/hvp builders in z-space
    r_theta = _build_residual_theta(X, Y, G_feat, aux)

    @jax.jit
    def r_z(z_):
        return r_theta(fwd(z_))

    gv = _build_grad_and_value(r_z)
    hvp_at = _build_hvp(r_z)

    delta = float(trust_radius)
    n_cg_total = 0

    # Initial metrics
    g, f, _ = gv(z)
    grad_norm = float(jnp.linalg.norm(g))

    nit = 0
    for k in range(int(maxiter)):
        nit = k + 1
        if grad_norm <= tol:
            break

        # Build HVP function at current z
        def Hv(v):
            return hvp_at(z, v)

        # Truncated CG step
        p, boundary_hit, it_cg = _truncated_cg(Hv, g, delta, maxiter=cg_maxiter, tol=tol)
        n_cg_total += it_cg

        # Predicted reduction
        Hp = Hv(p)
        m_pred = - float(jnp.dot(g, p)) - 0.5 * float(jnp.dot(p, Hp))
        if m_pred <= 0:
            # fallback small step
            p = - (1.0 / max(float(jnp.dot(g, g)), 1e-12)) * g
            Hp = Hv(p)
            m_pred = - float(jnp.dot(g, p)) - 0.5 * float(jnp.dot(p, Hp))

        # Evaluate new point
        z_new = z - p  # minimize f -> step opposite gradient direction sign from CG solution of H p = g; we solved with RHS g, so subtract p
        g_new, f_new, _ = gv(z_new)

        # Ratio of actual to predicted reduction
        ared = f - float(f_new)
        pred = m_pred
        rho = ared / max(pred, 1e-30)

        # Trust-region update rules
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and boundary_hit:
            delta = min(2.0 * delta, 1e6)

        # Accept or reject
        if rho > 1e-4 and f_new <= f:
            z = z_new
            g = g_new
            f = float(f_new)
            grad_norm = float(jnp.linalg.norm(g))
        else:
            # reject, try smaller region next iter
            pass

    theta_hat = fwd(z)
    return {
        "theta_hat": jnp.asarray(theta_hat),
        "obj": float(f),
        "nit": nit,
        "grad_norm": grad_norm,
        "n_cg": n_cg_total,
    }


__all__.extend(["solve_newton_cg"])
