#!/usr/bin/env python3
"""Outer optimizer for distributed MLE using MPI collectives + scipy.optimize.

All ranks call ``run_outer_loop`` (SPMD pattern).  Rank 0 drives the
optimization via ``scipy.optimize.minimize``; non-root ranks sit in a
worker loop responding to broadcast commands.

Communication per objective evaluation:

    1. ``Bcast``  command + θ_G   (root → all)
    2. Local inner solves          (no communication)
    3. ``Reduce`` nll, grad, H     (all → root)

Total data moved per evaluation: ~240 bytes × log₂(P) via tree collectives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import nn as jnn
from mpi4py import MPI

from code.estimation.distributed_worker import MarketData, MarketResult, solve_market

# ---------------------------------------------------------------------------
# MPI command protocol
# ---------------------------------------------------------------------------

CMD_EVAL = 1       # evaluate objective + gradient (+ optionally Hessian)
CMD_DONE = 0       # workers exit loop

# ---------------------------------------------------------------------------
# Transform for the 5 global parameters θ_G = [τ, α, γ, σ_e, λ_e]
# ---------------------------------------------------------------------------

EPS_TAU = 1e-6
EPS_ALPHA = 1e-6
POS_FLOOR = 1e-6


@dataclass(frozen=True)
class GlobalTransform:
    """Bijection between R^5 (unconstrained z_G) and bounded θ_G."""

    @staticmethod
    def fwd(z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=jnp.float64)
        tau = EPS_TAU + (1.0 - 2.0 * EPS_TAU) * jnn.sigmoid(z[0])
        alpha = EPS_ALPHA + (1.0 - 2.0 * EPS_ALPHA) * jnn.sigmoid(z[1])
        gamma = POS_FLOOR + jnn.softplus(z[2])
        sigma_e = POS_FLOOR + jnn.softplus(z[3])
        lambda_e = z[4]
        return jnp.array([tau, alpha, gamma, sigma_e, lambda_e], dtype=jnp.float64)

    @staticmethod
    def inv(theta_G: jnp.ndarray) -> jnp.ndarray:
        theta_G = jnp.asarray(theta_G, dtype=jnp.float64)
        tau, alpha, gamma, sigma_e, lambda_e = (
            theta_G[0], theta_G[1], theta_G[2], theta_G[3], theta_G[4],
        )
        tau_scaled = (tau - EPS_TAU) / (1.0 - 2.0 * EPS_TAU)
        alpha_scaled = (alpha - EPS_ALPHA) / (1.0 - 2.0 * EPS_ALPHA)

        def _logit(y):
            y = jnp.clip(y, 1e-12, 1.0 - 1e-12)
            return jnp.log(y) - jnp.log1p(-y)

        def _softplus_inv(y):
            y = jnp.maximum(y, 1e-12)
            return jnp.where(y < 20.0, jnp.log(jnp.expm1(y)),
                             y + jnp.log1p(-jnp.exp(-y)))

        tau_z = _logit(jnp.clip(tau_scaled, 1e-12, 1.0 - 1e-12))
        alpha_z = _logit(jnp.clip(alpha_scaled, 1e-12, 1.0 - 1e-12))
        gamma_z = _softplus_inv(jnp.maximum(gamma - POS_FLOOR, 1e-12))
        sigma_e_z = _softplus_inv(jnp.maximum(sigma_e - POS_FLOOR, 1e-12))
        return jnp.array([tau_z, alpha_z, gamma_z, sigma_e_z, lambda_e],
                         dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Per-iteration diagnostics
# ---------------------------------------------------------------------------


@dataclass
class OuterIterResult:
    iteration: int
    nll: float
    grad_norm_theta: float
    grad_norm_z: float
    step_size: float
    n_converged_inner: int
    n_markets: int
    wall_sec: float


# ---------------------------------------------------------------------------
# Local solve helpers
# ---------------------------------------------------------------------------


def _solve_local_markets(
    local_markets: List[MarketData],
    theta_G: np.ndarray,
    local_theta_m_inits: List[np.ndarray],
    inner_maxiter: int,
    inner_tol: float,
    compute_hessian: bool,
) -> List[MarketResult]:
    """Solve all markets assigned to this rank."""
    return [
        solve_market(md, theta_G, tm, inner_maxiter, inner_tol, compute_hessian)
        for md, tm in zip(local_markets, local_theta_m_inits)
    ]


def _local_aggregate(results: List[MarketResult]):
    """Sum NLL, gradient, Schur across this rank's markets."""
    local_nll = sum(r.nll_m for r in results)
    local_grad = sum(r.grad_G_m for r in results)
    local_schur = sum(r.schur_m for r in results)
    local_conv = sum(1 for r in results if r.inner_converged)
    return local_nll, local_grad, local_schur, local_conv


# ---------------------------------------------------------------------------
# MPI reduction helpers
# ---------------------------------------------------------------------------


def _reduce_scalars(comm, local_nll: float, local_conv: int):
    """Reduce NLL and convergence count to root."""
    total_nll = comm.reduce(local_nll, op=MPI.SUM, root=0)
    total_conv = comm.reduce(local_conv, op=MPI.SUM, root=0)
    return total_nll, total_conv


def _reduce_arrays(comm, local_grad: np.ndarray, local_schur: np.ndarray):
    """Reduce gradient (5,) and Schur (5,5) to root via buffer protocol."""
    total_grad = np.zeros(5, dtype=np.float64)
    total_schur = np.zeros((5, 5), dtype=np.float64)
    comm.Reduce(np.ascontiguousarray(local_grad, dtype=np.float64),
                total_grad, op=MPI.SUM, root=0)
    comm.Reduce(np.ascontiguousarray(local_schur, dtype=np.float64),
                total_schur, op=MPI.SUM, root=0)
    return total_grad, total_schur


# ---------------------------------------------------------------------------
# Shared MPI evaluation (all ranks call together)
# ---------------------------------------------------------------------------


def _mpi_eval(comm, local_markets, local_tm, theta_G_np,
              inner_maxiter, inner_tol, compute_hessian):
    """Broadcast θ_G, solve local markets, reduce results.

    All ranks must call this together.  Returns full aggregates on root,
    partial on workers.
    """
    # Broadcast theta_G from root
    comm.Bcast(theta_G_np, root=0)

    # Each rank solves its local markets
    local_results = _solve_local_markets(
        local_markets, theta_G_np, local_tm,
        inner_maxiter, inner_tol, compute_hessian,
    )
    local_nll, local_grad, local_schur, local_conv = _local_aggregate(local_results)

    # Reduce to root
    total_nll, n_conv = _reduce_scalars(comm, local_nll, local_conv)
    total_grad, total_schur = _reduce_arrays(comm, local_grad, local_schur)

    return local_results, total_nll, total_grad, total_schur, n_conv


# ---------------------------------------------------------------------------
# Worker loop (non-root ranks)
# ---------------------------------------------------------------------------


def _worker_loop(comm, local_markets, local_tm, inner_maxiter, inner_tol):
    """Non-root ranks wait for commands from root and participate in evals.

    Returns the final local_results for gathering.
    """
    local_results = None
    theta_G_buf = np.zeros(5, dtype=np.float64)
    cmd_buf = np.array(0, dtype=np.int32)

    while True:
        comm.Bcast(cmd_buf, root=0)
        cmd = int(cmd_buf)

        if cmd == CMD_DONE:
            break

        compute_hessian = bool(cmd_buf)  # CMD_EVAL=1 placeholder; hessian flag sent separately
        hess_flag = np.array(False, dtype=np.bool_)
        comm.Bcast(hess_flag, root=0)
        compute_hessian = bool(hess_flag)

        local_results_new, _, _, _, _ = _mpi_eval(
            comm, local_markets, local_tm, theta_G_buf,
            inner_maxiter, inner_tol, compute_hessian,
        )
        # Update warm-starts
        local_tm = [r.theta_m_hat for r in local_results_new]
        local_results = local_results_new

    return local_results, local_tm


# ---------------------------------------------------------------------------
# Root scipy callback machinery
# ---------------------------------------------------------------------------


def _build_root_callbacks(
    comm, local_markets, local_tm_container, transform,
    inner_maxiter, inner_tol, compute_hessian,
    frozen_mask, z_G_frozen, M_total, verbose,
):
    """Build fun/jac/hess callbacks for scipy.optimize.minimize.

    ``local_tm_container`` is a mutable list [local_tm] so the closure can
    update warm-starts across scipy iterations.

    Returns (fun, jac, hess_or_None, callback, eval_state).
    """
    # Mutable state shared across callbacks
    eval_state = {
        "n_evals": 0,
        "last_z": None,
        "last_nll": None,
        "last_grad_z": None,
        "last_hess_z": None,
        "last_n_conv": 0,
        "last_grad_norm_theta": 0.0,
        "history": [],
        "iter_start": time.perf_counter(),
    }

    def _do_eval(z_G_np):
        """Run one MPI evaluation round.  Only called if z_G changed."""
        # Tell workers to participate
        cmd_buf = np.array(CMD_EVAL, dtype=np.int32)
        comm.Bcast(cmd_buf, root=0)
        hess_flag = np.array(compute_hessian, dtype=np.bool_)
        comm.Bcast(hess_flag, root=0)

        # Transform z → θ
        theta_G = np.asarray(
            transform.fwd(jnp.asarray(z_G_np, dtype=jnp.float64)),
            dtype=np.float64,
        )

        local_results, total_nll, total_grad, total_schur, n_conv = _mpi_eval(
            comm, local_markets, local_tm_container[0], theta_G,
            inner_maxiter, inner_tol, compute_hessian,
        )

        # Update warm-starts
        local_tm_container[0] = [r.theta_m_hat for r in local_results]

        # Chain rule: transform gradient/Hessian to unconstrained space
        z_G_jax = jnp.asarray(z_G_np, dtype=jnp.float64)
        J_fwd = np.asarray(
            jax.jacobian(transform.fwd)(z_G_jax), dtype=np.float64)

        grad_z = J_fwd.T @ total_grad
        grad_norm_theta = float(np.linalg.norm(total_grad))

        # Apply frozen mask
        if frozen_mask is not None:
            grad_z[frozen_mask] = 0.0

        hess_z = None
        if compute_hessian:
            hess_z = J_fwd.T @ total_schur @ J_fwd
            if frozen_mask is not None:
                hess_z[frozen_mask, :] = 0.0
                hess_z[:, frozen_mask] = 0.0
                for idx in np.where(frozen_mask)[0]:
                    hess_z[idx, idx] = 1.0

        # Cache
        eval_state["n_evals"] += 1
        eval_state["last_z"] = z_G_np.copy()
        eval_state["last_nll"] = float(total_nll)
        eval_state["last_grad_z"] = grad_z.astype(np.float64)
        eval_state["last_hess_z"] = hess_z
        eval_state["last_n_conv"] = n_conv
        eval_state["last_grad_norm_theta"] = grad_norm_theta

    def _ensure_eval(z_G_flat):
        """Evaluate if z_G changed since last eval."""
        z_G_np = np.asarray(z_G_flat, dtype=np.float64)
        # Enforce frozen values
        if frozen_mask is not None:
            z_G_np[frozen_mask] = z_G_frozen[frozen_mask]
        if eval_state["last_z"] is None or not np.array_equal(z_G_np, eval_state["last_z"]):
            _do_eval(z_G_np)

    def fun(z_G_flat):
        _ensure_eval(z_G_flat)
        return eval_state["last_nll"]

    def jac(z_G_flat):
        _ensure_eval(z_G_flat)
        return eval_state["last_grad_z"]

    def hess(z_G_flat):
        _ensure_eval(z_G_flat)
        return eval_state["last_hess_z"]

    iter_count = [0]

    def callback(xk, *args):
        """Called by scipy after each iteration."""
        now = time.perf_counter()
        wall = now - eval_state["iter_start"]
        grad_norm_z = float(np.linalg.norm(eval_state["last_grad_z"])) if eval_state["last_grad_z"] is not None else 0.0

        entry = OuterIterResult(
            iteration=iter_count[0],
            nll=eval_state["last_nll"] or 0.0,
            grad_norm_theta=eval_state["last_grad_norm_theta"],
            grad_norm_z=grad_norm_z,
            step_size=0.0,  # scipy manages step size internally
            n_converged_inner=eval_state["last_n_conv"],
            n_markets=M_total,
            wall_sec=wall,
        )
        eval_state["history"].append(entry)

        if verbose:
            print(f"  [outer {iter_count[0]:3d}] nll={entry.nll:.4f}  "
                  f"|grad_z|={entry.grad_norm_z:.3e}  "
                  f"|grad_θ|={entry.grad_norm_theta:.3e}  "
                  f"inner_conv={entry.n_converged_inner}/{M_total}  "
                  f"wall={entry.wall_sec:.1f}s")

        iter_count[0] += 1
        eval_state["iter_start"] = time.perf_counter()

    hess_fn = hess if compute_hessian else None
    return fun, jac, hess_fn, callback, eval_state


# ---------------------------------------------------------------------------
# Outer loop (SPMD — all ranks call this)
# ---------------------------------------------------------------------------


def run_outer_loop(
    comm: MPI.Comm,
    local_markets: List[MarketData],
    local_theta_m_inits: List[np.ndarray],
    theta_G_init: np.ndarray,
    *,
    max_outer_iter: int = 50,
    outer_tol: float = 1e-5,
    inner_maxiter: int = 200,
    inner_tol: float = 1e-6,
    compute_hessian: bool = True,
    frozen_indices: Optional[List[int]] = None,
    method: str = "L-BFGS-B",
    verbose: bool = True,
) -> dict:
    """SPMD outer optimization loop.  All ranks must call this together.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator (typically COMM_WORLD).
    local_markets : list of MarketData
        This rank's partition of markets (received via scatter).
    local_theta_m_inits : list of (2*J,) arrays
        Initial local parameters for this rank's markets.
    theta_G_init : (5,) array
        Initial global parameters (only meaningful on rank 0, broadcast to all).
    compute_hessian : bool
        True → supply Hessian to scipy (use with 'trust-ncg' or 'Newton-CG').
        False → gradient-only (use with 'L-BFGS-B').
    method : str
        scipy.optimize.minimize method. Recommended: 'L-BFGS-B' (no Hessian)
        or 'trust-ncg' (with Hessian).

    Returns
    -------
    dict (meaningful on rank 0 only; other ranks return partial info).
    """
    from scipy.optimize import minimize

    rank = comm.Get_rank()
    size = comm.Get_size()
    M_total = comm.reduce(len(local_markets), op=MPI.SUM, root=0)
    M_total = comm.bcast(M_total, root=0)

    transform = GlobalTransform()

    # Initialise z_G on root, broadcast to all
    z_G = np.zeros(5, dtype=np.float64)
    if rank == 0:
        z_G[:] = np.asarray(
            transform.inv(jnp.asarray(theta_G_init)), dtype=np.float64)
    comm.Bcast(z_G, root=0)

    local_tm = [tm.copy() for tm in local_theta_m_inits]

    # Frozen parameter mask
    frozen_mask = None
    if frozen_indices:
        frozen_mask = np.zeros(5, dtype=bool)
        for idx in frozen_indices:
            frozen_mask[idx] = True
    z_G_frozen = z_G.copy()

    if rank == 0:
        # ---- Root: drive scipy.optimize.minimize ----
        local_tm_container = [local_tm]  # mutable wrapper for closure

        fun, jac_fn, hess_fn, callback, eval_state = _build_root_callbacks(
            comm, local_markets, local_tm_container, transform,
            inner_maxiter, inner_tol, compute_hessian,
            frozen_mask, z_G_frozen, M_total, verbose,
        )

        # Build scipy options
        scipy_opts = {
            "maxiter": max_outer_iter,
            "disp": False,
        }
        if method == "L-BFGS-B":
            scipy_opts["gtol"] = outer_tol

            # Use bounds to freeze parameters in L-BFGS-B
            bounds = [(None, None)] * 5
            if frozen_mask is not None:
                for i in range(5):
                    if frozen_mask[i]:
                        bounds[i] = (float(z_G_frozen[i]), float(z_G_frozen[i]))

            scipy_result = minimize(
                fun, z_G.copy(), jac=jac_fn,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options=scipy_opts,
            )
        elif method in ("trust-ncg", "Newton-CG"):
            scipy_opts["gtol"] = outer_tol
            scipy_result = minimize(
                fun, z_G.copy(), jac=jac_fn, hess=hess_fn,
                method=method,
                callback=callback,
                options=scipy_opts,
            )
        else:
            raise ValueError(f"Unsupported outer method: {method}")

        # Extract results
        z_G_hat = scipy_result.x
        if frozen_mask is not None:
            z_G_hat[frozen_mask] = z_G_frozen[frozen_mask]

        history = eval_state["history"]
        total_nll = scipy_result.fun
        grad_z = scipy_result.jac if hasattr(scipy_result, 'jac') and scipy_result.jac is not None else eval_state["last_grad_z"]
        final_grad_norm = float(np.linalg.norm(grad_z)) if grad_z is not None else 0.0
        converged = scipy_result.success

        # ---- Final inner solve at converged θ_G ----
        # Re-solve inner problems at the exact final θ_G so that the
        # gathered δ, q̄ are consistent with the converged outer solution.
        # (During optimization, warm-starts may lag behind the final point.)
        if verbose:
            print("  Running final inner solve at converged θ_G...")
        cmd_buf = np.array(CMD_EVAL, dtype=np.int32)
        comm.Bcast(cmd_buf, root=0)
        hess_flag = np.array(False, dtype=np.bool_)  # no Hessian needed
        comm.Bcast(hess_flag, root=0)

        theta_G_final = np.asarray(
            transform.fwd(jnp.asarray(z_G_hat, dtype=jnp.float64)),
            dtype=np.float64,
        )
        final_results, final_nll, _, _, final_n_conv = _mpi_eval(
            comm, local_markets, local_tm_container[0], theta_G_final,
            inner_maxiter, inner_tol, False,
        )
        local_tm = [r.theta_m_hat for r in final_results]
        if verbose:
            print(f"  Final inner solve: nll={final_nll:.4f}  "
                  f"inner_conv={final_n_conv}/{M_total}")

        # Tell workers to stop
        cmd_buf = np.array(CMD_DONE, dtype=np.int32)
        comm.Bcast(cmd_buf, root=0)

    else:
        # ---- Non-root: worker loop ----
        # Worker loop handles both optimization evals and the final
        # inner solve — it just responds to CMD_EVAL/CMD_DONE commands.
        _, local_tm = _worker_loop(
            comm, local_markets, local_tm, inner_maxiter, inner_tol)

        # Placeholders for non-root
        z_G_hat = np.zeros(5, dtype=np.float64)
        history = []
        total_nll = 0.0
        final_grad_norm = 0.0
        converged = False

    # ---- Gather final θ_m from all ranks ----
    all_local_results_pairs = comm.gather(
        [(md.market_id, tm) for md, tm in zip(local_markets, local_tm)],
        root=0,
    )

    if rank == 0:
        # Flatten and sort by market_id
        all_pairs = [pair for chunk in all_local_results_pairs for pair in chunk]
        all_pairs.sort(key=lambda p: p[0])
        theta_m_hats = [p[1] for p in all_pairs]

        theta_G_hat = np.asarray(
            transform.fwd(jnp.asarray(z_G_hat)), dtype=np.float64)

        return {
            "theta_G_hat": theta_G_hat,
            "theta_m_hats": theta_m_hats,
            "history": history,
            "total_nll": total_nll,
            "converged": converged,
            "n_outer_iters": len(history),
            "final_grad_norm_z": (history[-1].grad_norm_z
                                  if history else final_grad_norm),
            "scipy_result": {
                "success": scipy_result.success,
                "message": scipy_result.message,
                "nit": scipy_result.nit,
                "nfev": scipy_result.nfev,
            },
        }
    else:
        return {"rank": rank, "local_theta_m_hats": local_tm}
