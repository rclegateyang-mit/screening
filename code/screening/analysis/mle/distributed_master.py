#!/usr/bin/env python3
"""Outer optimizer for distributed MLE using MPI collectives + scipy.optimize.

All ranks call ``run_outer_loop`` (SPMD pattern).  Rank 0 drives the
optimization via ``scipy.optimize.minimize``; non-root ranks sit in a
worker loop responding to broadcast commands.

Global parameters: θ_G = [τ, γ̃] (tau and tilde_gamma).

Communication per objective evaluation:

    1. ``Bcast``  command + θ_G   (root → all)
    2. Local inner solves          (no communication)
    3. ``Reduce`` nll, grad, H     (all → root)

Total data moved per evaluation: ~80 bytes × log₂(P) via tree collectives.
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

from screening.analysis.mle.distributed_worker import MarketData, MarketResult, solve_market

# ---------------------------------------------------------------------------
# MPI command protocol
# ---------------------------------------------------------------------------

CMD_EVAL = 1       # evaluate objective + gradient (+ optionally Hessian)
CMD_DONE = 0       # workers exit loop
CMD_ACCEPT = 2     # step accepted — promote trial inner state to base

# ---------------------------------------------------------------------------
# Transform for the 2 global parameters θ_G = [τ, γ̃]
# ---------------------------------------------------------------------------

N_GLOBAL = 2
EPS_TAU = 1e-6
POS_FLOOR = 1e-6


@dataclass(frozen=True)
class GlobalTransform:
    """Bijection between R^2 (unconstrained z_G) and bounded θ_G."""

    @staticmethod
    def fwd(z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=jnp.float64)
        tau = EPS_TAU + (1.0 - 2.0 * EPS_TAU) * jnn.sigmoid(z[0])
        tilde_gamma = POS_FLOOR + jnn.softplus(z[1])
        return jnp.array([tau, tilde_gamma], dtype=jnp.float64)

    @staticmethod
    def inv(theta_G: jnp.ndarray) -> jnp.ndarray:
        theta_G = jnp.asarray(theta_G, dtype=jnp.float64)
        tau, tilde_gamma = theta_G[0], theta_G[1]
        tau_scaled = (tau - EPS_TAU) / (1.0 - 2.0 * EPS_TAU)

        def _logit(y):
            y = jnp.clip(y, 1e-12, 1.0 - 1e-12)
            return jnp.log(y) - jnp.log1p(-y)

        def _softplus_inv(y):
            y = jnp.maximum(y, 1e-12)
            return jnp.where(y < 20.0, jnp.log(jnp.expm1(y)),
                             y + jnp.log1p(-jnp.exp(-y)))

        tau_z = _logit(jnp.clip(tau_scaled, 1e-12, 1.0 - 1e-12))
        tg_z = _softplus_inv(jnp.maximum(tilde_gamma - POS_FLOOR, 1e-12))
        return jnp.array([tau_z, tg_z], dtype=jnp.float64)


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
    local_tilde_q_fixed: Optional[List[np.ndarray]] = None,
    profile_delta: bool = False,
    warm_start: bool = False,
) -> List[MarketResult]:
    """Solve all markets assigned to this rank."""
    if profile_delta:
        return [
            solve_market(md, theta_G, tm, inner_maxiter, inner_tol,
                         compute_hessian, profile_delta=True,
                         warm_start=warm_start)
            for md, tm in zip(local_markets, local_theta_m_inits)
        ]
    if local_tilde_q_fixed is None:
        return [
            solve_market(md, theta_G, tm, inner_maxiter, inner_tol, compute_hessian,
                         warm_start=warm_start)
            for md, tm in zip(local_markets, local_theta_m_inits)
        ]
    return [
        solve_market(md, theta_G, tm, inner_maxiter, inner_tol, compute_hessian,
                     tilde_q_fixed=tq, warm_start=warm_start)
        for md, tm, tq in zip(local_markets, local_theta_m_inits, local_tilde_q_fixed)
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
    """Reduce gradient (2,) and Schur (2,2) to root via buffer protocol."""
    total_grad = np.zeros(N_GLOBAL, dtype=np.float64)
    total_schur = np.zeros((N_GLOBAL, N_GLOBAL), dtype=np.float64)
    comm.Reduce(np.ascontiguousarray(local_grad, dtype=np.float64),
                total_grad, op=MPI.SUM, root=0)
    comm.Reduce(np.ascontiguousarray(local_schur, dtype=np.float64),
                total_schur, op=MPI.SUM, root=0)
    return total_grad, total_schur


# ---------------------------------------------------------------------------
# Shared MPI evaluation (all ranks call together)
# ---------------------------------------------------------------------------


def _mpi_eval(comm, local_markets, local_tm, theta_G_np,
              inner_maxiter, inner_tol, compute_hessian,
              warm_start=False):
    """Broadcast θ_G, solve local markets, reduce results."""
    comm.Bcast(theta_G_np, root=0)

    # Broadcast warm_start flag so all ranks agree
    warm_buf = np.array(warm_start, dtype=np.bool_)
    comm.Bcast(warm_buf, root=0)

    local_results = _solve_local_markets(
        local_markets, theta_G_np, local_tm,
        inner_maxiter, inner_tol, compute_hessian,
        warm_start=bool(warm_buf),
    )
    local_nll, local_grad, local_schur, local_conv = _local_aggregate(local_results)

    total_nll, n_conv = _reduce_scalars(comm, local_nll, local_conv)
    total_grad, total_schur = _reduce_arrays(comm, local_grad, local_schur)

    return local_results, total_nll, total_grad, total_schur, n_conv


# ---------------------------------------------------------------------------
# Worker loop (non-root ranks)
# ---------------------------------------------------------------------------


def _worker_loop(comm, local_markets, local_tm, inner_maxiter, inner_tol):
    """Non-root ranks wait for commands and participate in evals.

    Uses base_tm / trial_tm pattern to stay consistent with root's
    warm-start strategy during line search.
    """
    base_tm = [tm.copy() for tm in local_tm]
    trial_tm = None
    theta_G_buf = np.zeros(N_GLOBAL, dtype=np.float64)
    cmd_buf = np.array(0, dtype=np.int32)

    while True:
        comm.Bcast(cmd_buf, root=0)
        cmd = int(cmd_buf)

        if cmd == CMD_DONE:
            break

        if cmd == CMD_ACCEPT:
            # Promote trial state → base state
            if trial_tm is not None:
                base_tm = trial_tm
                trial_tm = None
            continue

        # CMD_EVAL
        hess_flag = np.array(False, dtype=np.bool_)
        comm.Bcast(hess_flag, root=0)
        compute_hessian = bool(hess_flag)

        local_results_new, _, _, _, _ = _mpi_eval(
            comm, local_markets, base_tm, theta_G_buf,
            inner_maxiter, inner_tol, compute_hessian,
        )
        trial_tm = [r.theta_m_hat for r in local_results_new]

    final_tm = trial_tm if trial_tm is not None else base_tm
    return None, final_tm


# ---------------------------------------------------------------------------
# Root scipy callback machinery
# ---------------------------------------------------------------------------


def _build_root_callbacks(
    comm, local_markets, local_tm_container, transform,
    inner_maxiter, inner_tol, compute_hessian,
    M_total, verbose,
):
    """Build fun/jac/hess callbacks for scipy.optimize.minimize.

    Inner-solve warm-start strategy: ``base_tm`` holds the accepted inner
    state.  During the line search all evaluations start from ``base_tm``
    so the profiled objective is smooth.  ``base_tm`` is updated ONLY in the
    callback (after L-BFGS-B accepts a step).
    """
    # base_tm: accepted inner state — updated only at callback
    base_tm = [tm.copy() for tm in local_tm_container[0]]
    # trial_tm: inner state from the most recent eval (may be a line-search trial)
    trial_tm_container = [None]

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
        "warm_start": False,
    }

    def _do_eval(z_G_np):
        cmd_buf = np.array(CMD_EVAL, dtype=np.int32)
        comm.Bcast(cmd_buf, root=0)
        hess_flag = np.array(compute_hessian, dtype=np.bool_)
        comm.Bcast(hess_flag, root=0)

        theta_G = np.asarray(
            transform.fwd(jnp.asarray(z_G_np, dtype=jnp.float64)),
            dtype=np.float64,
        )

        # Always start inner solve from base_tm (accepted state)
        local_results, total_nll, total_grad, total_schur, n_conv = _mpi_eval(
            comm, local_markets, base_tm, theta_G,
            inner_maxiter, inner_tol, compute_hessian,
            warm_start=eval_state["warm_start"],
        )

        # Store trial results (only promoted to base_tm at callback)
        trial_tm_container[0] = [r.theta_m_hat for r in local_results]

        z_G_jax = jnp.asarray(z_G_np, dtype=jnp.float64)
        J_fwd = np.asarray(
            jax.jacobian(transform.fwd)(z_G_jax), dtype=np.float64)

        grad_z = J_fwd.T @ total_grad
        grad_norm_theta = float(np.linalg.norm(total_grad))

        hess_z = None
        if compute_hessian:
            hess_z = J_fwd.T @ total_schur @ J_fwd

        eval_state["n_evals"] += 1
        eval_state["last_z"] = z_G_np.copy()
        eval_state["last_nll"] = float(total_nll)
        eval_state["last_grad_z"] = grad_z.astype(np.float64)
        eval_state["last_hess_z"] = hess_z
        eval_state["last_n_conv"] = n_conv
        eval_state["last_grad_norm_theta"] = grad_norm_theta

    def _ensure_eval(z_G_flat):
        z_G_np = np.asarray(z_G_flat, dtype=np.float64)
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
        now = time.perf_counter()
        wall = now - eval_state["iter_start"]
        grad_norm_z = float(np.linalg.norm(eval_state["last_grad_z"])) if eval_state["last_grad_z"] is not None else 0.0

        entry = OuterIterResult(
            iteration=iter_count[0],
            nll=eval_state["last_nll"] or 0.0,
            grad_norm_theta=eval_state["last_grad_norm_theta"],
            grad_norm_z=grad_norm_z,
            step_size=0.0,
            n_converged_inner=eval_state["last_n_conv"],
            n_markets=M_total,
            wall_sec=wall,
        )
        eval_state["history"].append(entry)

        # Step accepted → tell workers to promote, then promote root's state
        eval_state["warm_start"] = True
        accept_buf = np.array(CMD_ACCEPT, dtype=np.int32)
        comm.Bcast(accept_buf, root=0)
        if trial_tm_container[0] is not None:
            base_tm[:] = trial_tm_container[0]
            local_tm_container[0] = [tm.copy() for tm in base_tm]

        if verbose:
            theta_G_cur = np.asarray(
                transform.fwd(jnp.asarray(xk, dtype=jnp.float64)),
                dtype=np.float64)
            print(f"  [outer {iter_count[0]:3d}] nll={entry.nll:.4f}  "
                  f"|grad_z|={entry.grad_norm_z:.3e}  "
                  f"|grad_θ|={entry.grad_norm_theta:.3e}  "
                  f"tau={theta_G_cur[0]:.4f}  tg={theta_G_cur[1]:.4f}  "
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
    method: str = "L-BFGS-B",
    verbose: bool = True,
) -> dict:
    """SPMD outer optimization loop.  All ranks must call this together.

    Parameters
    ----------
    theta_G_init : (2,) array
        Initial global parameters [tau, tilde_gamma].
    """
    from scipy.optimize import minimize

    rank = comm.Get_rank()
    size = comm.Get_size()
    M_total = comm.reduce(len(local_markets), op=MPI.SUM, root=0)
    M_total = comm.bcast(M_total, root=0)

    transform = GlobalTransform()

    # Initialise z_G on root, broadcast to all
    z_G = np.zeros(N_GLOBAL, dtype=np.float64)
    if rank == 0:
        z_G[:] = np.asarray(
            transform.inv(jnp.asarray(theta_G_init)), dtype=np.float64)
    comm.Bcast(z_G, root=0)

    local_tm = [tm.copy() for tm in local_theta_m_inits]

    if rank == 0:
        # ---- Root: drive scipy.optimize.minimize ----
        local_tm_container = [local_tm]

        fun, jac_fn, hess_fn, callback, eval_state = _build_root_callbacks(
            comm, local_markets, local_tm_container, transform,
            inner_maxiter, inner_tol, compute_hessian,
            M_total, verbose,
        )

        scipy_opts = {
            "maxiter": max_outer_iter,
            "disp": verbose,
        }
        if method == "L-BFGS-B":
            scipy_opts["gtol"] = outer_tol
            scipy_opts["ftol"] = 1e-15  # prevent premature convergence
            bounds = [(None, None)] * N_GLOBAL

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

        z_G_hat = scipy_result.x

        history = eval_state["history"]
        total_nll = scipy_result.fun
        grad_z = scipy_result.jac if hasattr(scipy_result, 'jac') and scipy_result.jac is not None else eval_state["last_grad_z"]
        final_grad_norm = float(np.linalg.norm(grad_z)) if grad_z is not None else 0.0
        converged = scipy_result.success

        # ---- Final inner solve at converged θ_G ----
        if verbose:
            print("  Running final inner solve at converged θ_G...")
        cmd_buf = np.array(CMD_EVAL, dtype=np.int32)
        comm.Bcast(cmd_buf, root=0)
        hess_flag = np.array(False, dtype=np.bool_)
        comm.Bcast(hess_flag, root=0)

        theta_G_final = np.asarray(
            transform.fwd(jnp.asarray(z_G_hat, dtype=jnp.float64)),
            dtype=np.float64,
        )
        final_results, final_nll, _, _, final_n_conv = _mpi_eval(
            comm, local_markets, local_tm_container[0], theta_G_final,
            inner_maxiter, inner_tol, False,
            warm_start=True,
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
        _, local_tm = _worker_loop(
            comm, local_markets, local_tm, inner_maxiter, inner_tol)

        z_G_hat = np.zeros(N_GLOBAL, dtype=np.float64)
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


# ---------------------------------------------------------------------------
# Inner-only mode (freeze globals)
# ---------------------------------------------------------------------------


def run_inner_only(
    comm: MPI.Comm,
    local_markets: List[MarketData],
    local_theta_m_inits: List[np.ndarray],
    theta_G_fixed: np.ndarray,
    *,
    inner_maxiter: int = 1000,
    inner_tol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """Run inner solves only at a fixed θ_G (no outer optimization).

    All ranks must call this together (SPMD).
    """
    rank = comm.Get_rank()
    M_total = comm.reduce(len(local_markets), op=MPI.SUM, root=0)
    M_total = comm.bcast(M_total, root=0)

    theta_G_np = np.asarray(theta_G_fixed, dtype=np.float64)
    comm.Bcast(theta_G_np, root=0)

    local_tm = [tm.copy() for tm in local_theta_m_inits]

    if verbose and rank == 0:
        print(f"  Inner-only mode: theta_G fixed at {theta_G_np}")

    local_results = _solve_local_markets(
        local_markets, theta_G_np, local_tm,
        inner_maxiter, inner_tol, compute_hessian=False,
    )
    local_nll, local_grad, local_schur, local_conv = _local_aggregate(local_results)
    total_nll, n_conv = _reduce_scalars(comm, local_nll, local_conv)
    local_tm = [r.theta_m_hat for r in local_results]

    if verbose and rank == 0:
        print(f"  Inner-only solve: nll={total_nll:.4f}  "
              f"inner_conv={n_conv}/{M_total}")

    # Gather final theta_m from all ranks
    all_local_results_pairs = comm.gather(
        [(md.market_id, tm) for md, tm in zip(local_markets, local_tm)],
        root=0,
    )

    if rank == 0:
        all_pairs = [pair for chunk in all_local_results_pairs for pair in chunk]
        all_pairs.sort(key=lambda p: p[0])
        theta_m_hats = [p[1] for p in all_pairs]

        return {
            "theta_G_hat": theta_G_np,
            "theta_m_hats": theta_m_hats,
            "history": [],
            "total_nll": total_nll,
            "converged": True,
            "n_outer_iters": 0,
            "final_grad_norm_z": 0.0,
            "scipy_result": {},
        }
    else:
        return {"rank": rank, "local_theta_m_hats": local_tm}


# ---------------------------------------------------------------------------
# Delta-only mode (freeze globals + tilde_q)
# ---------------------------------------------------------------------------


def run_inner_delta_only(
    comm: MPI.Comm,
    local_markets: List[MarketData],
    local_theta_m_inits: List[np.ndarray],
    local_tilde_q_fixed: List[np.ndarray],
    theta_G_fixed: np.ndarray,
    *,
    inner_maxiter: int = 1000,
    inner_tol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """Run inner solves for delta only (tilde_q and theta_G both fixed).

    All ranks must call this together (SPMD).
    """
    rank = comm.Get_rank()
    M_total = comm.reduce(len(local_markets), op=MPI.SUM, root=0)
    M_total = comm.bcast(M_total, root=0)

    theta_G_np = np.asarray(theta_G_fixed, dtype=np.float64)
    comm.Bcast(theta_G_np, root=0)

    local_tm = [tm.copy() for tm in local_theta_m_inits]

    if verbose and rank == 0:
        print(f"  Delta-only mode: theta_G={theta_G_np}, tilde_q fixed at true values")

    local_results = _solve_local_markets(
        local_markets, theta_G_np, local_tm,
        inner_maxiter, inner_tol, compute_hessian=False,
        local_tilde_q_fixed=local_tilde_q_fixed,
    )
    local_nll, local_grad, local_schur, local_conv = _local_aggregate(local_results)
    total_nll, n_conv = _reduce_scalars(comm, local_nll, local_conv)
    local_tm = [r.theta_m_hat for r in local_results]

    if verbose and rank == 0:
        print(f"  Delta-only solve: nll={total_nll:.4f}  "
              f"inner_conv={n_conv}/{M_total}")

    all_local_results_pairs = comm.gather(
        [(md.market_id, tm) for md, tm in zip(local_markets, local_tm)],
        root=0,
    )

    if rank == 0:
        all_pairs = [pair for chunk in all_local_results_pairs for pair in chunk]
        all_pairs.sort(key=lambda p: p[0])
        theta_m_hats = [p[1] for p in all_pairs]

        return {
            "theta_G_hat": theta_G_np,
            "theta_m_hats": theta_m_hats,
            "history": [],
            "total_nll": total_nll,
            "converged": True,
            "n_outer_iters": 0,
            "final_grad_norm_z": 0.0,
            "scipy_result": {},
        }
    else:
        return {"rank": rank, "local_theta_m_hats": local_tm}


# ---------------------------------------------------------------------------
# Profile-delta mode (freeze globals, contraction for delta, optimize tilde_q)
# ---------------------------------------------------------------------------


def run_inner_profile_delta(
    comm: MPI.Comm,
    local_markets: List[MarketData],
    local_theta_m_inits: List[np.ndarray],
    theta_G_fixed: np.ndarray,
    *,
    inner_maxiter: int = 1000,
    inner_tol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """Run inner solves profiling out delta via BLP contraction.

    Optimizes over tilde_q only; delta is solved via contraction mapping
    at each step. theta_G is fixed. All ranks must call together (SPMD).
    """
    rank = comm.Get_rank()
    M_total = comm.reduce(len(local_markets), op=MPI.SUM, root=0)
    M_total = comm.bcast(M_total, root=0)

    theta_G_np = np.asarray(theta_G_fixed, dtype=np.float64)
    comm.Bcast(theta_G_np, root=0)

    local_tm = [tm.copy() for tm in local_theta_m_inits]

    if verbose and rank == 0:
        print(f"  Profile-delta mode: theta_G={theta_G_np}")
        print(f"  Optimizing tilde_q only; delta profiled via BLP contraction")

    local_results = _solve_local_markets(
        local_markets, theta_G_np, local_tm,
        inner_maxiter, inner_tol, compute_hessian=False,
        profile_delta=True,
    )
    local_nll, local_grad, local_schur, local_conv = _local_aggregate(local_results)
    total_nll, n_conv = _reduce_scalars(comm, local_nll, local_conv)
    local_tm = [r.theta_m_hat for r in local_results]

    if verbose and rank == 0:
        print(f"  Profile-delta solve: nll={total_nll:.4f}  "
              f"inner_conv={n_conv}/{M_total}")

    all_local_results_pairs = comm.gather(
        [(md.market_id, tm) for md, tm in zip(local_markets, local_tm)],
        root=0,
    )

    if rank == 0:
        all_pairs = [pair for chunk in all_local_results_pairs for pair in chunk]
        all_pairs.sort(key=lambda p: p[0])
        theta_m_hats = [p[1] for p in all_pairs]

        return {
            "theta_G_hat": theta_G_np,
            "theta_m_hats": theta_m_hats,
            "history": [],
            "total_nll": total_nll,
            "converged": True,
            "n_outer_iters": 0,
            "final_grad_norm_z": 0.0,
            "scipy_result": {},
        }
    else:
        return {"rank": rank, "local_theta_m_hats": local_tm}
