#!/usr/bin/env python3
"""MPI orchestration for distributed augmented-Lagrangian hybrid MLE+GMM.

All ranks call ``run_auglag_distributed`` (SPMD pattern).  Rank 0 drives
the outer augmented-Lagrangian loop; non-root ranks sit in a worker loop
responding to broadcast commands.

Communication per outer iteration:

    Step 2 (Market block):
        Bcast  19 floats (theta_G + nu + rho + mu + g_bar)
        Reduce  6 floats (nll + g_bar_contrib + n_converged)

    Step 4 (Global solve, repeated per L-BFGS eval):
        Bcast   6 floats (theta_G_trial)
        Reduce 35 floats (nll + g_bar + grad_nll + jac_g_bar)

At 35 floats (280 bytes) per global eval reduce, communication is
deeply latency-bound.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from mpi4py import MPI

from screening.analysis.auglag.worker import (
    HybridMarketData,
    HybridMarketState,
    forward_with_jac_eval,
    solve_market_subproblem,
)

# ---------------------------------------------------------------------------
# MPI command protocol
# ---------------------------------------------------------------------------

CMD_MARKET_BLOCK = 1   # Step 2: solve market subproblems
CMD_GLOBAL_EVAL  = 2   # Step 4: forward+jac callback for global solve
CMD_DONE         = 0   # exit worker loop


# ---------------------------------------------------------------------------
# Market block collective (Step 2)
# ---------------------------------------------------------------------------


def _mpi_market_block(
    comm: MPI.Comm,
    local_markets: List[HybridMarketData],
    local_states: List[HybridMarketState],
    inner_maxiter: int,
    inner_tol: float,
) -> Tuple[List[HybridMarketState], float, np.ndarray, int]:
    """All ranks call together for the market block step.

    1. Bcast 19-float buffer: [theta_G(6), nu(4), rho(1), mu(4), g_bar(4)]
    2. Each rank solves its markets
    3. Reduce 6-float buffer: [local_nll(1), local_g_bar_contrib(4), n_converged(1)]

    Returns (updated_local_states, total_nll, g_bar_mid, n_converged) on root.
    Non-root returns are partial (total_nll/g_bar_mid/n_converged are meaningless).
    """
    # Step 1: Bcast parameters
    bcast_buf = np.zeros(19, dtype=np.float64)
    comm.Bcast(bcast_buf, root=0)

    theta_G = bcast_buf[:6]
    nu = bcast_buf[6:10]
    rho = float(bcast_buf[10])
    mu = bcast_buf[11:15]
    g_bar = bcast_buf[15:19]

    # Step 2: Each rank solves its markets
    local_nll = 0.0
    local_g_bar_contrib = np.zeros(4, dtype=np.float64)
    local_n_converged = 0

    for i, (md, state) in enumerate(zip(local_markets, local_states)):
        # c_m = mu - g_bar + omega_m * g_m (computed locally from broadcast)
        c_m = mu - g_bar + md.omega * state.g_m

        new_state, conv = solve_market_subproblem(
            md, state, theta_G, nu, rho, c_m, inner_maxiter, inner_tol,
        )
        local_states[i] = new_state
        local_nll += new_state.nll
        local_g_bar_contrib += md.omega * new_state.g_m
        local_n_converged += int(conv)

    # Step 3: Reduce
    reduce_buf = np.array([local_nll] + local_g_bar_contrib.tolist()
                          + [float(local_n_converged)], dtype=np.float64)
    total_buf = np.zeros(6, dtype=np.float64)
    comm.Reduce(np.ascontiguousarray(reduce_buf), total_buf,
                op=MPI.SUM, root=0)

    total_nll = total_buf[0]
    g_bar_mid = total_buf[1:5]
    n_converged = int(total_buf[5])

    return local_states, total_nll, g_bar_mid, n_converged


# ---------------------------------------------------------------------------
# Global eval collective (Step 4)
# ---------------------------------------------------------------------------


def _mpi_global_eval(
    comm: MPI.Comm,
    local_markets: List[HybridMarketData],
    local_states: List[HybridMarketState],
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """All ranks call together for a global eval round-trip.

    1. Bcast theta_G_trial(6)
    2. Each rank evaluates forward_with_jac on its markets
    3. Reduce 35-float buffer: [nll(1), g_bar(4), grad_nll(6), jac_g_bar(24)]
    4. Side effect: updates local_states[i].g_m and .nll

    Returns (nll, g_bar(4,), grad_nll(6,), jac_g_bar(4,6)) on root.
    """
    theta_G_buf = np.zeros(6, dtype=np.float64)
    comm.Bcast(theta_G_buf, root=0)

    # Local computation
    local_nll = 0.0
    local_g_bar = np.zeros(4, dtype=np.float64)
    local_grad_nll = np.zeros(6, dtype=np.float64)
    local_jac_g_bar = np.zeros((4, 6), dtype=np.float64)

    for i, (md, state) in enumerate(zip(local_markets, local_states)):
        nll_m, g_m, grad_nll_m, jac_g_m = forward_with_jac_eval(
            md, state, theta_G_buf,
        )
        local_nll += nll_m
        local_g_bar += md.omega * g_m
        local_grad_nll += grad_nll_m
        local_jac_g_bar += md.omega * jac_g_m

        # Side effect: update cached state (safe because last eval
        # before dual update is always at the accepted theta_G)
        local_states[i].g_m = g_m
        local_states[i].nll = nll_m

    # Pack into contiguous 35-float buffer
    reduce_buf = np.concatenate([
        np.array([local_nll]),
        local_g_bar,
        local_grad_nll,
        local_jac_g_bar.ravel(),
    ]).astype(np.float64)

    total_buf = np.zeros(35, dtype=np.float64)
    comm.Reduce(np.ascontiguousarray(reduce_buf), total_buf,
                op=MPI.SUM, root=0)

    nll = total_buf[0]
    g_bar = total_buf[1:5]
    grad_nll = total_buf[5:11]
    jac_g_bar = total_buf[11:35].reshape(4, 6)

    return nll, g_bar, grad_nll, jac_g_bar


# ---------------------------------------------------------------------------
# Worker loop (non-root ranks)
# ---------------------------------------------------------------------------


def _worker_loop(
    comm: MPI.Comm,
    local_markets: List[HybridMarketData],
    local_states: List[HybridMarketState],
    inner_maxiter: int,
    inner_tol: float,
) -> List[HybridMarketState]:
    """Non-root ranks wait for commands and participate in collectives.

    Returns the final local_states.
    """
    cmd_buf = np.array(0, dtype=np.int32)

    while True:
        comm.Bcast(cmd_buf, root=0)
        cmd = int(cmd_buf)

        if cmd == CMD_DONE:
            break
        elif cmd == CMD_MARKET_BLOCK:
            local_states, _, _, _ = _mpi_market_block(
                comm, local_markets, local_states, inner_maxiter, inner_tol,
            )
        elif cmd == CMD_GLOBAL_EVAL:
            _mpi_global_eval(comm, local_markets, local_states)

    return local_states


# ---------------------------------------------------------------------------
# Global solve callback factory (Step 4)
# ---------------------------------------------------------------------------


def _build_global_callbacks(
    comm: MPI.Comm,
    local_markets: List[HybridMarketData],
    local_states: List[HybridMarketState],
    W: np.ndarray,
    nu: np.ndarray,
    rho: float,
    verbose: bool,
) -> Tuple[Callable, Callable, dict]:
    """Build fun/jac callbacks for scipy global solve.

    Returns (fun, jac, eval_state).
    """
    W_plus_rhoI = W + rho * np.eye(4)
    W_plus_rhoI_inv = np.linalg.inv(W_plus_rhoI)

    eval_state = {
        "last_theta_G": None,
        "last_phi": None,
        "last_grad_phi": None,
        "n_evals": 0,
    }

    def _do_eval(theta_G_flat):
        """Broadcast CMD_GLOBAL_EVAL and compute Phi + grad_Phi."""
        theta_G_np = np.asarray(theta_G_flat, dtype=np.float64)

        # Tell workers to participate
        cmd_buf = np.array(CMD_GLOBAL_EVAL, dtype=np.int32)
        comm.Bcast(cmd_buf, root=0)

        # Set the broadcast buffer for _mpi_global_eval
        theta_G_buf = theta_G_np.copy()
        # Need to put it into the Bcast that _mpi_global_eval will issue
        # Actually, _mpi_global_eval does its own Bcast, so we need to
        # ensure root's buffer has the right values. We call the function
        # directly — root participates in the collective too.

        # Overwrite theta_G_buf before the Bcast inside _mpi_global_eval
        # by providing root's view. But _mpi_global_eval allocates its own
        # buffer. We need a different approach: set the values before calling.

        # Actually the cleanest approach: _mpi_global_eval does Bcast from
        # a local buffer. Root needs to fill that buffer first. Let's inline
        # the root side here instead.

        # ---- Inline global eval (root side) ----
        # Bcast theta_G_trial
        comm.Bcast(theta_G_np, root=0)

        # Local computation on root's markets
        local_nll = 0.0
        local_g_bar = np.zeros(4, dtype=np.float64)
        local_grad_nll = np.zeros(6, dtype=np.float64)
        local_jac_g_bar = np.zeros((4, 6), dtype=np.float64)

        for i, (md, state) in enumerate(zip(local_markets, local_states)):
            nll_m, g_m, grad_nll_m, jac_g_m = forward_with_jac_eval(
                md, state, theta_G_np,
            )
            local_nll += nll_m
            local_g_bar += md.omega * g_m
            local_grad_nll += grad_nll_m
            local_jac_g_bar += md.omega * jac_g_m
            local_states[i].g_m = g_m
            local_states[i].nll = nll_m

        reduce_buf = np.concatenate([
            np.array([local_nll]),
            local_g_bar,
            local_grad_nll,
            local_jac_g_bar.ravel(),
        ]).astype(np.float64)

        total_buf = np.zeros(35, dtype=np.float64)
        comm.Reduce(np.ascontiguousarray(reduce_buf), total_buf,
                     op=MPI.SUM, root=0)

        total_nll = total_buf[0]
        g_bar = total_buf[1:5]
        grad_nll_total = total_buf[5:11]
        jac_g_bar = total_buf[11:35].reshape(4, 6)

        # Concentrate out mu*
        mu_star = W_plus_rhoI_inv @ (rho * g_bar - nu)

        # Phi value
        residual = mu_star - g_bar
        phi = (total_nll
               + 0.5 * mu_star @ W @ mu_star
               + nu @ residual
               + (rho / 2.0) * np.dot(residual, residual))

        # Gradient via envelope theorem
        lambda_eff = nu + rho * residual
        grad_phi = grad_nll_total - jac_g_bar.T @ lambda_eff

        eval_state["n_evals"] += 1
        eval_state["last_theta_G"] = theta_G_np.copy()
        eval_state["last_phi"] = float(phi)
        eval_state["last_grad_phi"] = grad_phi.astype(np.float64)

    def _ensure_eval(theta_G_flat):
        theta_G_np = np.asarray(theta_G_flat, dtype=np.float64)
        if (eval_state["last_theta_G"] is None
                or not np.array_equal(theta_G_np, eval_state["last_theta_G"])):
            _do_eval(theta_G_np)

    def fun(theta_G_flat):
        _ensure_eval(theta_G_flat)
        return eval_state["last_phi"]

    def jac(theta_G_flat):
        _ensure_eval(theta_G_flat)
        return eval_state["last_grad_phi"]

    return fun, jac, eval_state


# ---------------------------------------------------------------------------
# Outer augmented-Lagrangian loop (SPMD — all ranks call)
# ---------------------------------------------------------------------------


def run_auglag_distributed(
    comm: MPI.Comm,
    local_markets: List[HybridMarketData],
    local_states: List[HybridMarketState],
    theta_G_init: np.ndarray,
    W: np.ndarray,
    *,
    max_outer_iter: int = 200,
    inner_maxiter: int = 200,
    inner_tol: float = 1e-6,
    global_maxiter: int = 100,
    global_tol: float = 1e-5,
    verbose: bool = True,
) -> dict:
    """SPMD outer augmented-Lagrangian loop.

    All ranks must call this together.  Root drives the 6-step loop;
    workers respond to commands transparently.

    Returns a result dict (meaningful on root only).
    """
    from scipy.optimize import minimize as sp_minimize

    rank = comm.Get_rank()
    d_g = 4

    # Total market count
    M_local = len(local_markets)
    M_total = comm.reduce(M_local, op=MPI.SUM, root=0)
    M_total = comm.bcast(M_total, root=0)

    if rank != 0:
        # Workers: enter command loop
        local_states = _worker_loop(
            comm, local_markets, local_states, inner_maxiter, inner_tol,
        )
        return {
            "rank": rank,
            "local_states": local_states,
        }

    # ======= Root drives the outer loop =======

    theta_G = theta_G_init.copy()

    # Initial forward eval to get g_ms and g_bar
    cmd_buf = np.array(CMD_GLOBAL_EVAL, dtype=np.int32)
    comm.Bcast(cmd_buf, root=0)

    # Inline root-side global eval for initialization
    theta_G_np = theta_G.copy()
    comm.Bcast(theta_G_np, root=0)

    local_nll_init = 0.0
    local_g_bar_init = np.zeros(4, dtype=np.float64)
    local_grad_init = np.zeros(6, dtype=np.float64)
    local_jac_init = np.zeros((4, 6), dtype=np.float64)

    for i, (md, state) in enumerate(zip(local_markets, local_states)):
        nll_m, g_m, grad_m, jac_m = forward_with_jac_eval(md, state, theta_G_np)
        local_nll_init += nll_m
        local_g_bar_init += md.omega * g_m
        local_grad_init += grad_m
        local_jac_init += md.omega * jac_m
        local_states[i].g_m = g_m
        local_states[i].nll = nll_m

    reduce_buf = np.concatenate([
        np.array([local_nll_init]),
        local_g_bar_init,
        local_grad_init,
        local_jac_init.ravel(),
    ]).astype(np.float64)
    total_buf = np.zeros(35, dtype=np.float64)
    comm.Reduce(np.ascontiguousarray(reduce_buf), total_buf,
                op=MPI.SUM, root=0)

    total_nll = total_buf[0]
    g_bar = total_buf[1:5]

    # rho_0 = tr(W) / d_g
    rho = float(np.trace(W)) / d_g
    nu = np.zeros(d_g, dtype=np.float64)
    W_plus_rhoI_inv = np.linalg.inv(W + rho * np.eye(d_g))
    mu = W_plus_rhoI_inv @ (rho * g_bar)

    history = []
    converged = False

    if verbose:
        obj = total_nll + 0.5 * g_bar @ W @ g_bar
        print(f"AugLag init: nll={total_nll:.4f}  |g_bar|={np.linalg.norm(g_bar):.4e}  "
              f"obj={obj:.4f}  rho={rho:.4f}")
        print(f"  theta_G = {theta_G}")

    for k in range(max_outer_iter):
        iter_start = time.perf_counter()
        theta_G_old = theta_G.copy()
        g_bar_old = g_bar.copy()

        # --- Step 1+2: Market block ---
        cmd_buf = np.array(CMD_MARKET_BLOCK, dtype=np.int32)
        comm.Bcast(cmd_buf, root=0)

        # Pack broadcast buffer: [theta_G(6), nu(4), rho(1), mu(4), g_bar(4)]
        bcast_buf = np.concatenate([
            theta_G, nu, np.array([rho]), mu, g_bar,
        ]).astype(np.float64)
        comm.Bcast(bcast_buf, root=0)

        # Root solves its own markets
        local_nll_root = 0.0
        local_g_bar_root = np.zeros(4, dtype=np.float64)
        local_n_converged_root = 0

        for i, (md, state) in enumerate(zip(local_markets, local_states)):
            c_m = mu - g_bar + md.omega * state.g_m
            new_state, conv = solve_market_subproblem(
                md, state, theta_G, nu, rho, c_m, inner_maxiter, inner_tol,
            )
            local_states[i] = new_state
            local_nll_root += new_state.nll
            local_g_bar_root += md.omega * new_state.g_m
            local_n_converged_root += int(conv)

        # Reduce
        reduce_buf = np.array(
            [local_nll_root] + local_g_bar_root.tolist()
            + [float(local_n_converged_root)],
            dtype=np.float64,
        )
        total_buf_6 = np.zeros(6, dtype=np.float64)
        comm.Reduce(np.ascontiguousarray(reduce_buf), total_buf_6,
                     op=MPI.SUM, root=0)

        total_nll_mid = total_buf_6[0]
        g_bar_mid = total_buf_6[1:5]
        n_converged = int(total_buf_6[5])

        # --- Step 3: Intermediate mu update ---
        W_plus_rhoI_inv = np.linalg.inv(W + rho * np.eye(d_g))
        mu_mid = W_plus_rhoI_inv @ (rho * g_bar_mid - nu)

        # --- Step 4: Global block ---
        fun, jac_fn, eval_st = _build_global_callbacks(
            comm, local_markets, local_states, W, nu, rho, verbose,
        )

        bounds = [
            (1e-6, 1.0 - 1e-6),   # tau
            (1e-6, None),           # tilde_gamma
            (1e-6, 1.0 - 1e-6),   # alpha
            (1e-6, None),           # sigma_e
            (1e-6, None),           # eta
            (None, None),           # gamma0
        ]

        result = sp_minimize(
            fun, theta_G, method="L-BFGS-B", jac=jac_fn,
            bounds=bounds,
            options={"maxiter": global_maxiter, "gtol": global_tol,
                     "ftol": 1e-15},
        )
        theta_G = result.x.astype(np.float64)

        # --- Step 5: Final eval at accepted theta_G to update g_m caches ---
        cmd_buf = np.array(CMD_GLOBAL_EVAL, dtype=np.int32)
        comm.Bcast(cmd_buf, root=0)

        # Root-side inline global eval
        comm.Bcast(theta_G, root=0)

        local_nll_final = 0.0
        local_g_bar_final = np.zeros(4, dtype=np.float64)
        local_grad_final = np.zeros(6, dtype=np.float64)
        local_jac_final = np.zeros((4, 6), dtype=np.float64)

        for i, (md, state) in enumerate(zip(local_markets, local_states)):
            nll_m, g_m, grad_m, jac_m = forward_with_jac_eval(
                md, state, theta_G,
            )
            local_nll_final += nll_m
            local_g_bar_final += md.omega * g_m
            local_grad_final += grad_m
            local_jac_final += md.omega * jac_m
            local_states[i].g_m = g_m
            local_states[i].nll = nll_m

        reduce_buf_35 = np.concatenate([
            np.array([local_nll_final]),
            local_g_bar_final,
            local_grad_final,
            local_jac_final.ravel(),
        ]).astype(np.float64)
        total_buf_35 = np.zeros(35, dtype=np.float64)
        comm.Reduce(np.ascontiguousarray(reduce_buf_35), total_buf_35,
                     op=MPI.SUM, root=0)

        total_nll = total_buf_35[0]
        g_bar = total_buf_35[1:5]

        # --- Step 6: mu, nu update and convergence check ---
        mu = W_plus_rhoI_inv @ (rho * g_bar - nu)
        nu_new = nu + rho * (mu - g_bar)

        primal_res = np.linalg.norm(mu - g_bar)
        dual_res = rho * np.linalg.norm(g_bar - g_bar_old)
        param_change = (np.linalg.norm(theta_G - theta_G_old)
                        / max(1.0, np.linalg.norm(theta_G)))

        rel_primal = primal_res / max(1.0, np.linalg.norm(mu))
        rel_dual = dual_res / max(1.0, np.linalg.norm(nu_new))

        obj = total_nll + 0.5 * g_bar @ W @ g_bar

        iter_time = time.perf_counter() - iter_start
        entry = {
            'iter': int(k), 'obj': float(obj), 'nll': float(total_nll),
            'primal_res': float(primal_res), 'dual_res': float(dual_res),
            'rel_primal': float(rel_primal), 'rel_dual': float(rel_dual),
            'param_change': float(param_change), 'rho': float(rho),
            'theta_G': theta_G.tolist(), 'g_bar': g_bar.tolist(),
            'n_converged': int(n_converged), 'wall_sec': float(iter_time),
            'global_evals': int(eval_st["n_evals"]),
        }
        history.append(entry)

        if verbose:
            print(f"  [iter {k:3d}] obj={obj:.4f}  nll={total_nll:.4f}  "
                  f"|g_bar|={np.linalg.norm(g_bar):.4e}  "
                  f"primal={rel_primal:.3e}  dual={rel_dual:.3e}  "
                  f"dparam={param_change:.3e}  rho={rho:.2f}  "
                  f"inner_conv={n_converged}/{M_total}  "
                  f"g_evals={eval_st['n_evals']}  {iter_time:.1f}s")

        nu = nu_new

        converged = (rel_primal < 1e-4 and rel_dual < 1e-4
                     and param_change < 1e-5)
        if converged:
            if verbose:
                print(f"  Converged at iteration {k}.")
            break

        # --- rho update (residual balancing) ---
        if primal_res > 10.0 * dual_res:
            rho = min(2.0 * rho, 1e4)
        elif dual_res > 10.0 * primal_res:
            rho = max(rho / 2.0, 1e-4)

    # Tell workers to stop
    cmd_buf = np.array(CMD_DONE, dtype=np.int32)
    comm.Bcast(cmd_buf, root=0)

    return {
        'theta_G': theta_G,
        'local_states': local_states,
        'mu': mu,
        'nu': nu,
        'g_bar': g_bar,
        'total_nll': total_nll,
        'obj': total_nll + 0.5 * g_bar @ W @ g_bar,
        'converged': converged,
        'n_outer_iters': k + 1 if max_outer_iter > 0 else 0,
        'history': history,
        'rho': rho,
    }
