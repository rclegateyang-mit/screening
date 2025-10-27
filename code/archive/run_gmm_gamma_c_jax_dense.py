#!/usr/bin/env python3
"""Archived dense-instrument version of run_gmm_gamma_c_jax.py."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path as _Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxopt import LBFGS, LevenbergMarquardt

try:
    from ..jax_model import enable_x64, compute_choice_probabilities_jax
    from ..helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
        naive_theta_guess_gamma_V_c,
        compute_gmm_standard_errors,
    )
    from ..optimize_gmm import make_reparam
    from ..g_features import chamberlain_instruments_jax
except ImportError:  # pragma: no cover - fallback
    from jax_model import enable_x64, compute_choice_probabilities_jax
    from helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
        naive_theta_guess_gamma_V_c,
        compute_gmm_standard_errors,
    )
    from optimize_gmm import make_reparam
    from g_features import chamberlain_instruments_jax

os.environ.setdefault("JAX_ENABLE_X64", "1")


def parse_args() -> argparse.Namespace:
    root = _Path(__file__).resolve().parents[1]
    output_dir = root / "output"

    p = argparse.ArgumentParser(description="Run profiled JAX GMM for (gamma, c)")
    p.add_argument('--workers_path', type=str, default=str(output_dir / 'workers_dataset.csv'))
    p.add_argument('--firms_path', type=str, default=str(output_dir / 'equilibrium_firms.csv'))
    p.add_argument('--params_path', type=str, default=str(output_dir / 'parameters_effective.csv'))
    p.add_argument('--out_dir', type=str, default=str(output_dir))
    p.add_argument('--theta0_file', type=str, default=None)
    p.add_argument('--theta0_list', type=str, default=None,
                   help='Comma-separated γ, c(1..J); if omitted use baseline from data.')
    p.add_argument('--theta0_from_helper', action='store_true',
                   help='Initialise using helpers.naive_theta_guess_gamma_V_c.')
    p.add_argument('--maxiter', type=int, default=500)
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--share_solver', type=str, choices=['lm', 'blp'], default='lm')
    p.add_argument('--share_tol', type=float, default=1e-8,
                   help='Tolerance for the share-matching solver (residual sup-norm).')
    p.add_argument('--share_maxiter', type=int, default=500,
                   help='Maximum iterations for the share-matching solver.')
    p.add_argument('--share_debug', action='store_true',
                   help='Print diagnostics from the share-matching solver (useful for NaN debugging).')
    p.add_argument('--jac_chunk', type=int, default=32,
                   help='Chunk size when forming instruments (smaller ⇒ lower memory).')
    p.add_argument('--skip_statistics', action='store_true',
                   help='Skip robust SE / covariance computation to save memory.')
    p.add_argument('--threads', type=int, default=None)
    return p.parse_args()


def parse_theta0_gamma_c(
    args: argparse.Namespace,
    J: int,
    gamma0: float,
    c0: np.ndarray,
) -> np.ndarray:
    k_expected = 1 + J

    if args.theta0_list is not None:
        raw = args.theta0_list.replace('
', ' ').replace('	', ' ')
        values: list[float] = []
        for part in raw.split(','):
            values.extend([float(x.strip()) for x in part.split() if x.strip()])
        theta0 = np.asarray(values, dtype=float)
        if theta0.size != k_expected:
            raise ValueError(f"θ0 length {theta0.size} != 1+J={k_expected}.")
        return theta0

    if args.theta0_file is not None:
        file_path = args.theta0_file
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            theta0 = np.asarray(data['theta'], dtype=float)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            theta0 = df.iloc[0].values.astype(float)
        else:
            raise ValueError(f"Unsupported theta0 file format: {file_path}")
        if theta0.size != k_expected:
            raise ValueError(f"θ0 length {theta0.size} != 1+J={k_expected}.")
        return theta0

    return np.concatenate(([gamma0], c0))


def build_choice_matrix(chosen_firm: np.ndarray, J: int) -> np.ndarray:
    Y = np.zeros((chosen_firm.size, J + 1))
    Y[np.arange(chosen_firm.size), chosen_firm.astype(int)] = 1.0
    return Y


def main() -> None:
    args = parse_args()

    if args.threads is not None:
        print("Set CPU threads by exporting before Python:")
        print(f'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_cpu_thread_pool_size={int(args.threads)}"')

    enable_x64()

    params = read_parameters(args.params_path)
    firm_ids, w, _, _, xi, loc_firms, c_data = read_firms_data(args.firms_path)
    x_skill, ell_x, ell_y, chosen_firm = read_workers_data(args.workers_path)

    N = x_skill.size
    J = firm_ids.size

    alpha = params.get('alpha', 1.0)
    phi = params.get('varphi', params.get('phi', 1.0))
    mu_a = params.get('mu_a', 0.0)
    sigma_a = params.get('sigma_a', 0.12)
    gamma0_param = params.get('gamma', 0.05)

    V0_nat = alpha * np.log(np.maximum(w, 1e-300)) + xi
    C0_nat = np.maximum(c_data, 1e-10)

    if args.theta0_from_helper:
        print("Initialising θ via helpers.naive_theta_guess_gamma_V_c ...")
        theta_helper = naive_theta_guess_gamma_V_c(
            x_skill=x_skill,
            ell_x=ell_x,
            ell_y=ell_y,
            chosen_firm=chosen_firm,
            loc_firms=loc_firms,
            firm_ids=firm_ids,
            firms_csv_path=args.firms_path,
        )
        theta_helper = np.asarray(theta_helper, dtype=float)
        if theta_helper.size != 1 + 2 * J:
            raise ValueError("helpers.naive_theta_guess_gamma_V_c returned unexpected size")
        gamma_init = float(theta_helper[0])
        V_init_np = theta_helper[1:1 + J]
        c_init_np = np.maximum(theta_helper[1 + J:1 + 2 * J], 1e-8)
        theta0_np = np.concatenate(([gamma_init], c_init_np))
        V_init_default_np = V_init_np
    else:
        theta0_np = parse_theta0_gamma_c(args, J, gamma0_param, C0_nat)
        theta0_np = np.asarray(theta0_np, dtype=float)
        V_init_default_np = V0_nat.astype(float)

    theta0_np = np.asarray(theta0_np, dtype=float)
    assert theta0_np.size == 1 + J

    V_init_default = jnp.asarray(V_init_default_np, dtype=jnp.float64)

    D_nat = compute_worker_firm_distances(ell_x, ell_y, loc_firms)

    aux: Dict = {
        'D_nat': jnp.asarray(D_nat, dtype=jnp.float64),
        'phi': float(phi),
        'mu_a': float(mu_a),
        'sigma_a': float(sigma_a),
        'firm_ids': jnp.asarray(firm_ids, dtype=jnp.int32),
    }
    X = jnp.asarray(x_skill, dtype=jnp.float64)
    chosen = chosen_firm.astype(int)
    Y = build_choice_matrix(chosen, J)
    Y_jax = jnp.asarray(Y, dtype=jnp.float64)

    share_solver = args.share_solver.lower()
    share_tol = float(args.share_tol)
    share_maxiter = int(args.share_maxiter)
    share_debug = bool(args.share_debug)
    jac_chunk = max(1, int(args.jac_chunk))

    lb = jnp.full((theta0_np.size,), -jnp.inf)
    ub = jnp.full((theta0_np.size,), jnp.inf)
    lb = lb.at[0].set(0.0)
    ub = ub.at[0].set(1.0)
    for idx in range(J):
        lb = lb.at[1 + idx].set(1e-8)

    fwd, inv = make_reparam(lb, ub)
    theta0 = jnp.asarray(theta0_np, dtype=jnp.float64)
    z0 = inv(theta0)

    shares_empirical = np.bincount(chosen_firm, minlength=J + 1) / max(N, 1)
    shares_empirical_jax = jnp.asarray(shares_empirical[1:], dtype=jnp.float64)

    def share_residual_gamma_c(V: jnp.ndarray, gamma: jnp.ndarray, c_vec: jnp.ndarray) -> jnp.ndarray:
        theta_full = jnp.concatenate((gamma[None], V, c_vec))
        P = compute_choice_probabilities_jax(theta_full, X, aux)
        shares_model = jnp.mean(P[:, 1:], axis=0)
        return shares_model - shares_empirical_jax

    if share_solver == 'lm':
        lm_solver = LevenbergMarquardt(
            residual_fun=share_residual_gamma_c,
            tol=share_tol,
            maxiter=share_maxiter,
            implicit_diff=True,
        )
    else:
        lm_solver = None

    logs_shares_emp = jnp.log(jnp.clip(shares_empirical_jax, 1e-12, 1.0))

    def solve_V_blp(theta_gc: jnp.ndarray) -> jnp.ndarray:
        gamma = theta_gc[0]
        c_vec = theta_gc[1:]

        def update(state):
            V_curr, _ = state
            theta_full = jnp.concatenate((gamma[None], V_curr, c_vec))
            P = compute_choice_probabilities_jax(theta_full, X, aux)
            shares_model = jnp.mean(P[:, 1:], axis=0)
            log_diff = logs_shares_emp - jnp.log(jnp.clip(shares_model, 1e-12, 1.0))
            V_next = V_curr + log_diff
            err = jnp.max(jnp.abs(log_diff))
            if share_debug:
                share_min = jnp.min(shares_model)
                share_max = jnp.max(shares_model)
                jax.debug.print(
                    "[BLP] iter ΔV max={err:.3e}, share_min={share_min:.3e}, share_max={share_max:.3e}",
                    err=err,
                    share_min=share_min,
                    share_max=share_max,
                )
            return V_next, err

        def body(i, state):
            return jax.lax.cond(state[1] > share_tol, update, lambda s: s, state)

        init_state = (V_init_default, jnp.inf)
        V_final, _ = jax.lax.fori_loop(0, share_maxiter, body, init_state)
        return V_final

    def solve_V(theta_gc: jnp.ndarray) -> jnp.ndarray:
        if share_solver == 'lm':
            gamma = theta_gc[0]
            c_vec = theta_gc[1:]
            res = lm_solver.run(V_init_default, gamma, c_vec)
            return res.params
        return solve_V_blp(theta_gc)

    def theta_full_from_gc(theta_gc: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        V_solution = solve_V(theta_gc)
        theta_full = jnp.concatenate((theta_gc[0:1], V_solution, theta_gc[1:]))
        return theta_full, V_solution

    def log_odds_profile(theta_gc: jnp.ndarray) -> jnp.ndarray:
        theta_full, _ = theta_full_from_gc(theta_gc)
        P = compute_choice_probabilities_jax(theta_full, X, aux)
        log_prob = jnp.log(jnp.clip(P, 1e-12, 1.0))
        log_odds = log_prob[:, 1:] - log_prob[:, 0:1]
        return log_odds.reshape((-1,))

    log_odds_profile = jax.jit(log_odds_profile)

    def instruments_gamma_c(theta_gc: jnp.ndarray) -> jnp.ndarray:
        K = int(theta_gc.size)
        eye = jnp.eye(K, dtype=theta_gc.dtype)

        def jvp_single(vec):
            return jax.jvp(log_odds_profile, (theta_gc,), (vec,))[1]

        blocks = []
        for start in range(0, K, jac_chunk):
            stop = min(start + jac_chunk, K)
            block = eye[start:stop]
            block_jac = jax.vmap(jvp_single)(block)
            blocks.append(block_jac)

        J_theta = jnp.concatenate(blocks, axis=0)
        return J_theta.T.reshape((N, J, K))

    theta_gc_base = jnp.asarray(theta0_np, dtype=jnp.float64)
    G_feat = instruments_gamma_c(theta_gc_base)

    def residuals(theta_gc: jnp.ndarray) -> jnp.ndarray:
        theta_full, _ = theta_full_from_gc(theta_gc)
        P = compute_choice_probabilities_jax(theta_full, X, aux)
        R = Y_jax[:, 1:] - P[:, 1:]
        return R

    residuals = jax.jit(residuals)

    def moment_gc(theta_gc: jnp.ndarray) -> jnp.ndarray:
        R = residuals(theta_gc)
        return jnp.tensordot(R, G_feat, axes=([0, 1], [0, 1]))

    moment_gc = jax.jit(moment_gc)

    def gmm_objective(theta_gc: jnp.ndarray) -> jnp.ndarray:
        m = moment_gc(theta_gc)
        return jnp.dot(m, m)

    gmm_objective = jax.jit(gmm_objective)

    solver = LBFGS(fun=gmm_objective, value_and_grad=False, maxiter=int(args.maxiter), tol=args.tol)

    start = time.perf_counter()
    res = solver.run(z0)
    total_time = time.perf_counter() - start

    z_hat = res.params
    theta_hat_gc = np.asarray(fwd(z_hat))
    obj = float(gmm_objective(z_hat))
    nit = int(res.state.iter_num)
    grad_norm = float(jnp.linalg.norm(res.state.grad))

    print(f"[LBFGS] obj={obj:.6f}, nit={nit}, grad_norm={grad_norm:.3e}, time={total_time:.2f}s")

    theta_hat_gc_jax = jnp.asarray(theta_hat_gc, dtype=jnp.float64)
    theta_hat_full_jax, V_hat_jax = theta_full_from_gc(theta_hat_gc_jax)

    P_hat = compute_choice_probabilities_jax(theta_hat_full_jax, X, aux)
    shares_model = np.asarray(P_hat).mean(axis=0)

    shares_empirical = np.bincount(chosen_firm, minlength=J + 1) / max(N, 1)
    print("Market shares (empirical vs model):")
    for idx in range(J + 1):
        name = 'outside' if idx == 0 else f'firm_{idx}'
        print(f"  {name}: {shares_empirical[idx]:.6f} vs {shares_model[idx]:.6f}")

    moment_final = np.asarray(moment_gc(theta_hat_gc_jax))
    G_final = np.asarray(G_feat)
    theta_hat_np = np.asarray(theta_hat_full_jax, dtype=float)

    lb_np = np.asarray(lb)
    ub_np = np.asarray(ub)

    def moments_fn(th_vec: np.ndarray):
        th_vec = np.asarray(th_vec, dtype=float)
        theta_gc = th_vec
        theta_full, _ = theta_full_from_gc(jnp.asarray(theta_gc, dtype=jnp.float64))
        P = compute_choice_probabilities_jax(theta_full, X, aux)
        R = np.asarray(Y)[:, 1:] - np.asarray(P)[:, 1:]
        Psi = np.einsum('nj,njk->nk', R, G_final)
        m_vec = Psi.mean(axis=0)
        return m_vec, Psi

    se_vec = np.full(theta_hat_gc.size, np.nan, dtype=float)
    ci_radius = np.full(theta_hat_gc.size, np.nan, dtype=float)
    se_mode = None

    param_names = ['gamma'] + [f'c_{j+1}' for j in range(J)]

    if args.skip_statistics:
        print("Skipping robust standard errors (--skip_statistics).")
    else:
        try:
            se_results = compute_gmm_standard_errors(
                theta_hat_gc,
                moments_fn,
                mode='robust',
                bounds=(lb_np, ub_np),
            )
            se_vec = se_results['se']
            se_mode = se_results.get('mode')
            ci_radius = 1.96 * se_vec
            print("Robust standard errors:")
            for name, se in zip(param_names, se_vec):
                print(f"  {name}: {se:.6f}")
        except (MemoryError, RuntimeError, ValueError) as exc:
            print(f"Warning: skipping robust statistics due to: {exc}")
            se_vec = np.full(theta_hat_gc.size, np.nan, dtype=float)
            ci_radius = np.full(theta_hat_gc.size, np.nan, dtype=float)
            se_mode = None

    print("Solved-for V (inclusive values):")
    V_hat = np.asarray(V_hat_jax, dtype=float)
    for idx, v_val in enumerate(V_hat, start=1):
        print(f"  V_{idx}: {v_val:.6f}")

    out = {
        'solver': 'LBFGS',
        'objective': obj,
        'nit': nit,
        'grad_norm': grad_norm,
        'theta0_gamma_c': theta0_np.tolist(),
        'theta_hat_gamma_c': theta_hat_gc.tolist(),
        'V_hat': V_hat.tolist(),
        'time_sec': total_time,
        'distance_metrics': {
            'gamma_error': float(theta_hat_gc[0] - gamma0_param),
            'c_l2': float(np.linalg.norm(theta_hat_gc[1:] - C0_nat) / np.sqrt(J)),
        },
        'param_names': param_names,
        'robust_se': se_vec.tolist(),
        'ci_radius_95': ci_radius.tolist(),
        'se_mode': se_mode,
        'market_shares': {
            'empirical': shares_empirical.tolist(),
            'model': shares_model.tolist(),
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'gmm_gamma_c_profile_estimates_jax_dense.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
