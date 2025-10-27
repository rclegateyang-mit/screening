#!/usr/bin/env python3
"""
Executable to run JAX-based GMM estimation for (gamma, V, c) via moment minimization.

CLI
---
--workers_path, --firms_path, --params_path, --out_dir
--theta0_file | --theta0_list
--solver {gn,lm}
--maxiter, --tol
--threads (int): print XLA_FLAGS export line the user can set before Python.

Notes
-----
Chamberlain optimal instruments are evaluated implicitly via VJP pulls of the
log-odds map, so the script avoids materialising the dense (N, J, K) tensor.

The probability kernel and streamed moments are implemented in JAX and
JIT-compiled on first call. Box bounds on gamma∈[0,1] are handled via smooth
reparametrisation.
"""

from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

try:
    # Try relative imports first (when run as module)
    from .jax_model import enable_x64
    from .optimize_gmm import solve_gn, solve_lm
    from . import jax_model
    from .helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
        naive_theta_guess_gamma_V_c,
        compute_gmm_standard_errors,
    )
except ImportError:
    # Fall back to absolute imports (when run directly)
    from jax_model import enable_x64
    from optimize_gmm import solve_gn, solve_lm
    import jax_model
    from helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
        naive_theta_guess_gamma_V_c,
        compute_gmm_standard_errors,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run JAX GMM for (gamma, V, c)")
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / 'output'
    
    p.add_argument('--workers_path', type=str, default=str(output_dir / 'workers_dataset.csv'))
    p.add_argument('--firms_path', type=str, default=str(output_dir / 'equilibrium_firms.csv'))
    p.add_argument('--params_path', type=str, default=str(output_dir / 'parameters_effective.csv'))
    p.add_argument('--out_dir', type=str, default=str(output_dir))
    p.add_argument('--theta0_file', type=str, default=None)
    p.add_argument('--theta0_list', type=str, default=None,
                   help='Comma-separated γ, V(1..J), c(1..J); if omitted use baseline from data.')
    p.add_argument('--theta0_from_helper', action='store_true',
                   help='Initialise θ0 via helpers.naive_theta_guess_gamma_V_c using data in firms/workers CSVs.')
    p.add_argument('--k_step', type=int, default=1,
                   help='Number of K-step GMM iterations (recompute instruments each step).')
    p.add_argument('--lm_init_damping', type=float, default=None,
                   help='Initial Levenberg-Marquardt damping parameter λ₀ (larger ⇒ more regularisation).')
    p.add_argument('--solver', type=str, choices=['gn', 'lm'], default='gn')
    p.add_argument('--maxiter', type=int, default=500)
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--threads', type=int, default=None)
    p.add_argument('--chamberlain_mode', type=str, choices=['auto', 'full', 'chunked', 'fd'], default='auto')
    p.add_argument('--fd_rel_step', type=float, default=1e-5,
                   help='Relative step for finite-difference instruments (mode="fd").')
    p.add_argument('--fd_abs_step', type=float, default=1e-6,
                   help='Absolute step for finite-difference instruments (mode="fd").')
    p.add_argument('--skip_statistics', action='store_true',
                   help='Skip robust SE / covariance computation to save memory.')
    p.add_argument('--skip_plot', action='store_true',
                   help='Skip plotting to save memory/time.')
    p.add_argument('--jac_chunk', type=int, default=32,
                   help='Chunk size when streaming directional derivatives (affects SE construction).')
    return p.parse_args()


def parse_theta0_extended(
    args: argparse.Namespace,
    J: int,
    V0_nat: np.ndarray,
    C0_nat: np.ndarray,
    gamma0: float,
) -> np.ndarray:
    """Parse θ0 = [γ, V(1..J), c(1..J)] from CLI inputs or fall back to baseline."""

    K_expected = 1 + 2 * J

    if args.theta0_list is not None:
        raw = args.theta0_list.replace('\n', ' ').replace('\t', ' ')
        values: list[float] = []
        for part in raw.split(','):
            values.extend([float(x.strip()) for x in part.split() if x.strip()])
        theta0 = np.asarray(values, dtype=float)
        if theta0.size != K_expected:
            raise ValueError(
                f"θ0 length {theta0.size} != 1+2J={K_expected}. Provide γ, V(1..J), c(1..J)."
            )
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
            raise ValueError(f"Unsupported theta0 file format: {file_path}. Use .json or .csv")

        if theta0.size != K_expected:
            raise ValueError(
                f"θ0 length {theta0.size} != 1+2J={K_expected}. Provide γ, V(1..J), c(1..J)."
            )
        return theta0

    return np.concatenate(([gamma0], V0_nat, C0_nat))


def main():
    args = parse_args()

    if args.threads is not None:
        print("Set CPU threads by exporting before Python:")
        print(f'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_cpu_thread_pool_size={int(args.threads)}"')

    print(f"Chamberlain instrument mode: {args.chamberlain_mode}")

    enable_x64()

    # Load data
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

    # Baselines in natural order
    V0_nat = alpha * np.log(np.maximum(w, 1e-300)) + xi
    C0_nat = np.maximum(c_data, 1e-10)
    if args.theta0_from_helper:
        print("Initialising θ0 via helpers.naive_theta_guess_gamma_V_c ...")
        theta0_np = naive_theta_guess_gamma_V_c(
            x_skill=x_skill,
            ell_x=ell_x,
            ell_y=ell_y,
            chosen_firm=chosen_firm,
            loc_firms=loc_firms,
            firm_ids=firm_ids,
            gamma0=gamma0_param,
            firms_csv_path=args.firms_path,
        )
        theta0_np = np.asarray(theta0_np, dtype=float)
    else:
        theta0_np = parse_theta0_extended(args, J, V0_nat, C0_nat, gamma0_param)
        theta0_np = np.asarray(theta0_np, dtype=float)
    K = theta0_np.size
    assert K == 1 + 2 * J, f"theta0 length {K} != 1+2J={1+2*J}"

    # Distances (host), then move to device
    D_nat = compute_worker_firm_distances(ell_x, ell_y, loc_firms)  # (N,J)

    # Build aux for JAX model
    aux: Dict = {
        'D_nat': jnp.asarray(D_nat, dtype=jnp.float64),
        'phi': float(phi),
        'mu_a': float(mu_a),
        'sigma_a': float(sigma_a),
        'firm_ids': jnp.asarray(firm_ids, dtype=jnp.int32),
    }
    X = jnp.asarray(x_skill, dtype=jnp.float64)  # (N,)
    theta0 = jnp.asarray(theta0_np, dtype=jnp.float64)

    import time

    def prob_evaluator(theta_arr: jnp.ndarray) -> jnp.ndarray:
        return jax_model.compute_choice_probabilities_jax(theta_arr, X, aux)

    # Bounds: gamma in [0,1], others free
    lb = jnp.full((K,), -jnp.inf)
    ub = jnp.full((K,), jnp.inf)
    lb = lb.at[0].set(0.0)
    ub = ub.at[0].set(1.0)
    for idx in range(J):
        lb = lb.at[1 + J + idx].set(1e-8)
        ub = ub.at[1 + J + idx].set(jnp.inf)

    solver_name = 'GaussNewton' if args.solver == 'gn' else 'LevenbergMarquardt'
    lm_solver_kwargs: Dict[str, float] = {}
    if args.solver == 'lm' and args.lm_init_damping is not None:
        lm_solver_kwargs['damping_parameter'] = float(args.lm_init_damping)
        print("LM stabilisation:")
        print(f"  - Initial damping parameter λ₀={lm_solver_kwargs['damping_parameter']:.3g} (larger ⇒ more regularisation)")

    Y_onehot_np = pd.get_dummies(chosen_firm).reindex(columns=range(J + 1), fill_value=0).values
    Y_mat = jnp.asarray(Y_onehot_np, dtype=jnp.float64)

    k_steps = int(args.k_step) if args.k_step and args.k_step > 0 else 1

    os.makedirs(args.out_dir, exist_ok=True)

    mode_sel = args.chamberlain_mode.lower()
    use_fd = mode_sel == 'fd'
    jac_chunk = max(1, int(args.jac_chunk))
    fd_rel_step = float(args.fd_rel_step)
    fd_abs_step = float(args.fd_abs_step)

    @jax.jit
    def log_odds(theta_arr: jnp.ndarray) -> jnp.ndarray:
        P = prob_evaluator(theta_arr)
        log_prob = jnp.log(jnp.clip(P, 1e-300, 1.0))
        return log_prob[:, 1:] - log_prob[:, 0:1]

    @jax.jit
    def residuals(theta_arr: jnp.ndarray) -> jnp.ndarray:
        P = prob_evaluator(theta_arr)
        return Y_mat[:, 1:] - P[:, 1:]

    @jax.jit
    def jvp_log_odds(theta_arr: jnp.ndarray, direction: jnp.ndarray) -> jnp.ndarray:
        return jax.jvp(log_odds, (theta_arr,), (direction,))[1]

    @jax.jit
    def fd_direction(theta_arr: jnp.ndarray, direction: jnp.ndarray, delta: float) -> jnp.ndarray:
        theta_plus = theta_arr + delta * direction
        theta_minus = theta_arr - delta * direction
        return (log_odds(theta_plus) - log_odds(theta_minus)) / (2.0 * delta)

    if use_fd:
        def moment_stream(theta_arr: jnp.ndarray) -> jnp.ndarray:
            theta_arr = jnp.asarray(theta_arr, dtype=jnp.float64)
            R = residuals(theta_arr)
            steps_vec = jnp.maximum(
                fd_abs_step,
                fd_rel_step * jnp.maximum(1.0, jnp.abs(theta_arr)),
            )
            basis = jnp.eye(theta_arr.size, dtype=theta_arr.dtype)

            def fd_contrib(vec, delta):
                jac_slice = fd_direction(theta_arr, vec, delta)
                return jnp.tensordot(R, jac_slice, axes=((0, 1), (0, 1)))

            return jax.vmap(fd_contrib)(basis, steps_vec)
    else:
        def moment_stream(theta_arr: jnp.ndarray) -> jnp.ndarray:
            theta_arr = jnp.asarray(theta_arr, dtype=jnp.float64)
            R = residuals(theta_arr)
            _, vjp_fun = jax.vjp(log_odds, theta_arr)
            return vjp_fun(R)[0]

    moment_stream = jax.jit(moment_stream)

    def directional_block(theta_arr: jnp.ndarray, basis_block: jnp.ndarray, step_block):
        if use_fd:
            return jax.vmap(lambda vec, delta: fd_direction(theta_arr, vec, delta))(basis_block, step_block)
        return jax.vmap(lambda vec: jvp_log_odds(theta_arr, vec))(basis_block)

    theta_current = theta0
    total_start = time.perf_counter()
    total_build_time = 0.0
    total_solve_time = 0.0
    steps_info = []

    for step_idx in range(k_steps):
        t_build_start = time.perf_counter()
        _ = moment_stream(theta_current)
        build_time = float(time.perf_counter() - t_build_start)
        total_build_time += build_time

        t_solve_start = time.perf_counter()
        if args.solver == 'gn':
            res = solve_gn(
                theta_current,
                X,
                Y_mat,
                None,
                aux,
                maxiter=args.maxiter,
                tol=args.tol,
                lb=lb,
                ub=ub,
                moment_fn=moment_stream,
            )
        else:
            res = solve_lm(
                theta_current,
                X,
                Y_mat,
                None,
                aux,
                maxiter=args.maxiter,
                tol=args.tol,
                lb=lb,
                ub=ub,
                solver_kwargs=dict(lm_solver_kwargs),
                moment_fn=moment_stream,
            )
        solve_time = float(time.perf_counter() - t_solve_start)
        total_solve_time += solve_time

        g_density = None
        g_used = 'implicit'

        theta_hat_step = jnp.asarray(res['theta_hat'], dtype=jnp.float64)
        obj_step = float(res['obj'])
        nit_step = res['nit']
        grad_norm_step = float(res['grad_norm'])

        P_step = prob_evaluator(theta_hat_step)
        P_np = np.asarray(P_step)
        row_sums = P_np.sum(axis=1)
        try:
            P_min = float(np.nanmin(P_np))
        except ValueError:  # all NaN
            P_min = float('nan')
        try:
            P_max = float(np.nanmax(P_np))
        except ValueError:
            P_max = float('nan')
        try:
            row_sum_min = float(np.nanmin(row_sums))
        except ValueError:
            row_sum_min = float('nan')
        try:
            row_sum_max = float(np.nanmax(row_sums))
        except ValueError:
            row_sum_max = float('nan')
        has_nan = bool(np.isnan(P_np).any())
        has_inf = bool(np.isinf(P_np).any())

        try:
            moment_vec = moment_stream(theta_hat_step)
            moment_vec_np = np.asarray(moment_vec)
            moment_norm = float(np.linalg.norm(moment_vec_np))
            moment_max = float(np.nanmax(np.abs(moment_vec_np)))
            moment_nan = bool(np.isnan(moment_vec_np).any())
        except Exception as err:  # pragma: no cover - diagnostic path
            moment_norm = float('nan')
            moment_max = float('nan')
            moment_nan = True
            print(f"Warning: failed to evaluate moment at step {step_idx + 1}: {err}")

        print(
            f"[Step {step_idx + 1}] γ={float(theta_hat_step[0]):.6f}, obj={obj_step:.6e}, "
            f"grad_norm={grad_norm_step:.3e}, moment_norm={moment_norm:.3e}"
        )
        print(
            f"            P range=[{P_min:.2e}, {P_max:.2e}], row-sum range=[{row_sum_min:.6f}, {row_sum_max:.6f}], "
            f"has_nan={has_nan}, has_inf={has_inf}, moment_nan={moment_nan}, max|moment|={moment_max:.3e}"
        )

        steps_info.append({
            'step': step_idx + 1,
            'theta_hat': np.asarray(theta_hat_step).tolist(),
            'objective': obj_step,
            'nit': int(nit_step) if nit_step is not None else None,
            'grad_norm': grad_norm_step,
            'build_time_sec': build_time,
            'solve_time_sec': solve_time,
            'g_feat_density': g_density,
            'g_feat_storage': g_used,
            'instrument_mode': args.chamberlain_mode,
            'probability_min': P_min,
            'probability_max': P_max,
            'rowsum_min': row_sum_min,
            'rowsum_max': row_sum_max,
            'prob_has_nan': has_nan,
            'prob_has_inf': has_inf,
            'moment_norm': moment_norm,
            'moment_max_abs': moment_max,
            'moment_has_nan': moment_nan,
        })

        theta_current = theta_hat_step

    total_time = float(time.perf_counter() - total_start)

    print(
        f"Runtime summary: N={N}, J={J}, total={total_time:.2f}s "
        f"(build={total_build_time:.2f}s, solve={total_solve_time:.2f}s)"
    )

    theta_hat = np.asarray(theta_current)
    obj = steps_info[-1]['objective']
    nit = steps_info[-1]['nit']
    grad_norm = steps_info[-1]['grad_norm']
    g_density = steps_info[-1]['g_feat_density']
    g_used = steps_info[-1]['g_feat_storage']

    # Compute robust standard errors using streaming instruments
    theta_hat_np = np.asarray(theta_hat, dtype=float)
    Y_full_np = Y_onehot_np

    lb_np = np.asarray(lb)
    ub_np = np.asarray(ub)

    def moments_fn(th_vec: np.ndarray):
        th_vec = np.asarray(th_vec, dtype=float)
        theta_jax = jnp.asarray(th_vec, dtype=jnp.float64)
        m_vec = np.asarray(moment_stream(theta_jax))

        P = prob_evaluator(theta_jax)
        R = Y_mat[:, 1:] - P[:, 1:]
        R_const = jax.lax.stop_gradient(R)

        basis = jnp.eye(theta_jax.size, dtype=theta_jax.dtype)
        if use_fd:
            steps_full = jnp.maximum(
                fd_abs_step,
                fd_rel_step * jnp.maximum(1.0, jnp.abs(theta_jax)),
            )
        else:
            steps_full = None

        psi_blocks = []
        for start in range(0, theta_jax.size, jac_chunk):
            stop = min(start + jac_chunk, theta_jax.size)
            block = basis[start:stop]
            if use_fd:
                step_block = steps_full[start:stop]
                jac_block = directional_block(theta_jax, block, step_block)
            else:
                jac_block = directional_block(theta_jax, block, None)
            psi_block = jnp.einsum('cnj,nj->nc', jac_block, R_const).T
            psi_blocks.append(psi_block)

        Psi = jnp.concatenate(psi_blocks, axis=1)
        return m_vec, np.asarray(Psi)

    se_vec = np.full(theta_hat_np.size, np.nan, dtype=float)
    ci_radius = np.full(theta_hat_np.size, np.nan, dtype=float)
    ci_lower = np.full(theta_hat_np.size, np.nan, dtype=float)
    ci_upper = np.full(theta_hat_np.size, np.nan, dtype=float)
    se_mode = None

    if args.skip_statistics:
        print("Skipping robust standard errors (--skip_statistics).")
    else:
        try:
            se_results = compute_gmm_standard_errors(
                theta_hat_np,
                moments_fn,
                mode='robust',
                bounds=(lb_np, ub_np),
            )
            se_vec = se_results['se']
            se_mode = se_results.get('mode')
            ci_radius = 1.96 * se_vec
            ci_lower = theta_hat_np - ci_radius
            ci_upper = theta_hat_np + ci_radius

            print("Robust standard errors:")
            print(f"  gamma: {theta_hat_np[0]:.6f} ± {se_vec[0]:.6f}")
            for j_idx in range(J):
                print(f"  V_{j_idx+1}: {theta_hat_np[1+j_idx]:.6f} ± {se_vec[1+j_idx]:.6f}")
            for j_idx in range(J):
                print(f"  c_{j_idx+1}: {theta_hat_np[1+J+j_idx]:.6f} ± {se_vec[1+J+j_idx]:.6f}")
        except (MemoryError, RuntimeError, ValueError) as exc:
            print(f"Warning: skipping robust statistics due to: {exc}")
            se_vec = np.full(theta_hat_np.size, np.nan, dtype=float)
            ci_radius = np.full(theta_hat_np.size, np.nan, dtype=float)
            ci_lower = np.full(theta_hat_np.size, np.nan, dtype=float)
            ci_upper = np.full(theta_hat_np.size, np.nan, dtype=float)
            se_mode = None

    gamma_true = float(gamma0_param)
    V_true = V0_nat.astype(float)
    C_true = C0_nat.astype(float)
    V_hat = theta_hat_np[1:1+J]
    C_hat = theta_hat_np[1+J:]

    norm_scale = np.sqrt(J) if J > 0 else 1.0
    distance_metrics = {
        'gamma_error': float(theta_hat_np[0] - gamma_true),
        'V_l2': float(np.linalg.norm(V_hat - V_true) / norm_scale),
        'V_max_abs': float(np.max(np.abs(V_hat - V_true))),
        'c_l2': float(np.linalg.norm(C_hat - C_true) / norm_scale),
        'c_max_abs': float(np.max(np.abs(C_hat - C_true))),
    }

    print("\nDistance to baseline θ₀ (from firms/parameters files):")
    print(f"  |γ̂ - γ| = {abs(distance_metrics['gamma_error']):.6f}")
    print(f"  (1/√J)·||V̂ - V||₂ = {distance_metrics['V_l2']:.6f} (max abs {distance_metrics['V_max_abs']:.6f})")
    print(f"  (1/√J)·||ĉ - c||₂ = {distance_metrics['c_l2']:.6f} (max abs {distance_metrics['c_max_abs']:.6f})")

    shares_empirical = Y_full_np.mean(axis=0)
    shares_model = np.asarray(prob_evaluator(theta_hat), dtype=float).mean(axis=0)

    print("\nMarket shares (mean across workers):")
    for idx in range(J + 1):
        if idx == 0:
            name = 'outside'
        else:
            name = f'firm_{idx}'
        print(f"  {name}: empirical={shares_empirical[idx]:.6f}, model={shares_model[idx]:.6f}")

    true_vector = np.concatenate(([gamma_true], V_true, C_true))

    labels = ["gamma"] + [f"V_{idx+1}" for idx in range(J)] + [f"c_{idx+1}" for idx in range(J)]

    if args.skip_plot:
        ci_plot_path = None
        print("Skipping plot generation (--skip_plot).")
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

        ci_gamma = ci_radius[0] if np.isfinite(ci_radius[0]) else 0.0
        axes[0].errorbar(
            [0],
            [theta_hat_np[0]],
            yerr=[ci_gamma],
            fmt='o',
            capsize=4,
            label='Estimate ±95% CI',
        )
        axes[0].scatter([0], [true_vector[0]], marker='x', color='red', label='Baseline θ₀')
        axes[0].set_xticks([0])
        axes[0].set_xticklabels(['gamma'])
        axes[0].set_ylabel('γ')
        axes[0].set_title('Gamma estimate')
        axes[0].legend()

        v_indices = np.arange(J)
        ci_V = np.where(np.isfinite(ci_radius[1:1+J]), ci_radius[1:1+J], 0.0)
        axes[1].errorbar(
            v_indices,
            theta_hat_np[1:1+J],
            yerr=ci_V,
            fmt='o',
            capsize=4,
            label='Estimate ±95% CI',
        )
        axes[1].scatter(v_indices, true_vector[1:1+J], marker='x', color='red', label='Baseline θ₀')
        axes[1].set_xticks(v_indices)
        axes[1].set_xticklabels([f"V_{idx+1}" for idx in range(J)], rotation=45, ha='right')
        axes[1].set_ylabel('V')
        axes[1].set_title('Firm V estimates')
        axes[1].legend()

        c_indices = np.arange(J)
        ci_C = np.where(np.isfinite(ci_radius[1+J:]), ci_radius[1+J:], 0.0)
        axes[2].errorbar(
            c_indices,
            C_hat,
            yerr=ci_C,
            fmt='o',
            capsize=4,
            label='Estimate ±95% CI',
        )
        axes[2].scatter(c_indices, C_true, marker='x', color='red', label='Baseline θ₀')
        axes[2].set_xticks(c_indices)
        axes[2].set_xticklabels([f"c_{idx+1}" for idx in range(J)], rotation=45, ha='right')
        axes[2].set_ylabel('c')
        axes[2].set_title('Firm cutoff estimates')
        axes[2].legend()
        ci_plot_path = os.path.join(args.out_dir, 'theta_ci_plot.png')
        fig.savefig(ci_plot_path, dpi=200)
        plt.close(fig)


    # Prepare output
    out = {
        'solver': solver_name,
        'N': int(N),
        'J': int(J),
        'K': int(K),
        'p': int(K),
        'theta0': theta0_np.tolist(),
        'theta_hat': theta_hat.tolist(),
        'objective': obj,
        'nit': int(nit) if nit is not None else None,
        'grad_norm': grad_norm,
        'g_feat_mode': 'chamberlain',
        'chamberlain_mode': args.chamberlain_mode,
        'g_feat_density': g_density,
        'g_feat_storage': g_used,
        'k_step': int(k_steps),
        'steps': steps_info,
        'standard_errors': se_vec.tolist(),
        'se_mode': se_mode,
        'ci_lower': ci_lower.tolist(),
        'ci_upper': ci_upper.tolist(),
        'distance_metrics': distance_metrics,
        'baseline_theta0': {
            'gamma': gamma_true,
            'V': V_true.tolist(),
            'c': C_true.tolist(),
        },
        'market_shares': {
            'empirical': shares_empirical.tolist(),
            'model': shares_model.tolist(),
        },
        'timings': {
            'total_time_sec': total_time,
            'total_build_time_sec': total_build_time,
            'total_solve_time_sec': total_solve_time,
            'per_step': steps_info,
        },
        'threads_used': int(args.threads) if args.threads is not None else None,
    }
    if ci_plot_path is not None:
        out['ci_plot'] = ci_plot_path
    out_path = os.path.join(args.out_dir, 'gmm_gamma_Vc_estimates_jax.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"[{solver_name}] K={k_steps}, obj={obj:.6e}, nit={nit}, grad_norm={grad_norm:.3e}")
    print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
