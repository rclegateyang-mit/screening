#!/usr/bin/env python3
"""Maximum-likelihood estimation of (gamma, V, c) using JAX."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxopt import LBFGS
import matplotlib.pyplot as plt

os.environ.setdefault("JAX_ENABLE_X64", "1")

try:
    from .jax_model import enable_x64, compute_choice_probabilities_jax
    from .helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
    )
    from .optimize_gmm import make_reparam
except ImportError:
    from jax_model import enable_x64, compute_choice_probabilities_jax
    from helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
    )
    from optimize_gmm import make_reparam


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent.parent
    output_dir = root / "output"

    parser = argparse.ArgumentParser(description="Run JAX MLE for (gamma, V, c)")
    parser.add_argument('--workers_path', type=str, default=str(output_dir / 'workers_dataset.csv'))
    parser.add_argument('--firms_path', type=str, default=str(output_dir / 'equilibrium_firms.csv'))
    parser.add_argument('--params_path', type=str, default=str(output_dir / 'parameters_effective.csv'))
    parser.add_argument('--out_dir', type=str, default=str(output_dir))
    parser.add_argument('--theta0_file', type=str, default=None)
    parser.add_argument('--theta0_list', type=str, default=None,
                        help='Comma-separated γ, V(1..J), c(1..J); if omitted use baseline from data.')
    parser.add_argument('--theta0_from_helper', action='store_true',
                        help='Initialise θ0 via helpers.naive_theta_guess_gamma_V_c using data in firms/workers CSVs.')
    parser.add_argument('--maxiter', type=int, default=500)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--skip_statistics', action='store_true',
                        help='Skip robust SE / covariance computation to save memory.')
    parser.add_argument('--skip_plot', action='store_true',
                        help='Skip plotting to save memory/time.')
    return parser.parse_args()


def parse_theta0_extended(
    args: argparse.Namespace,
    J: int,
    V0_nat: np.ndarray,
    C0_nat: np.ndarray,
    gamma0: float,
) -> np.ndarray:
    k_expected = 1 + 2 * J

    if args.theta0_list is not None:
        raw = args.theta0_list.replace('\n', ' ').replace('\t', ' ')
        values: list[float] = []
        for part in raw.split(','):
            values.extend([float(x.strip()) for x in part.split() if x.strip()])
        theta0 = np.asarray(values, dtype=float)
        if theta0.size != k_expected:
            raise ValueError(f"θ0 length {theta0.size} != 1+2J={k_expected}.")
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
            raise ValueError(f"θ0 length {theta0.size} != 1+2J={k_expected}.")
        return theta0

    return np.concatenate(([gamma0], V0_nat, C0_nat))


def main() -> None:
    args = parse_args()

    build_start = time.perf_counter()

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
        try:
            from .helpers import naive_theta_guess_gamma_V_c  # lazy import to avoid cycle
        except ImportError:  # pragma: no cover - fallback when running as script
            from helpers import naive_theta_guess_gamma_V_c

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

    assert theta0_np.size == 1 + 2 * J

    D_nat = compute_worker_firm_distances(ell_x, ell_y, loc_firms)

    aux: Dict = {
        'D_nat': jnp.asarray(D_nat, dtype=jnp.float64),
        'phi': float(phi),
        'mu_a': float(mu_a),
        'sigma_a': float(sigma_a),
        'firm_ids': jnp.asarray(firm_ids, dtype=jnp.int32),
    }
    X = jnp.asarray(x_skill, dtype=jnp.float64)
    choice_idx = jnp.asarray(chosen_firm.astype(np.int32))

    lb = jnp.full((theta0_np.size,), -jnp.inf)
    ub = jnp.full((theta0_np.size,), jnp.inf)
    lb = lb.at[0].set(0.0)
    ub = ub.at[0].set(1.0)
    for idx in range(J):
        lb = lb.at[1 + J + idx].set(1e-8)
        ub = ub.at[1 + J + idx].set(jnp.inf)

    fwd, inv = make_reparam(lb, ub)
    theta0 = jnp.asarray(theta0_np, dtype=jnp.float64)
    z0 = inv(theta0)

    def _per_obs_nll(theta: jnp.ndarray) -> jnp.ndarray:
        P = compute_choice_probabilities_jax(theta, X, aux)
        probs = jnp.take_along_axis(P, choice_idx[:, None], axis=1).squeeze(axis=1)
        safe_probs = probs + 1e-12
        return -jnp.log(safe_probs)

    def _neg_log_likelihood_theta(theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(_per_obs_nll(theta))

    neg_log_likelihood_theta = jax.jit(_neg_log_likelihood_theta)
    score_fn = jax.jacfwd(_per_obs_nll)

    def neg_log_likelihood_z(z: jnp.ndarray) -> jnp.ndarray:
        theta = fwd(z)
        return neg_log_likelihood_theta(theta)

    neg_log_likelihood_z = jax.jit(neg_log_likelihood_z)

    solver = LBFGS(fun=neg_log_likelihood_z, value_and_grad=False, maxiter=int(args.maxiter), tol=args.tol)

    build_time = time.perf_counter() - build_start

    solve_start = time.perf_counter()
    res = solver.run(z0)
    solve_time = time.perf_counter() - solve_start
    total_time = build_time + solve_time

    z_hat = res.params
    theta_hat = np.asarray(fwd(z_hat))
    obj = float(neg_log_likelihood_z(z_hat))
    nit = int(res.state.iter_num)
    grad_norm = float(jnp.linalg.norm(res.state.grad))

    print(
        f"[LBFGS] negloglik={obj:.6f}, nit={nit}, grad_norm={grad_norm:.3e}, "
        f"build={build_time:.2f}s, solve={solve_time:.2f}s, total={total_time:.2f}s"
    )

    theta_hat_jax = jnp.asarray(theta_hat, dtype=jnp.float64)

    shares_empirical = np.bincount(chosen_firm, minlength=J + 1) / max(N, 1)
    shares_model = np.asarray(compute_choice_probabilities_jax(theta_hat_jax, X, aux)).mean(axis=0)

    print("Market shares (empirical vs model):")
    for idx in range(J + 1):
        name = 'outside' if idx == 0 else f'firm_{idx}'
        print(f"  {name}: {shares_empirical[idx]:.6f} vs {shares_model[idx]:.6f}")

    param_names = ['gamma'] + [f'V_{j+1}' for j in range(J)] + [f'c_{j+1}' for j in range(J)]
    se_robust = np.full(len(param_names), np.nan, dtype=float)
    cov_robust = None
    if args.skip_statistics:
        print("Skipping robust standard errors (--skip_statistics).")
        ci_radius = np.full(len(param_names), np.nan, dtype=float)
    else:
        try:
            hessian = jax.hessian(_neg_log_likelihood_theta)(theta_hat_jax) / max(N, 1)
            score_mat = score_fn(theta_hat_jax)
            score_mat = np.asarray(score_mat, dtype=float).reshape(N, theta_hat.size)
            if not np.all(np.isfinite(score_mat)):
                raise ValueError("Non-finite score contributions detected; check numerical stability of likelihood.")

            J_theta = np.asarray(hessian, dtype=float)
            J_theta = 0.5 * (J_theta + J_theta.T)

            with np.errstate(divide='ignore', invalid='ignore'):
                S_theta = (score_mat.T @ score_mat) / max(N, 1)
            S_theta = 0.5 * (S_theta + S_theta.T)

            try:
                J_inv = np.linalg.inv(J_theta)
            except np.linalg.LinAlgError:
                J_inv = np.linalg.pinv(J_theta)

            cov_robust = J_inv @ S_theta @ J_inv / max(N, 1)
            cov_robust = 0.5 * (cov_robust + cov_robust.T)
            se_robust = np.sqrt(np.maximum(np.diag(cov_robust), 0.0))
            print("Robust (sandwich) standard errors:")
            for name, se in zip(param_names, se_robust):
                print(f"  {name}: {se:.6f}")
        except (MemoryError, RuntimeError, ValueError) as exc:
            print(f"Warning: skipping robust statistics due to: {exc}")
            cov_robust = None
            se_robust = np.full(len(param_names), np.nan, dtype=float)
        finally:
            ci_radius = 1.96 * se_robust

    theta_hat_np = np.asarray(theta_hat, dtype=float)

    gamma_base = float(gamma0_param)
    V_base = V0_nat.astype(float)
    C_base = C0_nat.astype(float)
    V_hat = theta_hat_np[1 : 1 + J]
    C_hat = theta_hat_np[1 + J : 1 + 2 * J]

    norm_scale = np.sqrt(J) if J > 0 else 1.0
    distance_metrics = {
        'gamma_error': float(theta_hat_np[0] - gamma_base),
        'V_l2': float(np.linalg.norm(V_hat - V_base) / norm_scale),
        'V_max_abs': float(np.max(np.abs(V_hat - V_base))) if J > 0 else 0.0,
        'c_l2': float(np.linalg.norm(C_hat - C_base) / norm_scale),
        'c_max_abs': float(np.max(np.abs(C_hat - C_base))) if J > 0 else 0.0,
    }

    true_vector = np.concatenate(([gamma_base], V_base, C_base))

    if args.skip_plot:
        theta_plot_path = None
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
        ci_V = np.where(np.isfinite(ci_radius[1 : 1 + J]), ci_radius[1 : 1 + J], 0.0)
        axes[1].errorbar(
            v_indices,
            V_hat,
            yerr=ci_V,
            fmt='o',
            capsize=4,
            label='Estimate ±95% CI',
        )
        axes[1].scatter(v_indices, true_vector[1 : 1 + J], marker='x', color='red', label='Baseline θ₀')
        axes[1].set_xticks(v_indices)
        axes[1].set_xticklabels([f"V_{idx+1}" for idx in range(J)], rotation=45, ha='right')
        axes[1].set_ylabel('V')
        axes[1].set_title('Firm V estimates')
        axes[1].legend()

        c_indices = np.arange(J)
        ci_C = np.where(np.isfinite(ci_radius[1 + J :]), ci_radius[1 + J :], 0.0)
        axes[2].errorbar(
            c_indices,
            C_hat,
            yerr=ci_C,
            fmt='o',
            capsize=4,
            label='Estimate ±95% CI',
        )
        axes[2].scatter(c_indices, true_vector[1 + J :], marker='x', color='red', label='Baseline θ₀')
        axes[2].set_xticks(c_indices)
        axes[2].set_xticklabels([f"c_{idx+1}" for idx in range(J)], rotation=45, ha='right')
        axes[2].set_ylabel('c')
        axes[2].set_title('Firm cutoff estimates')
        axes[2].legend()

        theta_plot_path = os.path.join(args.out_dir, 'mle_theta_ci_plot.png')
        fig.savefig(theta_plot_path, dpi=200)
        plt.close(fig)

    out = {
        'solver': 'LBFGS',
        'objective': obj,
        'nit': nit,
        'grad_norm': grad_norm,
        'theta0': theta0_np.tolist(),
        'theta_hat': theta_hat.tolist(),
        'time_sec': total_time,
        'timings': {
            'build_time_sec': build_time,
            'solve_time_sec': solve_time,
            'total_time_sec': total_time,
        },
        'distance_metrics': distance_metrics,
        'ci_radius_95': ci_radius.tolist(),
        'param_names': param_names,
        'robust_se': se_robust.tolist(),
        'cov_robust': cov_robust.tolist() if cov_robust is not None else None,
        'market_shares': {
            'empirical': shares_empirical.tolist(),
            'model': shares_model.tolist(),
        },
    }

    if theta_plot_path is not None:
        out['theta_plot_path'] = theta_plot_path

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'mle_gamma_Vc_estimates_jax.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
