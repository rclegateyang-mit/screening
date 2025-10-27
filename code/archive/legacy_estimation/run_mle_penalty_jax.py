#!/usr/bin/env python3
"""Penalized maximum-likelihood estimation of (gamma, beta, V, c) using JAX."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxopt import LBFGS

os.environ.setdefault("JAX_ENABLE_X64", "1")

try:
    from .jax_model import enable_x64, compute_penalty_components_jax
    from .helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
        load_weight_matrix,
    )
    from .optimize_gmm import make_reparam
except ImportError:  # pragma: no cover
    from jax_model import enable_x64, compute_penalty_components_jax
    from helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
        load_weight_matrix,
    )
    from optimize_gmm import make_reparam


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent.parent
    output_dir = root / "output"

    parser = argparse.ArgumentParser(
        description="Run penalized JAX MLE for (gamma, beta, V, c)"
    )
    parser.add_argument("--workers_path", type=str, default=str(output_dir / "workers_dataset.csv"))
    parser.add_argument("--firms_path", type=str, default=str(output_dir / "equilibrium_firms.csv"))
    parser.add_argument("--params_path", type=str, default=str(output_dir / "parameters_effective.csv"))
    parser.add_argument("--weight_matrix_path", type=str, default=None,
                        help="Path to JxJ weight matrix (csv/json/npy). Defaults to identity.")
    parser.add_argument("--out_dir", type=str, default=str(output_dir))
    parser.add_argument("--theta0_file", type=str, default=None)
    parser.add_argument("--theta0_list", type=str, default=None,
                        help="Comma-separated γ, β, V(1..J), c(1..J); if omitted use baseline from data.")
    parser.add_argument("--theta0_from_helper", action="store_true",
                        help="Initialise θ0 via helpers.naive_theta_guess_gamma_V_c using data in firms/workers CSVs.")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--skip_statistics", action="store_true",
                        help="Skip covariance computation.")
    parser.add_argument("--skip_plot", action="store_true",
                        help="Skip plotting.")
    return parser.parse_args()


def parse_theta0_extended(
    args: argparse.Namespace,
    J: int,
    V0_nat: np.ndarray,
    C0_nat: np.ndarray,
    gamma0: float,
    beta0: float,
) -> np.ndarray:
    k_expected = 2 + 2 * J

    if args.theta0_list is not None:
        raw = args.theta0_list.replace("\n", " ").replace("\t", " ")
        values: list[float] = []
        for part in raw.split(","):
            values.extend([float(x.strip()) for x in part.split() if x.strip()])
        theta0 = np.asarray(values, dtype=float)
        if theta0.size != k_expected:
            raise ValueError(f"θ0 length {theta0.size} != 2+2J={k_expected}.")
        return theta0

    if args.theta0_file is not None:
        file_path = args.theta0_file
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                data = json.load(f)
            theta0 = np.asarray(data["theta"], dtype=float)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            theta0 = df.iloc[0].values.astype(float)
        else:
            raise ValueError(f"Unsupported theta0 file format: {file_path}")
        if theta0.size != k_expected:
            raise ValueError(f"θ0 length {theta0.size} != 2+2J={k_expected}.")
        return theta0

    return np.concatenate(([gamma0], [beta0], V0_nat, C0_nat))


def main() -> None:
    args = parse_args()

    build_start = time.perf_counter()

    if args.threads is not None:
        print("Set CPU threads by exporting before Python:")
        print(
            f'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_cpu_thread_pool_size={int(args.threads)}"'
        )

    enable_x64()

    params = read_parameters(args.params_path)
    firm_ids, w, Y, _, xi, loc_firms, c_data = read_firms_data(args.firms_path)
    x_skill, ell_x, ell_y, chosen_firm = read_workers_data(args.workers_path)

    N = x_skill.size
    J = firm_ids.size

    alpha = params.get("alpha", 1.0)
    phi_param = params.get("varphi", params.get("phi", 1.0))
    mu_a_param = params.get("mu_a", 0.0)
    sigma_a_param = params.get("sigma_a", 0.12)
    gamma0_param = params.get("gamma", 0.05)
    beta0_param = params.get("beta", 0.5)

    weight_matrix_np = load_weight_matrix(args.weight_matrix_path, J)

    V0_nat = alpha * np.log(np.maximum(w, 1e-300)) + xi
    C0_nat = np.maximum(c_data, 1e-10)

    if args.theta0_from_helper:
        try:
            from .helpers import naive_theta_guess_gamma_V_c  # type: ignore
        except ImportError:  # pragma: no cover
            from helpers import naive_theta_guess_gamma_V_c  # type: ignore

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
        theta0_np = np.concatenate(([theta0_np[0]], [beta0_param], theta0_np[1:]))
    else:
        theta0_np = parse_theta0_extended(args, J, V0_nat, C0_nat, gamma0_param, beta0_param)
        theta0_np = np.asarray(theta0_np, dtype=float)

    if theta0_np.size != 2 + 2 * J:
        raise ValueError(f"Expected θ0 of length {2 + 2 * J}, got {theta0_np.size}.")

    D_nat = compute_worker_firm_distances(ell_x, ell_y, loc_firms)

    aux: Dict[str, jnp.ndarray] = {
        "D_nat": jnp.asarray(D_nat, dtype=jnp.float64),
        "phi": jnp.asarray(float(phi_param), dtype=jnp.float64),
        "mu_a": jnp.asarray(float(mu_a_param), dtype=jnp.float64),
        "sigma_a": jnp.asarray(float(sigma_a_param), dtype=jnp.float64),
        "firm_ids": jnp.asarray(firm_ids, dtype=jnp.int32),
    }
    X = jnp.asarray(x_skill, dtype=jnp.float64)
    choice_idx = jnp.asarray(chosen_firm.astype(np.int32))
    w_nat = jnp.asarray(w, dtype=jnp.float64)
    Y_nat = jnp.asarray(Y, dtype=jnp.float64)
    weight_matrix = jnp.asarray(weight_matrix_np, dtype=jnp.float64)

    counts_full = np.bincount(chosen_firm, minlength=J + 1).astype(float)
    L_data_np = counts_full[1:]
    L_data_jax = jnp.asarray(L_data_np, dtype=jnp.float64)

    lb = jnp.full((theta0_np.size,), -jnp.inf)
    ub = jnp.full((theta0_np.size,), jnp.inf)
    lb = lb.at[0].set(0.0)
    ub = ub.at[0].set(1.0)
    lb = lb.at[1].set(1e-6)
    ub = ub.at[1].set(1.0 - 1e-6)
    for idx in range(J):
        lb = lb.at[2 + J + idx].set(1e-8)
        ub = ub.at[2 + J + idx].set(jnp.inf)

    fwd, inv = make_reparam(lb, ub)
    theta0 = jnp.asarray(theta0_np, dtype=jnp.float64)
    z0 = inv(theta0)

    def base_components(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return compute_penalty_components_jax(
            theta,
            X,
            choice_idx,
            aux,
            w_nat,
            Y_nat,
            L_data_jax,
        )

    components_fn = jax.jit(base_components)

    def penalized_objective_theta(theta: jnp.ndarray) -> jnp.ndarray:
        _, per_obs_nll, m_vec, _, _ = components_fn(theta)
        nll = jnp.sum(per_obs_nll)
        penalty = 0.5 * (m_vec @ (weight_matrix @ m_vec))
        return nll + penalty

    penalized_objective_theta = jax.jit(penalized_objective_theta)

    def penalized_objective_z(z: jnp.ndarray) -> jnp.ndarray:
        theta = fwd(z)
        return penalized_objective_theta(theta)

    penalized_objective_z = jax.jit(penalized_objective_z)

    solver = LBFGS(
        fun=penalized_objective_z,
        value_and_grad=False,
        maxiter=int(args.maxiter),
        tol=args.tol,
    )

    build_time = time.perf_counter() - build_start

    solve_start = time.perf_counter()
    res = solver.run(z0)
    solve_time = time.perf_counter() - solve_start
    total_time = build_time + solve_time

    z_hat = res.params
    theta_hat = np.asarray(fwd(z_hat))
    theta_hat_jax = jnp.asarray(theta_hat, dtype=jnp.float64)

    P_hat, per_obs_nll_hat, m_hat, L_hat, S_hat = components_fn(theta_hat_jax)
    nll_hat = float(jnp.sum(per_obs_nll_hat))
    penalty_hat = float(0.5 * m_hat @ (weight_matrix @ m_hat))
    obj = float(penalized_objective_theta(theta_hat_jax))
    nit = int(res.state.iter_num)
    grad_norm = float(jnp.linalg.norm(res.state.grad))

    print(
        f"[LBFGS] objective={obj:.6f} (nll={nll_hat:.6f}, penalty={penalty_hat:.6f}), "
        f"nit={nit}, grad_norm={grad_norm:.3e}, build={build_time:.2f}s, "
        f"solve={solve_time:.2f}s, total={total_time:.2f}s"
    )

    shares_empirical = np.bincount(chosen_firm, minlength=J + 1) / max(N, 1)
    shares_model = np.asarray(P_hat).mean(axis=0)

    print("Market shares (empirical vs model):")
    for idx in range(J + 1):
        name = "outside" if idx == 0 else f"firm_{idx}"
        print(f"  {name}: {shares_empirical[idx]:.6f} vs {shares_model[idx]:.6f}")

    m_hat_np = np.asarray(m_hat)
    L_data_out_np = np.asarray(L_hat)
    S_hat_np = np.asarray(S_hat)
    labor_model_np = np.sum(np.asarray(P_hat)[:, 1:], axis=0)

    param_names = ["gamma", "beta"] + [f"V_{j+1}" for j in range(J)] + [f"c_{j+1}" for j in range(J)]
    se_penalized = np.full(len(param_names), np.nan, dtype=float)
    cov_penalized = None
    if args.skip_statistics:
        print("Skipping covariance computation (--skip_statistics).")
        ci_radius = np.full(len(param_names), np.nan, dtype=float)
    else:
        try:
            hessian = jax.hessian(penalized_objective_theta)(theta_hat_jax)
            hessian_np = np.asarray(hessian, dtype=float)
            hessian_np = 0.5 * (hessian_np + hessian_np.T)

            try:
                cov_penalized = np.linalg.inv(hessian_np)
            except np.linalg.LinAlgError:
                cov_penalized = np.linalg.pinv(hessian_np)

            cov_penalized = 0.5 * (cov_penalized + cov_penalized.T)
            se_penalized = np.sqrt(np.maximum(np.diag(cov_penalized), 0.0))
            print("Square-root diagonal of Hessian inverse (penalized SE proxy):")
            for name, se in zip(param_names, se_penalized):
                print(f"  {name}: {se:.6f}")
        except (MemoryError, RuntimeError, ValueError) as exc:
            print(f"Warning: skipping covariance computation due to {exc}")
            cov_penalized = None
            se_penalized = np.full(len(param_names), np.nan, dtype=float)
        finally:
            ci_radius = 1.96 * se_penalized

    theta_hat_np = np.asarray(theta_hat, dtype=float)

    gamma_base = float(gamma0_param)
    beta_base = float(beta0_param)
    V_base = V0_nat.astype(float)
    C_base = C0_nat.astype(float)
    gamma_hat = theta_hat_np[0]
    beta_hat = theta_hat_np[1]
    V_hat = theta_hat_np[2 : 2 + J]
    C_hat = theta_hat_np[2 + J :]

    norm_scale = np.sqrt(J) if J > 0 else 1.0
    distance_metrics = {
        "gamma_error": float(gamma_hat - gamma_base),
        "beta_error": float(beta_hat - beta_base),
        "V_l2": float(np.linalg.norm(V_hat - V_base) / norm_scale),
        "V_max_abs": float(np.max(np.abs(V_hat - V_base))) if J > 0 else 0.0,
        "c_l2": float(np.linalg.norm(C_hat - C_base) / norm_scale),
        "c_max_abs": float(np.max(np.abs(C_hat - C_base))) if J > 0 else 0.0,
    }

    true_vector = np.concatenate(([gamma_base], [beta_base], V_base, C_base))
    true_params = {
        "gamma": gamma_base,
        "beta": beta_base,
        "V": V_base.tolist(),
        "c": C_base.tolist(),
    }

    if args.skip_plot:
        theta_plot_path = None
        print("Skipping plot generation (--skip_plot).")
    else:
        fig, axes = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)

        ci_gamma = ci_radius[0] if np.isfinite(ci_radius[0]) else 0.0
        axes[0].errorbar(
            [0],
            [gamma_hat],
            yerr=[ci_gamma],
            fmt="o",
            capsize=4,
            label="Estimate ±95% CI",
        )
        axes[0].scatter([0], [true_vector[0]], marker="x", color="red", label="Baseline θ₀")
        axes[0].set_xticks([0])
        axes[0].set_xticklabels(["gamma"])
        axes[0].set_ylabel("γ")
        axes[0].set_title("Gamma estimate")
        axes[0].legend()

        ci_beta = ci_radius[1] if np.isfinite(ci_radius[1]) else 0.0
        axes[1].errorbar(
            [0],
            [beta_hat],
            yerr=[ci_beta],
            fmt="o",
            capsize=4,
            label="Estimate ±95% CI",
        )
        axes[1].scatter([0], [true_vector[1]], marker="x", color="red", label="Baseline θ₀")
        axes[1].set_xticks([0])
        axes[1].set_xticklabels(["beta"])
        axes[1].set_ylabel("β")
        axes[1].set_title("Beta estimate")
        axes[1].legend()

        v_indices = np.arange(J)
        ci_V = np.where(np.isfinite(ci_radius[2 : 2 + J]), ci_radius[2 : 2 + J], 0.0)
        axes[2].errorbar(
            v_indices,
            V_hat,
            yerr=ci_V,
            fmt="o",
            capsize=4,
            label="Estimate ±95% CI",
        )
        axes[2].scatter(
            v_indices,
            true_vector[2 : 2 + J],
            marker="x",
            color="red",
            label="Baseline θ₀",
        )
        axes[2].set_xticks(v_indices)
        axes[2].set_xticklabels([f"V_{idx + 1}" for idx in range(J)], rotation=45, ha="right")
        axes[2].set_ylabel("V")
        axes[2].set_title("Firm V estimates")
        axes[2].legend()

        ci_C = np.where(np.isfinite(ci_radius[2 + J :]), ci_radius[2 + J :], 0.0)
        c_indices = np.arange(J)
        axes[3].errorbar(
            c_indices,
            C_hat,
            yerr=ci_C,
            fmt="o",
            capsize=4,
            label="Estimate ±95% CI",
        )
        axes[3].scatter(
            c_indices,
            true_vector[2 + J :],
            marker="x",
            color="red",
            label="Baseline θ₀",
        )
        axes[3].set_xticks(c_indices)
        axes[3].set_xticklabels([f"c_{idx + 1}" for idx in range(J)], rotation=45, ha="right")
        axes[3].set_ylabel("c")
        axes[3].set_title("Firm cutoff estimates")
        axes[3].legend()

        theta_plot_path = os.path.join(args.out_dir, "mle_penalized_theta_ci_plot.png")
        fig.savefig(theta_plot_path, dpi=200)
        plt.close(fig)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "mle_gamma_beta_Vc_penalty_estimates_jax.json")

    out = {
        "solver": "LBFGS",
        "objective": obj,
        "objective_breakdown": {
            "neg_log_likelihood": nll_hat,
            "penalty": penalty_hat,
        },
        "nit": nit,
        "grad_norm": grad_norm,
        "theta0": theta0_np.tolist(),
        "theta_hat": theta_hat.tolist(),
        "moment_vector": m_hat_np.tolist(),
        "labor_supplied_data": L_data_out_np.tolist(),
        "labor_supplied_model": labor_model_np.tolist(),
        "average_skill": S_hat_np.tolist(),
        "weight_matrix": weight_matrix_np.tolist(),
        "time_sec": total_time,
        "timings": {
            "build_time_sec": build_time,
            "solve_time_sec": solve_time,
            "total_time_sec": total_time,
        },
        "distance_metrics": distance_metrics,
        "ci_radius_95": ci_radius.tolist(),
        "param_names": param_names,
        "penalized_se_proxy": se_penalized.tolist(),
        "cov_penalized": cov_penalized.tolist() if cov_penalized is not None else None,
        "true_params": true_params,
        "market_shares": {
            "empirical": shares_empirical.tolist(),
            "model": shares_model.tolist(),
        },
    }

    if theta_plot_path is not None:
        out["theta_plot_path"] = theta_plot_path

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
