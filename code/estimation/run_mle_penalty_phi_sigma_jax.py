#!/usr/bin/env python3
"""Penalized MLE for (γ, β, φ, σ_skill, λ_skill, V, c) with JAX.

This script focuses on a single, up-to-date estimator that augments the baseline
MLE with a quadratic penalty on labor-market moments. The implementation is
structured to minimise dependencies and maximise throughput by:

- loading data once and moving everything to JAX device memory,
- using an efficient custom reparameterisation tailored to the parameter bounds,
- compiling the objective and its gradient exactly once via ``jax.jit``,
- using ``jaxopt``'s LBFGS solver with user-provided gradients.

The resulting code is the reference implementation for the project.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import nn as jnn
from jaxopt import LBFGS

try:
    from .. import get_data_dir, get_output_dir
    from .helpers import (
        compute_worker_firm_distances,
        load_weight_matrix,
        naive_theta_guess_gamma_V_c,
        read_firms_data,
        read_parameters,
        read_workers_data,
    )
    from .jax_model import (
        compute_penalty_components_with_params_jax,
        enable_x64,
    )
except ImportError:  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from code import get_data_dir, get_output_dir  # type: ignore
    from helpers import (
        compute_worker_firm_distances,  # type: ignore
        load_weight_matrix,  # type: ignore
        naive_theta_guess_gamma_V_c,  # type: ignore
        read_firms_data,  # type: ignore
        read_parameters,  # type: ignore
        read_workers_data,  # type: ignore
    )
    from jax_model import (  # type: ignore
        compute_penalty_components_with_params_jax,
        enable_x64,
    )


# ---------------------------------------------------------------------------
# Dataclasses and small utilities
# ---------------------------------------------------------------------------

EPS_GAMMA = 1e-6
EPS_BETA = 1e-6
POS_FLOOR = 1e-6
C_FLOOR = 1e-8


@dataclass(frozen=True)
class Baseline:
    gamma: float
    beta: float
    phi: float
    sigma_a: float
    sigma_skill: float
    lambda_skill: float
    V: np.ndarray
    C: np.ndarray

    def vector(self) -> np.ndarray:
        head = np.array(
            [self.gamma, self.beta, self.phi, self.sigma_skill, self.lambda_skill],
            dtype=float,
        )
        return np.concatenate((head, self.V, self.C))


@dataclass(frozen=True)
class ModelData:
    X: jnp.ndarray
    choice_idx: jnp.ndarray
    aux_template: Dict[str, jnp.ndarray]
    w_nat: jnp.ndarray
    Y_nat: jnp.ndarray
    labor_counts: jnp.ndarray
    weight_matrix: jnp.ndarray
    N: int
    J: int
    firm_ids: np.ndarray
    shares_empirical: np.ndarray


@dataclass(frozen=True)
class ParameterTransform:
    fwd: Any
    inv: Any


def _logit(y: jnp.ndarray) -> jnp.ndarray:
    y = jnp.clip(y, 1e-12, 1.0 - 1e-12)
    return jnp.log(y) - jnp.log1p(-y)


def _softplus_inv(y: jnp.ndarray) -> jnp.ndarray:
    y = jnp.maximum(y, 1e-12)
    return jnp.where(y < 20.0, jnp.log(jnp.expm1(y)), y + jnp.log1p(-jnp.exp(-y)))


def make_transform(J: int) -> ParameterTransform:
    """Return (forward, inverse) transforms between ℝ^K and bounded θ."""

    def fwd(z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=jnp.float64)
        gamma = EPS_GAMMA + (1.0 - 2.0 * EPS_GAMMA) * jnn.sigmoid(z[0])
        beta = EPS_BETA + (1.0 - 2.0 * EPS_BETA) * jnn.sigmoid(z[1])
        phi = POS_FLOOR + jnn.softplus(z[2])
        sigma_skill = POS_FLOOR + jnn.softplus(z[3])
        lambda_skill = z[4]
        V = z[5 : 5 + J]
        c = C_FLOOR + jnn.softplus(z[5 + J :])
        return jnp.concatenate(
            (
                jnp.array(
                    [gamma, beta, phi, sigma_skill, lambda_skill], dtype=jnp.float64
                ),
                V,
                c,
            )
        )

    def inv(theta: jnp.ndarray) -> jnp.ndarray:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        gamma = theta[0]
        beta = theta[1]
        phi = theta[2]
        sigma_skill = theta[3]
        lambda_skill = theta[4]
        V = theta[5 : 5 + J]
        c = theta[5 + J :]

        gamma_scaled = (gamma - EPS_GAMMA) / (1.0 - 2.0 * EPS_GAMMA)
        beta_scaled = (beta - EPS_BETA) / (1.0 - 2.0 * EPS_BETA)

        gamma_z = _logit(jnp.clip(gamma_scaled, 1e-12, 1.0 - 1e-12))
        beta_z = _logit(jnp.clip(beta_scaled, 1e-12, 1.0 - 1e-12))
        phi_z = _softplus_inv(jnp.maximum(phi - POS_FLOOR, 1e-12))
        sigma_skill_z = _softplus_inv(jnp.maximum(sigma_skill - POS_FLOOR, 1e-12))
        c_z = _softplus_inv(jnp.maximum(c - C_FLOOR, 1e-12))

        return jnp.concatenate(
            (
                jnp.array(
                    [gamma_z, beta_z, phi_z, sigma_skill_z, lambda_skill],
                    dtype=jnp.float64,
                ),
                V,
                c_z,
            )
        )

    return ParameterTransform(fwd=fwd, inv=inv)


def transform_theta_and_jacobian(theta: np.ndarray, J: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply affine transformation to θ and return (θ̃, Jac)."""

    theta = np.asarray(theta, dtype=float).reshape(-1)
    K = theta.size
    if K != 5 + 2 * J:
        raise ValueError(f"Expected θ size {5 + 2 * J}, got {K}.")

    gamma, beta, phi, sigma_skill, lambda_skill = theta[:5]
    V = theta[5 : 5 + J]
    c = theta[5 + J :]

    inv_sigma = 1.0 / sigma_skill
    phi_tilde = phi * inv_sigma
    c_tilde = (c - lambda_skill) * inv_sigma

    theta_tilde = theta.copy()
    theta_tilde[2] = phi_tilde
    theta_tilde[5 + J :] = c_tilde

    jac = np.eye(K, dtype=float)
    jac[2, :] = 0.0
    jac[2, 2] = inv_sigma
    jac[2, 3] = -phi * inv_sigma**2

    for j in range(J):
        row = 5 + J + j
        jac[row, :] = 0.0
        jac[row, 3] = -(c[j] - lambda_skill) * inv_sigma**2
        jac[row, 4] = -inv_sigma
        jac[row, 5 + J + j] = inv_sigma

    return theta_tilde, jac


# ---------------------------------------------------------------------------
# CLI parsing and data preparation
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    data_dir = get_data_dir(create=True)
    output_dir = get_output_dir(create=True)

    parser = argparse.ArgumentParser(
        description="Penalized JAX MLE for (γ, β, φ, σₐ, V, c)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workers_path", type=str, default=str(data_dir / "workers_dataset.csv"))
    parser.add_argument("--firms_path", type=str, default=str(data_dir / "equilibrium_firms.csv"))
    parser.add_argument("--params_path", type=str, default=str(data_dir / "parameters_effective.csv"))
    parser.add_argument("--weight_matrix_path", type=str, default=None,
                        help="Path to J×J weight matrix (csv/json/npy); defaults to identity.")
    parser.add_argument("--out_dir", type=str, default=str(output_dir))
    parser.add_argument("--theta0_file", type=str, default=None)
    parser.add_argument("--theta0_list", type=str, default=None,
                        help="Comma-separated γ, β, φ, σₐ, V(1..J), c(1..J).")
    parser.add_argument("--theta0_from_helper", action="store_true",
                        help="Initialise γ,V,c via helpers.naive_theta_guess_gamma_V_c and append β,φ,σₐ baselines.")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--skip_statistics", action="store_true",
                        help="Skip Hessian-based SE approximation (faster, no covariance output).")
    parser.add_argument("--skip_plot", action="store_true",
                        help="Skip diagnostic plot generation.")
    return parser.parse_args()


def _parse_theta_list(raw: str, expected: int) -> np.ndarray:
    raw = raw.replace("\n", " ").replace("\t", " ")
    values: list[float] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.extend(float(x) for x in stripped.split())
    theta = np.asarray(values, dtype=float)
    if theta.size != expected:
        raise ValueError(f"θ₀ length {theta.size} != expected {expected}.")
    return theta


def _load_theta_from_file(path: str, expected: int) -> np.ndarray:
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        theta = np.asarray(data["theta"], dtype=float)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        theta = df.iloc[0].values.astype(float)
    else:
        raise ValueError(f"Unsupported theta0 file format: {path}")
    if theta.size != expected:
        raise ValueError(f"θ₀ length {theta.size} != expected {expected}.")
    return theta


def _initial_theta(
    args: argparse.Namespace,
    J: int,
    V0_nat: np.ndarray,
    C0_nat: np.ndarray,
    gamma0: float,
    beta_init: float,
    phi_init: float,
    sigma_skill_init: float,
    lambda_skill_init: float,
    helper_kwargs: Dict[str, Any],
) -> np.ndarray:
    expected = 5 + 2 * J

    if args.theta0_list is not None:
        return _parse_theta_list(args.theta0_list, expected)

    if args.theta0_file is not None:
        return _load_theta_from_file(args.theta0_file, expected)

    if args.theta0_from_helper:
        print("Initialising γ,V,c via helpers.naive_theta_guess_gamma_V_c ...")
        theta_partial = naive_theta_guess_gamma_V_c(**helper_kwargs)
        theta_partial = np.asarray(theta_partial, dtype=float)
        return np.concatenate(
            (
                theta_partial[:1],
                [beta_init, phi_init, sigma_skill_init, lambda_skill_init],
                theta_partial[1:],
            )
        )

    return np.concatenate(
        (
            [gamma0, beta_init, phi_init, sigma_skill_init, lambda_skill_init],
            V0_nat,
            C0_nat,
        )
    )


def _enforce_feasible(theta: np.ndarray, J: int) -> np.ndarray:
    theta = theta.astype(float, copy=True)
    theta[0] = np.clip(theta[0], EPS_GAMMA + 1e-12, 1.0 - EPS_GAMMA - 1e-12)
    theta[1] = np.clip(theta[1], EPS_BETA + 1e-12, 1.0 - EPS_BETA - 1e-12)
    theta[2] = max(theta[2], POS_FLOOR + 1e-9)
    theta[3] = max(theta[3], POS_FLOOR + 1e-9)
    theta[5 + J :] = np.maximum(theta[5 + J :], C_FLOOR + 1e-9)
    return theta


def prepare_problem(args: argparse.Namespace) -> Tuple[ModelData, Baseline, jnp.ndarray, jnp.ndarray, ParameterTransform]:
    params_path = Path(args.params_path)
    firms_path = Path(args.firms_path)
    workers_path = Path(args.workers_path)

    params = read_parameters(str(params_path))
    firm_ids, w, Y, _, xi, loc_firms, c_data = read_firms_data(str(firms_path))
    x_skill, ell_x, ell_y, chosen_firm = read_workers_data(str(workers_path))

    N = x_skill.size
    J = firm_ids.size

    alpha = params.get("alpha", 1.0)
    phi_baseline = params.get("varphi", params.get("phi", 1.0))
    mu_a = params.get("mu_a", 0.0)
    sigma_baseline = params.get("sigma_a", 0.12)
    gamma_baseline = params.get("gamma", 0.05)
    beta_baseline = params.get("beta", 0.5)

    beta_init = 0.3
    phi_init = 1.3
    sigma_skill_init = 1.0
    lambda_skill_init = 0.0

    V0_nat = alpha * np.log(np.maximum(w, 1e-300)) + xi
    C0_nat = np.maximum(c_data, 1e-10)

    helper_kwargs = dict(
        x_skill=x_skill,
        ell_x=ell_x,
        ell_y=ell_y,
        chosen_firm=chosen_firm,
        loc_firms=loc_firms,
        firm_ids=firm_ids,
        gamma0=gamma_baseline,
        firms_csv_path=str(firms_path),
    )

    theta0_np = _initial_theta(
        args,
        J,
        V0_nat,
        C0_nat,
        gamma_baseline,
        beta_init,
        phi_init,
        sigma_skill_init,
        lambda_skill_init,
        helper_kwargs,
    )
    theta0_np = _enforce_feasible(theta0_np, J)

    D_nat = compute_worker_firm_distances(ell_x, ell_y, loc_firms)
    weight_matrix_np = load_weight_matrix(args.weight_matrix_path, 4)

    counts_full = np.bincount(chosen_firm, minlength=J + 1).astype(float)
    shares_empirical = counts_full / max(N, 1)
    labor_counts = counts_full[1:]

    model_data = ModelData(
        X=jnp.asarray(x_skill, dtype=jnp.float64),
        choice_idx=jnp.asarray(chosen_firm.astype(np.int32)),
        aux_template={
            "D_nat": jnp.asarray(D_nat, dtype=jnp.float64),
            "mu_a": jnp.asarray(float(mu_a), dtype=jnp.float64),
            "firm_ids": jnp.asarray(firm_ids, dtype=jnp.int32),
        },
        w_nat=jnp.asarray(w, dtype=jnp.float64),
        Y_nat=jnp.asarray(Y, dtype=jnp.float64),
        labor_counts=jnp.asarray(labor_counts, dtype=jnp.float64),
        weight_matrix=jnp.asarray(weight_matrix_np, dtype=jnp.float64),
        N=N,
        J=J,
        firm_ids=firm_ids,
        shares_empirical=shares_empirical,
    )

    baseline = Baseline(
        gamma=float(gamma_baseline),
        beta=float(beta_baseline),
        phi=float(phi_baseline),
        sigma_a=float(sigma_baseline),
        sigma_skill=float(sigma_skill_init),
        lambda_skill=float(lambda_skill_init),
        V=V0_nat.astype(float),
        C=C0_nat.astype(float),
    )

    transform = make_transform(J)
    theta0 = jnp.asarray(theta0_np, dtype=jnp.float64)
    z0 = transform.inv(theta0)

    return model_data, baseline, theta0, z0, transform


# ---------------------------------------------------------------------------
# Model evaluation and optimisation
# ---------------------------------------------------------------------------


def make_components_fn(data: ModelData):
    """Return a JIT-compiled function computing model components."""

    J = data.J

    @jax.jit
    def components(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        phi = theta[2]
        sigma_skill = theta[3]
        lambda_skill = theta[4]
        theta_core = jnp.concatenate((theta[:2], theta[5:]))
        sigma_a_fixed = jnp.array(1.0, dtype=theta.dtype)
        P_nat, per_obs_nll, _, L_data, S_nat = compute_penalty_components_with_params_jax(
            theta_core,
            data.X,
            data.choice_idx,
            data.aux_template,
            data.w_nat,
            data.Y_nat,
            data.labor_counts,
            phi,
            sigma_a_fixed,
        )

        beta = theta[1]
        c_nat = theta[5 + J :]

        safe_Y = jnp.maximum(data.Y_nat, 1e-12)
        safe_L = jnp.maximum(L_data, 1e-12)
        safe_w = jnp.maximum(data.w_nat, 1e-12)
        safe_sigma_skill = jnp.maximum(sigma_skill, 1e-6)
        safe_S_shift = jnp.maximum(S_nat - lambda_skill, 1e-12)

        scale = data.Y_nat / (safe_L * safe_w)
        beta_term = (1.0 - beta) * scale
        adjustment = S_nat - lambda_skill - beta_term * c_nat + lambda_skill * beta_term

        m1 = (
            jnp.log(safe_Y)
            - (1.0 - beta) * jnp.log(safe_L)
            - ((1.0 - beta) / safe_sigma_skill) * jnp.log(safe_S_shift)
        )
        m2 = adjustment
        m3 = scale * c_nat * adjustment
        m4 = scale * adjustment

        m_total = jnp.stack(
            (
                jnp.sum(m1),
                jnp.sum(m2),
                jnp.sum(m3),
                jnp.sum(m4),
            )
        )

        return P_nat, per_obs_nll, m_total, L_data, S_nat

    return components


def make_value_fn(data: ModelData, components_fn):
    """Return a JIT-compiled penalized objective in θ-space."""

    @jax.jit
    def value(theta: jnp.ndarray) -> jnp.ndarray:
        _, per_obs_nll, m_total, _, _ = components_fn(theta)
        nll = jnp.sum(per_obs_nll)
        penalty = 0.5 * m_total @ (data.weight_matrix @ m_total)
        return nll + penalty

    return value


def make_value_and_grad_z(value_fn, transform: ParameterTransform):
    """Return JIT-compiled (value, grad) function in unconstrained z-space."""

    def loss_z(z: jnp.ndarray) -> jnp.ndarray:
        theta = transform.fwd(z)
        return value_fn(theta)

    return jax.jit(jax.value_and_grad(loss_z))


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------


def distance_metrics(theta_hat: np.ndarray, baseline: Baseline) -> Dict[str, float]:
    J = baseline.V.size
    norm_scale = np.sqrt(J) if J > 0 else 1.0
    return {
        "gamma_error": float(theta_hat[0] - baseline.gamma),
        "beta_error": float(theta_hat[1] - baseline.beta),
        "phi_error": float(theta_hat[2] - baseline.phi),
        "sigma_skill_error": float(theta_hat[3] - baseline.sigma_skill),
        "lambda_skill_error": float(theta_hat[4] - baseline.lambda_skill),
        "V_l2": float(np.linalg.norm(theta_hat[5 : 5 + J] - baseline.V) / norm_scale),
        "V_max_abs": float(np.max(np.abs(theta_hat[5 : 5 + J] - baseline.V))) if J > 0 else 0.0,
        "c_l2": float(np.linalg.norm(theta_hat[5 + J :] - baseline.C) / norm_scale),
        "c_max_abs": float(np.max(np.abs(theta_hat[5 + J :] - baseline.C))) if J > 0 else 0.0,
    }


def make_param_names_raw(J: int) -> list[str]:
    return (
        ["gamma", "beta", "phi", "sigma_skill", "lambda_skill"]
        + [f"V_{idx + 1}" for idx in range(J)]
        + [f"c_{idx + 1}" for idx in range(J)]
    )


def make_param_names_transformed(J: int) -> list[str]:
    return (
        ["gamma", "beta", "phi_over_sigma_skill", "sigma_skill", "lambda_skill"]
        + [f"V_{idx + 1}" for idx in range(J)]
        + [f"c_tilde_{idx + 1}" for idx in range(J)]
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.threads is not None:
        print("Set CPU threads by exporting before Python:")
        print(
            f'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_cpu_thread_pool_size={int(args.threads)}"'
        )

    enable_x64()

    build_start = time.perf_counter()
    data, baseline, theta0, z0, transform = prepare_problem(args)
    components_fn = make_components_fn(data)
    value_fn = make_value_fn(data, components_fn)
    value_and_grad_z = make_value_and_grad_z(value_fn, transform)
    build_time = time.perf_counter() - build_start

    solver = LBFGS(
        fun=value_and_grad_z,
        value_and_grad=True,
        maxiter=int(args.maxiter),
        tol=float(args.tol),
    )

    solve_start = time.perf_counter()
    res = solver.run(z0)
    jax.block_until_ready(res.state.value)
    solve_time = time.perf_counter() - solve_start
    total_time = build_time + solve_time

    z_hat = res.params
    theta_hat = transform.fwd(z_hat)
    theta_hat_np = np.asarray(theta_hat, dtype=float)
    theta_hat_tilde_np, jac_transform = transform_theta_and_jacobian(theta_hat_np, data.J)
    theta0_np_raw = np.asarray(theta0, dtype=float)
    theta0_tilde_np, _ = transform_theta_and_jacobian(theta0_np_raw, data.J)
    baseline_vec = baseline.vector()
    baseline_raw_np = np.asarray(baseline_vec, dtype=float)
    baseline_tilde_np, _ = transform_theta_and_jacobian(baseline_raw_np, data.J)

    obj = float(res.state.value)
    grad_norm = float(np.linalg.norm(np.asarray(res.state.grad)))
    nit = int(res.state.iter_num)

    P_hat, per_obs_nll, m_hat, L_hat, S_hat = components_fn(theta_hat)
    nll_hat = float(jnp.sum(per_obs_nll))
    penalty_hat = float(0.5 * m_hat @ (data.weight_matrix @ m_hat))

    shares_model = np.asarray(P_hat, dtype=float).mean(axis=0)
    m_hat_np = np.asarray(m_hat, dtype=float)
    L_hat_np = np.asarray(L_hat, dtype=float)
    S_hat_np = np.asarray(S_hat, dtype=float)
    labor_model_np = np.asarray(P_hat, dtype=float)[:, 1:].sum(axis=0)

    print(
        f"[LBFGS] objective={obj:.6f} (nll={nll_hat:.6f}, penalty={penalty_hat:.6f}), "
        f"nit={nit}, grad_norm={grad_norm:.3e}, build={build_time:.2f}s, "
        f"solve={solve_time:.2f}s, total={total_time:.2f}s"
    )

    print("Market shares (empirical vs model):")
    for idx in range(data.J + 1):
        name = "outside" if idx == 0 else f"firm_{idx}"
        print(
            f"  {name}: {data.shares_empirical[idx]:.6f} vs {shares_model[idx]:.6f}"
        )

    param_names_raw = make_param_names_raw(data.J)
    param_names_trans = make_param_names_transformed(data.J)

    hessian_full: np.ndarray | None = None
    jmicro_mat: np.ndarray | None = None
    jfirm_mat: np.ndarray | None = None
    cov_penalized_trans: np.ndarray | None = None

    if args.skip_statistics:
        cov_penalized = None
        cov_penalized_trans = None
        se_penalized = np.full(len(param_names_raw), np.nan, dtype=float)
        ci_radius = np.full(len(param_names_raw), np.nan, dtype=float)
        se_penalized_trans = np.full(len(param_names_trans), np.nan, dtype=float)
        ci_radius_trans = np.full(len(param_names_trans), np.nan, dtype=float)
        print("Skipping covariance computation (--skip_statistics).")
    else:
        try:
            hessian = jax.hessian(value_fn)(theta_hat)
            hessian_np = np.asarray(hessian, dtype=float)
            hessian_np = 0.5 * (hessian_np + hessian_np.T)

            H = hessian_np
            hessian_full = H

            def per_obs_loss(th: jnp.ndarray) -> jnp.ndarray:
                return components_fn(th)[1]

            score_nll = jax.jacobian(per_obs_loss)(theta_hat)
            score_nll_np = np.asarray(score_nll, dtype=float)
            
            score_matrix = score_nll_np
            Jmicro = score_matrix.T @ score_matrix
            jmicro_mat = Jmicro

            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H)

            def moments_only(th: jnp.ndarray) -> jnp.ndarray:
                return components_fn(th)[2]

            jac_m = jax.jacobian(moments_only)(theta_hat)
            jac_m_np = np.asarray(jac_m, dtype=float)  # (M, K) with M=4 moments
            weight_matrix_np = np.asarray(data.weight_matrix, dtype=float)
            g_firm = jac_m_np.T @ (weight_matrix_np @ m_hat_np)
            Jfirm = g_firm[:, None] @ g_firm[None, :]
            jfirm_mat = Jfirm

            Omega = Jmicro + Jfirm

            H_inv = 0.5 * (H_inv + H_inv.T)
            cov_penalized = H_inv @ Omega @ H_inv
            cov_penalized = 0.5 * (cov_penalized + cov_penalized.T)
            se_penalized = np.sqrt(np.maximum(np.diag(cov_penalized), 0.0))
            ci_radius = 1.96 * se_penalized

            cov_penalized_trans = jac_transform @ cov_penalized @ jac_transform.T
            cov_penalized_trans = 0.5 * (cov_penalized_trans + cov_penalized_trans.T)
            se_penalized_trans = np.sqrt(np.maximum(np.diag(cov_penalized_trans), 0.0))
            ci_radius_trans = 1.96 * se_penalized_trans

            print("Standard errors (raw θ) via H^{-1} Ω H^{-1}:")
            for name, se in zip(param_names_raw, se_penalized):
                print(f"  {name}: {se:.6f}")
            print("Standard errors (transformed θ̃):")
            for name, se in zip(param_names_trans, se_penalized_trans):
                print(f"  {name}: {se:.6f}")
        except (MemoryError, RuntimeError, ValueError) as exc:
            print(f"Warning: skipping covariance computation due to {exc}")
            cov_penalized = None
            cov_penalized_trans = None
            se_penalized = np.full(len(param_names_raw), np.nan, dtype=float)
            ci_radius = np.full(len(param_names_raw), np.nan, dtype=float)
            se_penalized_trans = np.full(len(param_names_trans), np.nan, dtype=float)
            ci_radius_trans = np.full(len(param_names_trans), np.nan, dtype=float)

    distance_dict = distance_metrics(theta_hat_np, baseline)

    if args.skip_plot:
        theta_plot_path = None
        print("Skipping plot generation (--skip_plot).")
    else:
        theta_plot_path = out_dir / "mle_penalized_phi_sigma_skill_theta_ci_plot.png"
        fig, axes = plt.subplots(7, 1, figsize=(10, 21), constrained_layout=True)

        def plot_scalar(ax, index: int, label: str, estimate: float, truth: float) -> None:
            radius = ci_radius_trans[index] if np.isfinite(ci_radius_trans[index]) else 0.0
            ax.errorbar(
                [0],
                [estimate],
                yerr=[radius],
                fmt="o",
                capsize=4,
                label="Estimate ±95% CI",
            )
            ax.scatter([0], [truth], marker="x", color="red", label="Baseline θ₀")
            ax.set_xticks([0])
            ax.set_xticklabels([label])
            ax.set_ylabel(label)
            ax.set_title(f"{label} estimate")
            ax.legend()

        plot_scalar(axes[0], 0, "gamma", theta_hat_tilde_np[0], baseline_tilde_np[0])
        plot_scalar(axes[1], 1, "beta", theta_hat_tilde_np[1], baseline_tilde_np[1])
        plot_scalar(axes[2], 2, "phi/σ_skill", theta_hat_tilde_np[2], baseline_tilde_np[2])
        plot_scalar(axes[3], 3, "sigma_skill", theta_hat_tilde_np[3], baseline_tilde_np[3])
        plot_scalar(axes[4], 4, "lambda_skill", theta_hat_tilde_np[4], baseline_tilde_np[4])

        v_indices = np.arange(data.J)
        ci_V = np.where(
            np.isfinite(ci_radius_trans[5 : 5 + data.J]),
            ci_radius_trans[5 : 5 + data.J],
            0.0,
        )
        axes[5].errorbar(
            v_indices,
            theta_hat_tilde_np[5 : 5 + data.J],
            yerr=ci_V,
            fmt="o",
            capsize=4,
            label="Estimate ±95% CI",
        )
        axes[5].scatter(
            v_indices,
            baseline.V,
            marker="x",
            color="red",
            label="Baseline θ₀",
        )
        axes[5].set_xticks(v_indices)
        axes[5].set_xticklabels([f"V_{idx + 1}" for idx in range(data.J)], rotation=45, ha="right")
        axes[5].set_ylabel("V")
        axes[5].set_title("Firm V estimates")
        axes[5].legend()

        ci_C = np.where(
            np.isfinite(ci_radius_trans[5 + data.J :]),
            ci_radius_trans[5 + data.J :],
            0.0,
        )
        c_indices = np.arange(data.J)
        axes[6].errorbar(
            c_indices,
            theta_hat_tilde_np[5 + data.J :],
            yerr=ci_C,
            fmt="o",
            capsize=4,
            label="Estimate ±95% CI",
        )
        axes[6].scatter(
            c_indices,
            baseline_tilde_np[5 + data.J :],
            marker="x",
            color="red",
            label="Baseline θ₀",
        )
        axes[6].set_xticks(c_indices)
        axes[6].set_xticklabels([f"c_{idx + 1}" for idx in range(data.J)], rotation=45, ha="right")
        axes[6].set_ylabel("c_tilde")
        axes[6].set_title("Firm cutoff estimates (tilde)")
        axes[6].legend()

        fig.savefig(theta_plot_path, dpi=200)
        plt.close(fig)

    out_path = out_dir / "mle_gamma_beta_phi_sigma_skill_lambda_skill_Vc_penalty_estimates_jax.json"

    out = {
        "solver": "LBFGS",
        "objective": obj,
        "objective_breakdown": {
            "neg_log_likelihood": nll_hat,
            "penalty": penalty_hat,
        },
        "nit": nit,
        "grad_norm": grad_norm,
        "theta0": theta0_np_raw.tolist(),
        "theta0_transformed": theta0_tilde_np.tolist(),
        "theta_hat": theta_hat_np.tolist(),
        "theta_hat_transformed": theta_hat_tilde_np.tolist(),
        "moment_vector": m_hat_np.tolist(),
        "labor_supplied_data": data.labor_counts.tolist(),
        "labor_supplied_model": labor_model_np.tolist(),
        "average_skill": S_hat_np.tolist(),
        "weight_matrix": np.asarray(data.weight_matrix, dtype=float).tolist(),
        "time_sec": total_time,
        "timings": {
            "build_time_sec": build_time,
            "solve_time_sec": solve_time,
            "total_time_sec": total_time,
        },
        "distance_metrics": distance_dict,
        "ci_radius_95": ci_radius.tolist(),
        "ci_radius_95_transformed": ci_radius_trans.tolist(),
        "param_names": param_names_raw,
        "param_names_transformed": param_names_trans,
        "standard_errors": se_penalized.tolist(),
        "standard_errors_transformed": se_penalized_trans.tolist(),
        "covariance": cov_penalized.tolist() if cov_penalized is not None else None,
        "covariance_transformed": (
            cov_penalized_trans.tolist() if cov_penalized_trans is not None else None
        ),
        "hessian": hessian_full.tolist() if hessian_full is not None else None,
        "J_micro": jmicro_mat.tolist() if jmicro_mat is not None else None,
        "J_firm": jfirm_mat.tolist() if jfirm_mat is not None else None,
        "true_params": {
            "gamma": baseline.gamma,
            "beta": baseline.beta,
            "phi": baseline.phi,
            "sigma_a": baseline.sigma_a,
            "sigma_skill": baseline.sigma_skill,
            "lambda_skill": baseline.lambda_skill,
            "V": baseline.V.tolist(),
            "c": baseline.C.tolist(),
        },
        "true_params_transformed": {
            "gamma": baseline_tilde_np[0],
            "beta": baseline_tilde_np[1],
            "phi_over_sigma_skill": baseline_tilde_np[2],
            "sigma_skill": baseline_tilde_np[3],
            "lambda_skill": baseline_tilde_np[4],
            "V": baseline_tilde_np[5 : 5 + data.J].tolist(),
            "c_tilde": baseline_tilde_np[5 + data.J :].tolist(),
        },
        "market_shares": {
            "empirical": data.shares_empirical.tolist(),
            "model": shares_model.tolist(),
        },
    }

    if theta_plot_path is not None:
        out["theta_plot_path"] = str(theta_plot_path)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
