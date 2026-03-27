#!/usr/bin/env python3
"""Penalized MLE for (τ, α, γ, σ_e, λ_e, δ, q̄) with JAX.

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
    from ... import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_ESTIMATION
    from ..lib.helpers import (
        compute_worker_firm_distances,
        load_weight_matrix,
        naive_theta_guess_tau_delta_qbar,
        read_firms_data,
        read_parameters,
        read_workers_data,
    )
    from ..lib.jax_model import (
        compute_penalty_components_with_params_jax,
        enable_x64,
    )
    from ..lib.blp_contraction import solve_delta_contraction_batched
except ImportError:  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from screening import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_ESTIMATION  # type: ignore
    from screening.analysis.lib.helpers import (  # type: ignore
        compute_worker_firm_distances,
        load_weight_matrix,
        naive_theta_guess_tau_delta_qbar,
        read_firms_data,
        read_parameters,
        read_workers_data,
    )
    from screening.analysis.lib.jax_model import (  # type: ignore
        compute_penalty_components_with_params_jax,
        enable_x64,
    )
    from screening.analysis.lib.blp_contraction import solve_delta_contraction_batched  # type: ignore


# ---------------------------------------------------------------------------
# Dataclasses and small utilities
# ---------------------------------------------------------------------------

EPS_TAU = 1e-6
EPS_ALPHA = 1e-6
POS_FLOOR = 1e-6


@dataclass(frozen=True)
class Baseline:
    tau: float
    alpha: float
    gamma1: float
    sigma_e_baseline: float
    sigma_e: float
    lambda_e: float
    delta: np.ndarray
    Qbar: np.ndarray

    def vector(self) -> np.ndarray:
        head = np.array(
            [self.tau, self.alpha, self.gamma1, self.sigma_e, self.lambda_e],
            dtype=float,
        )
        return np.concatenate((head, self.delta, self.Qbar))


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
    # Multi-market batched fields (populated when M > 0)
    M: int = 0
    J_per: int = 0
    N_per: int = 0
    X_batch: jnp.ndarray | None = None
    choice_batch: jnp.ndarray | None = None
    D_batch: jnp.ndarray | None = None
    gamma0_scalar: jnp.ndarray | None = None
    w_batch: jnp.ndarray | None = None
    Y_batch: jnp.ndarray | None = None
    labor_batch: jnp.ndarray | None = None
    shares_emp_batch: jnp.ndarray | None = None
    delta_init_batch: jnp.ndarray | None = None


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
    """Return (forward, inverse) transforms between R^K and bounded theta.

    ln_qbar is unconstrained (log-space cutoffs), so identity transform.
    """

    def fwd(z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=jnp.float64)
        tau = EPS_TAU + (1.0 - 2.0 * EPS_TAU) * jnn.sigmoid(z[0])
        alpha = EPS_ALPHA + (1.0 - 2.0 * EPS_ALPHA) * jnn.sigmoid(z[1])
        gamma1 = POS_FLOOR + jnn.softplus(z[2])
        sigma_e = POS_FLOOR + jnn.softplus(z[3])
        lambda_e = z[4]
        delta = z[5 : 5 + J]
        ln_qbar = z[5 + J :]  # already unconstrained
        return jnp.concatenate(
            (
                jnp.array(
                    [tau, alpha, gamma1, sigma_e, lambda_e], dtype=jnp.float64
                ),
                delta,
                ln_qbar,
            )
        )

    def inv(theta: jnp.ndarray) -> jnp.ndarray:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        tau = theta[0]
        alpha = theta[1]
        gamma1 = theta[2]
        sigma_e = theta[3]
        lambda_e = theta[4]
        delta = theta[5 : 5 + J]
        ln_qbar = theta[5 + J :]

        tau_scaled = (tau - EPS_TAU) / (1.0 - 2.0 * EPS_TAU)
        alpha_scaled = (alpha - EPS_ALPHA) / (1.0 - 2.0 * EPS_ALPHA)

        tau_z = _logit(jnp.clip(tau_scaled, 1e-12, 1.0 - 1e-12))
        alpha_z = _logit(jnp.clip(alpha_scaled, 1e-12, 1.0 - 1e-12))
        gamma1_z = _softplus_inv(jnp.maximum(gamma1 - POS_FLOOR, 1e-12))
        sigma_e_z = _softplus_inv(jnp.maximum(sigma_e - POS_FLOOR, 1e-12))

        return jnp.concatenate(
            (
                jnp.array(
                    [tau_z, alpha_z, gamma1_z, sigma_e_z, lambda_e],
                    dtype=jnp.float64,
                ),
                delta,
                ln_qbar,  # identity — no transform needed
            )
        )

    return ParameterTransform(fwd=fwd, inv=inv)


def make_transform_no_delta(J: int) -> ParameterTransform:
    """Return (forward, inverse) transforms for the contraction-mapped parameter vector.

    Parameter vector: [tau, alpha, gamma1, sigma_e, lambda_e, ln_qbar_1..J] (length 5+J).
    Same transforms as make_transform but without delta entries.
    ln_qbar is unconstrained (identity transform).
    """

    def fwd(z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=jnp.float64)
        tau = EPS_TAU + (1.0 - 2.0 * EPS_TAU) * jnn.sigmoid(z[0])
        alpha = EPS_ALPHA + (1.0 - 2.0 * EPS_ALPHA) * jnn.sigmoid(z[1])
        gamma1 = POS_FLOOR + jnn.softplus(z[2])
        sigma_e = POS_FLOOR + jnn.softplus(z[3])
        lambda_e = z[4]
        ln_qbar = z[5:]  # identity
        return jnp.concatenate(
            (
                jnp.array(
                    [tau, alpha, gamma1, sigma_e, lambda_e], dtype=jnp.float64
                ),
                ln_qbar,
            )
        )

    def inv(theta: jnp.ndarray) -> jnp.ndarray:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        tau = theta[0]
        alpha = theta[1]
        gamma1 = theta[2]
        sigma_e = theta[3]
        lambda_e = theta[4]
        ln_qbar = theta[5:]

        tau_scaled = (tau - EPS_TAU) / (1.0 - 2.0 * EPS_TAU)
        alpha_scaled = (alpha - EPS_ALPHA) / (1.0 - 2.0 * EPS_ALPHA)

        tau_z = _logit(jnp.clip(tau_scaled, 1e-12, 1.0 - 1e-12))
        alpha_z = _logit(jnp.clip(alpha_scaled, 1e-12, 1.0 - 1e-12))
        gamma1_z = _softplus_inv(jnp.maximum(gamma1 - POS_FLOOR, 1e-12))
        sigma_e_z = _softplus_inv(jnp.maximum(sigma_e - POS_FLOOR, 1e-12))

        return jnp.concatenate(
            (
                jnp.array(
                    [tau_z, alpha_z, gamma1_z, sigma_e_z, lambda_e],
                    dtype=jnp.float64,
                ),
                ln_qbar,  # identity
            )
        )

    return ParameterTransform(fwd=fwd, inv=inv)


def transform_theta_and_jacobian(theta: np.ndarray, J: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply affine transformation to theta and return (theta_tilde, Jac)."""

    theta = np.asarray(theta, dtype=float).reshape(-1)
    K = theta.size
    if K != 5 + 2 * J:
        raise ValueError(f"Expected theta size {5 + 2 * J}, got {K}.")

    tau, alpha, gamma1, sigma_e, lambda_e = theta[:5]
    delta = theta[5 : 5 + J]
    ln_qbar = theta[5 + J :]

    inv_sigma = 1.0 / sigma_e
    gamma1_tilde = gamma1 * inv_sigma
    qbar_tilde = (ln_qbar - lambda_e) * inv_sigma

    theta_tilde = theta.copy()
    theta_tilde[2] = gamma1_tilde
    theta_tilde[5 + J :] = qbar_tilde

    jac = np.eye(K, dtype=float)
    jac[2, :] = 0.0
    jac[2, 2] = inv_sigma
    jac[2, 3] = -gamma1 * inv_sigma**2

    for j in range(J):
        row = 5 + J + j
        jac[row, :] = 0.0
        jac[row, 3] = -(ln_qbar[j] - lambda_e) * inv_sigma**2
        jac[row, 4] = -inv_sigma
        jac[row, 5 + J + j] = inv_sigma

    return theta_tilde, jac


# ---------------------------------------------------------------------------
# CLI parsing and data preparation
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    raw_dir = get_data_subdir(DATA_RAW, create=True)
    clean_dir = get_data_subdir(DATA_CLEAN, create=True)
    build_dir = get_data_subdir(DATA_BUILD, create=True)
    est_dir = get_output_subdir(OUTPUT_ESTIMATION, create=True)

    parser = argparse.ArgumentParser(
        description="Penalized JAX MLE for (tau, alpha, gamma1, sigma_e, delta, ln_qbar)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workers_path", type=str, default=str(build_dir / "workers_dataset.csv"))
    parser.add_argument("--firms_path", type=str, default=str(clean_dir / "equilibrium_firms.csv"))
    parser.add_argument("--params_path", type=str, default=str(raw_dir / "parameters_effective.csv"))
    parser.add_argument("--weight_matrix_path", type=str, default=None,
                        help="Path to JxJ weight matrix (csv/json/npy); defaults to identity.")
    parser.add_argument("--out_dir", type=str, default=str(est_dir))
    parser.add_argument("--theta0_file", type=str, default=None)
    parser.add_argument("--theta0_list", type=str, default=None,
                        help="Comma-separated tau, alpha, gamma1, sigma_e, delta(1..J), ln_qbar(1..J).")
    parser.add_argument("--theta0_from_helper", action="store_true",
                        help="Initialise tau,delta,qbar via helpers.naive_theta_guess_tau_delta_qbar and append alpha,gamma1,sigma_e baselines.")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--freeze", type=str, default=None,
                        help="Comma-separated parameter names to freeze at initial values "
                             "(e.g. 'gamma1,sigma_e,lambda_e'). Frozen params are held fixed during optimisation.")
    parser.add_argument("--penalty_weight", type=float, default=1.0,
                        help="Weight on the GMM penalty term. 0.0 = pure MLE (no screening FOC moments).")
    parser.add_argument("--skip_statistics", action="store_true",
                        help="Skip Hessian-based SE approximation (faster, no covariance output).")
    parser.add_argument("--skip_plot", action="store_true",
                        help="Skip diagnostic plot generation.")
    parser.add_argument("--delta_method", type=str, default="joint",
                        choices=["joint", "contraction"],
                        help="How to handle deltas: 'joint' optimises them directly; "
                             "'contraction' profiles them out via BLP contraction mapping.")
    parser.add_argument("--contraction_tol", type=float, default=5e-2,
                        help="Convergence tolerance for BLP contraction mapping. "
                             "The screening cutoff structure may prevent tighter convergence.")
    parser.add_argument("--contraction_maxiter", type=int, default=500,
                        help="Max iterations for BLP contraction mapping.")
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
        raise ValueError(f"theta_0 length {theta.size} != expected {expected}.")
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
        raise ValueError(f"theta_0 length {theta.size} != expected {expected}.")
    return theta


def _initial_theta(
    args: argparse.Namespace,
    J: int,
    delta0_nat: np.ndarray,
    qbar0_nat: np.ndarray,
    tau0: float,
    alpha_init: float,
    gamma_init: float,
    sigma_e_init: float,
    lambda_e_init: float,
    helper_kwargs: Dict[str, Any],
) -> np.ndarray:
    expected = 5 + 2 * J

    if args.theta0_list is not None:
        return _parse_theta_list(args.theta0_list, expected)

    if args.theta0_file is not None:
        return _load_theta_from_file(args.theta0_file, expected)

    if args.theta0_from_helper:
        print("Initialising tau,delta,qbar via helpers.naive_theta_guess_tau_delta_qbar ...")
        theta_partial = naive_theta_guess_tau_delta_qbar(**helper_kwargs)
        theta_partial = np.asarray(theta_partial, dtype=float)
        return np.concatenate(
            (
                theta_partial[:1],
                [alpha_init, gamma_init, sigma_e_init, lambda_e_init],
                theta_partial[1:],
            )
        )

    return np.concatenate(
        (
            [tau0, alpha_init, gamma_init, sigma_e_init, lambda_e_init],
            delta0_nat,
            qbar0_nat,
        )
    )


def _enforce_feasible(theta: np.ndarray, J: int) -> np.ndarray:
    theta = theta.astype(float, copy=True)
    theta[0] = np.clip(theta[0], EPS_TAU + 1e-12, 1.0 - EPS_TAU - 1e-12)
    theta[1] = np.clip(theta[1], EPS_ALPHA + 1e-12, 1.0 - EPS_ALPHA - 1e-12)
    theta[2] = max(theta[2], POS_FLOOR + 1e-9)
    theta[3] = max(theta[3], POS_FLOOR + 1e-9)
    # ln_qbar (theta[5+J:]) is unconstrained — no floor needed
    return theta


def _enforce_feasible_no_delta(theta: np.ndarray) -> np.ndarray:
    """Enforce bounds on the contraction-mapped theta (no delta entries)."""
    theta = theta.astype(float, copy=True)
    theta[0] = np.clip(theta[0], EPS_TAU + 1e-12, 1.0 - EPS_TAU - 1e-12)
    theta[1] = np.clip(theta[1], EPS_ALPHA + 1e-12, 1.0 - EPS_ALPHA - 1e-12)
    theta[2] = max(theta[2], POS_FLOOR + 1e-9)
    theta[3] = max(theta[3], POS_FLOOR + 1e-9)
    # ln_qbar (theta[5:]) is unconstrained — no floor needed
    return theta


def _load_multi_market(firms_path: str, workers_path: str):
    """Load multi-market data into batched arrays for vmap processing.

    Returns flat arrays (for theta0 init) and batched arrays (M, N_per, ...)
    for efficient per-market computation via jax.vmap.
    """
    from scipy.spatial.distance import cdist

    firms_df = pd.read_csv(firms_path)
    workers_df = pd.read_csv(workers_path)

    firms_df = firms_df.sort_values(['market_id', 'firm_id']).reset_index(drop=True)
    workers_df = workers_df.sort_values(['market_id']).reset_index(drop=True)

    markets = sorted(firms_df['market_id'].unique())
    M = len(markets)

    J_counts = firms_df.groupby('market_id').size()
    N_counts = workers_df.groupby('market_id').size()
    if J_counts.nunique() != 1 or N_counts.nunique() != 1:
        raise ValueError(
            "Multi-market vmap requires equal J and N across markets; "
            f"got J sizes {J_counts.unique()} and N sizes {N_counts.unique()}"
        )
    J_per = int(J_counts.iloc[0])
    N_per = int(N_counts.iloc[0])
    J_total = M * J_per

    # Pre-allocate batched arrays
    D_batch = np.zeros((M, N_per, J_per))
    X_batch = np.zeros((M, N_per))
    choice_batch = np.zeros((M, N_per), dtype=np.int32)
    w_batch = np.zeros((M, J_per))
    Y_batch = np.zeros((M, J_per))
    xi_batch = np.zeros((M, J_per))
    qbar_batch = np.zeros((M, J_per))
    labor_batch = np.zeros((M, J_per))

    qbar_col = 'qbar' if 'qbar' in firms_df.columns else 'c'

    for i, mid in enumerate(markets):
        fdf = firms_df[firms_df['market_id'] == mid].sort_values('firm_id')
        wdf = workers_df[workers_df['market_id'] == mid]

        X_batch[i] = wdf['x_skill'].values
        choice_batch[i] = wdf['chosen_firm'].values.astype(np.int32)

        w_batch[i] = fdf['w'].values
        Y_batch[i] = fdf['Y'].values
        xi_batch[i] = fdf['xi'].values
        qbar_batch[i] = fdf[qbar_col].values
        loc_firms_m = fdf[['x', 'y']].values

        worker_locs = np.column_stack([wdf['ell_x'].values, wdf['ell_y'].values])
        D_batch[i] = cdist(worker_locs, loc_firms_m, metric='euclidean')

        counts = np.bincount(choice_batch[i], minlength=J_per + 1).astype(float)
        labor_batch[i] = counts[1:]

    # Flat arrays for theta0 init and baseline
    w_flat = w_batch.reshape(-1)
    xi_flat = xi_batch.reshape(-1)
    qbar_flat = qbar_batch.reshape(-1)
    Y_flat = Y_batch.reshape(-1)
    firm_ids_global = np.arange(1, J_total + 1)

    print(f"  Multi-market data: M={M}, J_per={J_per}, N_per={N_per}, "
          f"J_total={J_total}, N_total={M * N_per}")

    return dict(
        firm_ids=firm_ids_global,
        w_flat=w_flat, Y_flat=Y_flat, xi_flat=xi_flat, qbar_flat=qbar_flat,
        D_batch=D_batch, X_batch=X_batch, choice_batch=choice_batch,
        w_batch=w_batch, Y_batch=Y_batch, labor_batch=labor_batch,
        M=M, J_per=J_per, N_per=N_per,
    )


def prepare_problem(args: argparse.Namespace) -> Tuple[ModelData, Baseline, jnp.ndarray, jnp.ndarray, ParameterTransform]:
    params_path = Path(args.params_path)
    firms_path = Path(args.firms_path)
    workers_path = Path(args.workers_path)

    params = read_parameters(str(params_path))

    # --- Detect multi-market data ---
    _header = pd.read_csv(firms_path, nrows=0).columns
    is_multi_market = 'market_id' in _header

    if is_multi_market:
        mm = _load_multi_market(str(firms_path), str(workers_path))
        firm_ids = mm['firm_ids']
        w = mm['w_flat']
        Y = mm['Y_flat']
        xi = mm['xi_flat']
        qbar_data = mm['qbar_flat']
        J = firm_ids.size
        N = mm['M'] * mm['N_per']
    else:
        firm_ids, w, Y, _, xi, loc_firms, qbar_data = read_firms_data(str(firms_path))
        x_skill, ell_x, ell_y, chosen_firm = read_workers_data(str(workers_path))
        N = x_skill.size
        J = firm_ids.size
        mm = None

    alpha_param = params.get("alpha", 1.0)
    gamma0_val = float(params.get("gamma0", 0.76))
    gamma1_baseline = float(params.get("gamma1", params.get("gamma", 0.94)))
    sigma_baseline = float(params.get("sigma_e", params.get("sigma_a", 0.135)))
    tau_baseline = float(params.get("tau", 0.05))
    alpha_baseline = float(params.get("alpha", params.get("beta", 0.5)))

    alpha_init = float(alpha_baseline)
    gamma1_init = float(gamma1_baseline)
    sigma_e_init = float(sigma_baseline)
    lambda_e_init = 0.0

    eta_param = params.get("eta", 1.0)
    delta0_nat = eta_param * np.log(np.maximum(w, 1e-300)) + xi
    qbar0_nat = np.log(np.maximum(qbar_data, 1e-300))  # ln_qbar (log space)

    if is_multi_market and args.theta0_from_helper:
        print("  WARNING: --theta0_from_helper not supported for multi-market; "
              "using default init (delta from data, qbar from data)")
        args_copy = argparse.Namespace(**vars(args))
        args_copy.theta0_from_helper = False
        args_copy.theta0_list = None
        args_copy.theta0_file = None
    else:
        args_copy = args

    if not is_multi_market:
        helper_kwargs = dict(
            x_skill=x_skill, ell_x=ell_x, ell_y=ell_y,
            chosen_firm=chosen_firm, loc_firms=loc_firms, firm_ids=firm_ids,
            tau0=tau_baseline, firms_csv_path=str(firms_path),
            gamma0=gamma0_val, gamma1=gamma1_init, alpha=alpha_init,
        )
    else:
        helper_kwargs = {}  # not used when theta0_from_helper is disabled

    theta0_np = _initial_theta(
        args_copy, J, delta0_nat, qbar0_nat, tau_baseline,
        alpha_init, gamma1_init, sigma_e_init, lambda_e_init, helper_kwargs,
    )
    theta0_np = _enforce_feasible(theta0_np, J)

    weight_matrix_np = load_weight_matrix(args.weight_matrix_path, 4)

    # Detect contraction method
    delta_method = getattr(args, 'delta_method', 'joint')

    if is_multi_market:
        # Batched model data for vmap processing
        M, J_per, N_per = mm['M'], mm['J_per'], mm['N_per']

        # Flat labor counts and shares (for reporting only)
        labor_flat = mm['labor_batch'].reshape(-1)
        shares_empirical = np.zeros(J + 1)  # placeholder

        # Compute per-market empirical firm shares from choice data.
        # Floor at 0.5/N_per to avoid log(0) in the BLP contraction for
        # firms with zero observed workers (standard BLP practice).
        choice_np = mm['choice_batch']  # (M, N_per) int32
        share_floor = 0.5 / N_per
        shares_emp_np = np.zeros((M, J_per), dtype=float)
        for m_idx in range(M):
            for j_idx in range(J_per):
                shares_emp_np[m_idx, j_idx] = np.mean(choice_np[m_idx] == (j_idx + 1))
        # Apply floor and renormalize so firm shares sum to <= 1
        shares_emp_np = np.maximum(shares_emp_np, share_floor)
        row_sums = shares_emp_np.sum(axis=1, keepdims=True)
        # Cap total firm share at 1 (preserve outside option share)
        too_large = row_sums > 1.0
        shares_emp_np = np.where(too_large, shares_emp_np / row_sums, shares_emp_np)

        model_data = ModelData(
            X=jnp.zeros(0),  # unused in batched path
            choice_idx=jnp.zeros(0, dtype=jnp.int32),
            aux_template={},
            w_nat=jnp.asarray(w, dtype=jnp.float64),
            Y_nat=jnp.asarray(Y, dtype=jnp.float64),
            labor_counts=jnp.asarray(labor_flat, dtype=jnp.float64),
            weight_matrix=jnp.asarray(weight_matrix_np, dtype=jnp.float64),
            N=N, J=J, firm_ids=firm_ids,
            shares_empirical=shares_empirical,
            M=M, J_per=J_per, N_per=N_per,
            X_batch=jnp.asarray(mm['X_batch'], dtype=jnp.float64),
            choice_batch=jnp.asarray(mm['choice_batch'], dtype=jnp.int32),
            D_batch=jnp.asarray(mm['D_batch'], dtype=jnp.float64),
            gamma0_scalar=jnp.asarray(float(gamma0_val), dtype=jnp.float64),
            w_batch=jnp.asarray(mm['w_batch'], dtype=jnp.float64),
            Y_batch=jnp.asarray(mm['Y_batch'], dtype=jnp.float64),
            labor_batch=jnp.asarray(mm['labor_batch'], dtype=jnp.float64),
            shares_emp_batch=jnp.asarray(shares_emp_np, dtype=jnp.float64),
            delta_init_batch=jnp.asarray(delta0_nat.reshape(M, J_per), dtype=jnp.float64),
        )
    else:
        D_nat = compute_worker_firm_distances(ell_x, ell_y, loc_firms)
        counts_full = np.bincount(chosen_firm, minlength=J + 1).astype(float)
        shares_empirical = counts_full / max(N, 1)
        labor_counts = counts_full[1:]

        model_data = ModelData(
            X=jnp.asarray(x_skill, dtype=jnp.float64),
            choice_idx=jnp.asarray(chosen_firm.astype(np.int32)),
            aux_template={
                "D_nat": jnp.asarray(D_nat, dtype=jnp.float64),
                "gamma0": jnp.asarray(float(gamma0_val), dtype=jnp.float64),
                "firm_ids": jnp.asarray(firm_ids, dtype=jnp.int32),
            },
            w_nat=jnp.asarray(w, dtype=jnp.float64),
            Y_nat=jnp.asarray(Y, dtype=jnp.float64),
            labor_counts=jnp.asarray(labor_counts, dtype=jnp.float64),
            weight_matrix=jnp.asarray(weight_matrix_np, dtype=jnp.float64),
            N=N, J=J, firm_ids=firm_ids,
            shares_empirical=shares_empirical,
        )

    baseline = Baseline(
        tau=float(tau_baseline),
        alpha=float(alpha_baseline),
        gamma1=float(gamma1_baseline),
        sigma_e_baseline=float(sigma_baseline),
        sigma_e=float(sigma_e_init),
        lambda_e=float(lambda_e_init),
        delta=delta0_nat.astype(float),
        Qbar=qbar0_nat.astype(float),
    )

    if delta_method == "contraction":
        # Contraction method: theta = [tau, alpha, gamma1, sigma_e, lambda_e, ln_qbar_1..J]
        theta0_no_delta = np.concatenate((
            theta0_np[:5],     # tau, alpha, gamma1, sigma_e, lambda_e
            theta0_np[5 + J:], # qbar
        ))
        theta0_no_delta = _enforce_feasible_no_delta(theta0_no_delta)
        transform = make_transform_no_delta(J)
        theta0 = jnp.asarray(theta0_no_delta, dtype=jnp.float64)
        z0 = transform.inv(theta0)
    else:
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
        gamma1 = theta[2]
        sigma_e = theta[3]
        lambda_e = theta[4]
        theta_core = jnp.concatenate((theta[:2], theta[5:]))
        sigma_e_fixed = jnp.array(1.0, dtype=theta.dtype)
        P_nat, per_obs_nll, _, L_data, Q_nat = compute_penalty_components_with_params_jax(
            theta_core,
            data.X,
            data.choice_idx,
            data.aux_template,
            data.w_nat,
            data.Y_nat,
            data.labor_counts,
            gamma1,
            sigma_e_fixed,
        )

        alpha = theta[1]
        ln_qbar_nat = theta[5 + J :]
        qbar_nat_levels = jnp.exp(ln_qbar_nat)

        safe_Y = jnp.maximum(data.Y_nat, 1e-12)
        safe_L = jnp.maximum(L_data, 1e-12)
        safe_w = jnp.maximum(data.w_nat, 1e-12)
        safe_sigma_e = jnp.maximum(sigma_e, 1e-6)
        safe_Q_shift = jnp.maximum(Q_nat - lambda_e, 1e-12)

        scale = data.Y_nat / (safe_L * safe_w)
        alpha_term = (1.0 - alpha) * scale
        adjustment = Q_nat - lambda_e - alpha_term * qbar_nat_levels + lambda_e * alpha_term

        m1 = (
            jnp.log(safe_Y)
            - (1.0 - alpha) * jnp.log(safe_L)
            - ((1.0 - alpha) / safe_sigma_e) * jnp.log(safe_Q_shift)
        )
        m2 = adjustment
        m3 = scale * qbar_nat_levels * adjustment
        m4 = scale * adjustment

        m_total = jnp.stack(
            (
                jnp.sum(m1),
                jnp.sum(m2),
                jnp.sum(m3),
                jnp.sum(m4),
            )
        )

        return P_nat, per_obs_nll, m_total, L_data, Q_nat

    return components


def make_components_fn_batched(data: ModelData):
    """Return a JIT-compiled components function for multi-market data.

    Uses jax.vmap to process each market independently, avoiding the
    O(N_total * J_total * J_total) memory blowup of the single-market path.
    """
    from screening.analysis.lib.jax_model import compute_penalty_components_jax

    M = data.M
    J_per = data.J_per
    J_total = data.J

    def _single_market_core(theta_core, X_m, choice_m, D_m, w_m, Y_m, L_m,
                            gamma0, gamma1, sigma_e):
        aux = {"D_nat": D_m, "gamma0": gamma0, "gamma1": gamma1, "sigma_e": sigma_e}
        _, per_obs_nll, _, L_data, Q_nat = compute_penalty_components_jax(
            theta_core, X_m, choice_m, aux, w_m, Y_m, L_m,
        )
        return jnp.sum(per_obs_nll), Q_nat, L_data

    _vmapped_core = jax.vmap(
        _single_market_core,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None),
    )

    @jax.jit
    def components(theta: jnp.ndarray):
        gamma1 = theta[2]
        sigma_e = theta[3]
        lambda_e = theta[4]
        tau = theta[0]
        alpha = theta[1]

        delta_all = theta[5 : 5 + J_total]
        qbar_all = theta[5 + J_total :]
        delta_batch = delta_all.reshape(M, J_per)
        qbar_batch = qbar_all.reshape(M, J_per)

        # Per-market theta_core: [tau, alpha, delta_m, qbar_m]
        tau_alpha = jnp.broadcast_to(
            jnp.stack([tau, alpha]), (M, 2)
        )
        theta_core_batch = jnp.concatenate(
            [tau_alpha, delta_batch, qbar_batch], axis=1
        )

        sigma_e_fixed = jnp.array(1.0, dtype=theta.dtype)

        nlls, Q_batch, L_batch = _vmapped_core(
            theta_core_batch,
            data.X_batch,
            data.choice_batch,
            data.D_batch,
            data.w_batch,
            data.Y_batch,
            data.labor_batch,
            data.gamma0_scalar,
            gamma1,
            sigma_e_fixed,
        )

        per_market_nll = nlls  # (M,)

        # Moment vector: flatten Q, L and compute over all firms
        Q_flat = Q_batch.reshape(J_total)
        L_flat = L_batch.reshape(J_total)
        w_flat = data.w_nat
        Y_flat = data.Y_nat

        safe_alpha = jnp.clip(alpha, 1e-6, 1.0 - 1e-6)
        safe_sigma_e = jnp.maximum(sigma_e, 1e-6)
        safe_Q_shift = jnp.maximum(Q_flat - lambda_e, 1e-12)
        safe_L = jnp.maximum(L_flat, 1e-12)
        safe_Y = jnp.maximum(Y_flat, 1e-12)
        safe_w = jnp.maximum(w_flat, 1e-12)

        scale = Y_flat / (safe_L * safe_w)
        adjustment = (Q_flat - lambda_e
                      - (1.0 - alpha) * scale * qbar_all
                      + lambda_e * (1.0 - alpha) * scale)

        m1 = (jnp.log(safe_Y) - (1.0 - alpha) * jnp.log(safe_L)
              - ((1.0 - alpha) / safe_sigma_e) * jnp.log(safe_Q_shift))
        m2 = adjustment
        m3 = scale * qbar_all * adjustment
        m4 = scale * adjustment
        m_total = jnp.stack((
            jnp.sum(m1), jnp.sum(m2), jnp.sum(m3), jnp.sum(m4),
        ))

        return None, per_market_nll, m_total, L_flat, Q_flat

    return components


def make_components_fn_batched_contraction(data: ModelData, contraction_tol: float = 1e-12,
                                           contraction_maxiter: int = 1000):
    """Return a components function that solves delta via BLP contraction.

    The theta vector has shape (5+J,): [tau, alpha, gamma1, sigma_e, lambda_e, ln_qbar_1..J].
    Deltas are solved externally via a Python-loop contraction (not JIT-traced),
    then passed into a JIT-compiled objective evaluation.

    The returned function is NOT itself jit'd because the contraction uses a
    Python loop. The L-BFGS optimizer calls value_and_grad on a wrapper that
    differentiates only through the JIT'd objective (deltas are stop_gradient'd).
    """
    from screening.analysis.lib.jax_model import compute_penalty_components_jax
    from screening.analysis.lib.blp_contraction import _contraction_step, _model_shares_vmapped

    M = data.M
    J_per = data.J_per
    J_total = data.J

    # Mutable cache for warm-starting delta across L-BFGS iterations.
    # First call starts from eta * log(w) + xi; subsequent calls reuse
    # the previous solution, which is nearby when theta changes slowly.
    delta_cache = [data.delta_init_batch]

    def _single_market_core(theta_core, X_m, choice_m, D_m, w_m, Y_m, L_m,
                            gamma0, gamma1, sigma_e):
        aux = {"D_nat": D_m, "gamma0": gamma0, "gamma1": gamma1, "sigma_e": sigma_e}
        _, per_obs_nll, _, L_data, Q_nat = compute_penalty_components_jax(
            theta_core, X_m, choice_m, aux, w_m, Y_m, L_m,
        )
        return jnp.sum(per_obs_nll), Q_nat, L_data

    _vmapped_core = jax.vmap(
        _single_market_core,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None),
    )

    @jax.jit
    def _objective_given_delta(theta: jnp.ndarray, delta_batch: jnp.ndarray):
        """JIT-compiled objective given already-solved deltas (treated as constant)."""
        tau = theta[0]
        alpha = theta[1]
        gamma1 = theta[2]
        sigma_e = theta[3]
        lambda_e = theta[4]
        qbar_all = theta[5:]
        qbar_batch = qbar_all.reshape(M, J_per)

        sigma_e_fixed = jnp.array(1.0, dtype=theta.dtype)

        tau_alpha = jnp.broadcast_to(jnp.stack([tau, alpha]), (M, 2))
        theta_core_batch = jnp.concatenate(
            [tau_alpha, delta_batch, qbar_batch], axis=1
        )

        nlls, Q_batch, L_batch = _vmapped_core(
            theta_core_batch,
            data.X_batch,
            data.choice_batch,
            data.D_batch,
            data.w_batch,
            data.Y_batch,
            data.labor_batch,
            data.gamma0_scalar,
            gamma1,
            sigma_e_fixed,
        )

        per_market_nll = nlls
        Q_flat = Q_batch.reshape(J_total)
        L_flat = L_batch.reshape(J_total)

        safe_alpha = jnp.clip(alpha, 1e-6, 1.0 - 1e-6)
        safe_sigma_e = jnp.maximum(sigma_e, 1e-6)
        safe_Q_shift = jnp.maximum(Q_flat - lambda_e, 1e-12)
        safe_L = jnp.maximum(L_flat, 1e-12)
        safe_Y = jnp.maximum(data.Y_nat, 1e-12)
        safe_w = jnp.maximum(data.w_nat, 1e-12)

        scale = data.Y_nat / (safe_L * safe_w)
        adjustment = (Q_flat - lambda_e
                      - (1.0 - alpha) * scale * qbar_all
                      + lambda_e * (1.0 - alpha) * scale)

        m1 = (jnp.log(safe_Y) - (1.0 - alpha) * jnp.log(safe_L)
              - ((1.0 - alpha) / safe_sigma_e) * jnp.log(safe_Q_shift))
        m2 = adjustment
        m3 = scale * qbar_all * adjustment
        m4 = scale * adjustment
        m_total = jnp.stack((
            jnp.sum(m1), jnp.sum(m2), jnp.sum(m3), jnp.sum(m4),
        ))

        return None, per_market_nll, m_total, L_flat, Q_flat

    def components(theta: jnp.ndarray):
        """Solve delta via Python-loop contraction, then evaluate JIT'd objective."""
        tau = theta[0]
        gamma1 = theta[2]
        qbar_all = theta[5:]
        qbar_batch = qbar_all.reshape(M, J_per)

        sigma_e_fixed = jnp.array(1.0, dtype=theta.dtype)

        # Run contraction in Python loop (not JAX-traced)
        log_s_data_batch = jnp.log(jnp.maximum(data.shares_emp_batch, 1e-300))
        delta = delta_cache[0]  # warm start from previous solution
        for it in range(contraction_maxiter):
            delta, err = _contraction_step(
                delta, tau, qbar_batch, data.X_batch, data.D_batch,
                log_s_data_batch, gamma1, sigma_e_fixed, data.gamma0_scalar,
            )
            if float(err) <= contraction_tol:
                break
        delta_cache[0] = delta  # cache for next L-BFGS iteration

        # Pass solved delta (as constant) to JIT'd objective
        delta_stopped = jax.lax.stop_gradient(delta)
        return _objective_given_delta(theta, delta_stopped)

    return components


def build_frozen_mask(freeze_str: str | None, J: int) -> tuple[jnp.ndarray, list[str]]:
    """Parse --freeze argument into a boolean mask over the parameter vector.

    Returns (frozen_mask, frozen_names) where frozen_mask is shape (5+2J,).
    """
    K = 5 + 2 * J
    mask = np.zeros(K, dtype=bool)
    if freeze_str is None:
        return jnp.array(mask), []

    scalar_map = {"tau": 0, "alpha": 1, "gamma1": 2, "sigma_e": 3, "lambda_e": 4}
    frozen_names = []
    for tok in freeze_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in scalar_map:
            mask[scalar_map[tok]] = True
            frozen_names.append(tok)
        elif tok == "delta":
            mask[5 : 5 + J] = True
            frozen_names.append("delta")
        elif tok == "qbar":
            mask[5 + J :] = True
            frozen_names.append("qbar")
        else:
            raise ValueError(
                f"Unknown parameter name '{tok}' in --freeze. "
                f"Valid names: {list(scalar_map.keys()) + ['delta', 'qbar']}"
            )
    return jnp.array(mask), frozen_names


def build_frozen_mask_no_delta(freeze_str: str | None, J: int) -> tuple[jnp.ndarray, list[str]]:
    """Parse --freeze for contraction-mapped theta (no delta entries).

    Parameter vector: [tau, alpha, gamma1, sigma_e, lambda_e, ln_qbar_1..J] (length 5+J).
    """
    K = 5 + J
    mask = np.zeros(K, dtype=bool)
    if freeze_str is None:
        return jnp.array(mask), []

    scalar_map = {"tau": 0, "alpha": 1, "gamma1": 2, "sigma_e": 3, "lambda_e": 4}
    frozen_names = []
    for tok in freeze_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in scalar_map:
            mask[scalar_map[tok]] = True
            frozen_names.append(tok)
        elif tok == "delta":
            print("  WARNING: --freeze delta is a no-op with --delta_method contraction "
                  "(deltas are solved via contraction mapping)")
        elif tok == "qbar":
            mask[5:] = True
            frozen_names.append("qbar")
        else:
            raise ValueError(
                f"Unknown parameter name '{tok}' in --freeze. "
                f"Valid names: {list(scalar_map.keys()) + ['delta', 'qbar']}"
            )
    return jnp.array(mask), frozen_names


def make_value_fn(data: ModelData, components_fn, penalty_weight: float = 1.0):
    """Return a JIT-compiled penalized objective in theta-space."""

    pw = jnp.array(penalty_weight, dtype=jnp.float64)

    @jax.jit
    def value(theta: jnp.ndarray) -> jnp.ndarray:
        _, per_obs_nll, m_total, _, _ = components_fn(theta)
        nll = jnp.sum(per_obs_nll)
        if penalty_weight == 0.0:
            return nll
        penalty = 0.5 * m_total @ (data.weight_matrix @ m_total)
        return nll + pw * penalty

    return value


def make_value_and_grad_z(
    value_fn,
    transform: ParameterTransform,
    frozen_mask: jnp.ndarray | None = None,
    frozen_vals: jnp.ndarray | None = None,
):
    """Return JIT-compiled (value, grad) function in unconstrained z-space.

    If frozen_mask is provided, frozen parameters are replaced with frozen_vals
    after the forward transform, making their gradients exactly zero.
    """

    if frozen_mask is not None and jnp.any(frozen_mask):
        _mask = frozen_mask
        _vals = frozen_vals

        def loss_z(z: jnp.ndarray) -> jnp.ndarray:
            theta = transform.fwd(z)
            theta = jnp.where(_mask, _vals, theta)
            return value_fn(theta)
    else:

        def loss_z(z: jnp.ndarray) -> jnp.ndarray:
            theta = transform.fwd(z)
            return value_fn(theta)

    return jax.jit(jax.value_and_grad(loss_z))


def make_contraction_value_and_grad_z(
    data: ModelData,
    transform: ParameterTransform,
    penalty_weight: float = 1.0,
    frozen_mask: jnp.ndarray | None = None,
    frozen_vals: jnp.ndarray | None = None,
    contraction_tol: float = 1e-2,
    contraction_maxiter: int = 500,
):
    """Return a (value, grad) function for the BLP contraction path.

    Architecture:
    1. Python loop solves delta (not JAX-traced) → concrete delta array
    2. JIT'd function evaluates objective given (theta, delta_concrete)
    3. jax.value_and_grad differentiates through (2) only
    """
    from screening.analysis.lib.jax_model import compute_penalty_components_jax
    from screening.analysis.lib.blp_contraction import _contraction_step

    M = data.M
    J_per = data.J_per
    J_total = data.J
    pw = jnp.array(penalty_weight, dtype=jnp.float64)
    delta_cache = [data.delta_init_batch]
    has_frozen = frozen_mask is not None and jnp.any(frozen_mask)

    def _single_market_core(theta_core, X_m, choice_m, D_m, w_m, Y_m, L_m,
                            gamma0, gamma1, sigma_e):
        aux = {"D_nat": D_m, "gamma0": gamma0, "gamma1": gamma1, "sigma_e": sigma_e}
        _, per_obs_nll, _, L_data, Q_nat = compute_penalty_components_jax(
            theta_core, X_m, choice_m, aux, w_m, Y_m, L_m,
        )
        return jnp.sum(per_obs_nll), Q_nat, L_data

    _vmapped_core = jax.vmap(
        _single_market_core,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None),
    )

    @jax.jit
    def _objective_given_delta(theta: jnp.ndarray, delta_batch: jnp.ndarray):
        """Evaluate objective given solved deltas (constant w.r.t. theta)."""
        tau = theta[0]
        alpha = theta[1]
        gamma1 = theta[2]
        sigma_e = theta[3]
        lambda_e = theta[4]
        qbar_all = theta[5:]
        qbar_batch = qbar_all.reshape(M, J_per)
        sigma_e_fixed = jnp.array(1.0, dtype=theta.dtype)

        tau_alpha = jnp.broadcast_to(jnp.stack([tau, alpha]), (M, 2))
        theta_core_batch = jnp.concatenate(
            [tau_alpha, delta_batch, qbar_batch], axis=1
        )

        nlls, Q_batch, L_batch = _vmapped_core(
            theta_core_batch,
            data.X_batch, data.choice_batch, data.D_batch,
            data.w_batch, data.Y_batch, data.labor_batch,
            data.gamma0_scalar, gamma1, sigma_e_fixed,
        )

        nll = jnp.sum(nlls)

        if penalty_weight == 0.0:
            return nll

        Q_flat = Q_batch.reshape(J_total)
        L_flat = L_batch.reshape(J_total)
        safe_alpha = jnp.clip(alpha, 1e-6, 1.0 - 1e-6)
        safe_sigma_e = jnp.maximum(sigma_e, 1e-6)
        safe_Q_shift = jnp.maximum(Q_flat - lambda_e, 1e-12)
        safe_L = jnp.maximum(L_flat, 1e-12)
        safe_Y = jnp.maximum(data.Y_nat, 1e-12)
        safe_w = jnp.maximum(data.w_nat, 1e-12)

        scale = data.Y_nat / (safe_L * safe_w)
        adjustment = (Q_flat - lambda_e
                      - (1.0 - alpha) * scale * qbar_all
                      + lambda_e * (1.0 - alpha) * scale)
        m1 = (jnp.log(safe_Y) - (1.0 - alpha) * jnp.log(safe_L)
              - ((1.0 - alpha) / safe_sigma_e) * jnp.log(safe_Q_shift))
        m2 = adjustment
        m3 = scale * qbar_all * adjustment
        m4 = scale * adjustment
        m_total = jnp.stack((jnp.sum(m1), jnp.sum(m2), jnp.sum(m3), jnp.sum(m4)))

        penalty = 0.5 * m_total @ (data.weight_matrix @ m_total)
        return nll + pw * penalty

    # Pre-compute the z→objective function for a given delta.
    # jax.value_and_grad will be called on this each iteration, but since
    # _objective_given_delta is @jax.jit and delta has constant shape,
    # XLA caches the compiled kernel after the first call.
    def _loss_z_given_delta(z: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        theta = transform.fwd(z)
        if has_frozen:
            theta = jnp.where(frozen_mask, frozen_vals, theta)
        return _objective_given_delta(theta, delta)

    _vg_fn = jax.value_and_grad(_loss_z_given_delta, argnums=0)

    def value_and_grad_z(z: jnp.ndarray):
        """Solve contraction (Python), then compute value+grad (JIT)."""
        theta = transform.fwd(z)
        if has_frozen:
            theta = jnp.where(frozen_mask, frozen_vals, theta)

        tau = theta[0]
        gamma1 = theta[2]
        qbar_batch = theta[5:].reshape(M, J_per)
        sigma_e_fixed = jnp.array(1.0, dtype=theta.dtype)

        # --- Contraction (Python loop, not traced) ---
        log_s_data = jnp.log(jnp.maximum(data.shares_emp_batch, 1e-300))
        delta = delta_cache[0]  # warm start from previous solution
        for it in range(contraction_maxiter):
            delta, err = _contraction_step(
                delta, tau, qbar_batch, data.X_batch, data.D_batch,
                log_s_data, gamma1, sigma_e_fixed, data.gamma0_scalar,
            )
            err_val = float(err)
            if jnp.isnan(err) or jnp.any(jnp.isnan(delta)):
                # Diagnose which markets/firms went nan
                nan_mask = jnp.isnan(delta)  # (M, J_per)
                nan_markets = jnp.any(nan_mask, axis=1)  # (M,)
                n_nan_markets = int(jnp.sum(nan_markets))
                n_nan_firms = int(jnp.sum(nan_mask))
                # Check if nan firms correspond to small empirical shares
                min_shares_per_market = jnp.min(data.shares_emp_batch, axis=1)
                nan_market_ids = jnp.where(nan_markets)[0]
                if nan_market_ids.size > 0:
                    nan_min_shares = min_shares_per_market[nan_market_ids]
                    ok_market_ids = jnp.where(~nan_markets)[0]
                    ok_min_shares = min_shares_per_market[ok_market_ids] if ok_market_ids.size > 0 else jnp.array([])
                    print(f"  [nan-diag] {n_nan_markets} markets, {n_nan_firms} firms have nan delta. "
                          f"Min emp share in nan markets: {float(jnp.min(nan_min_shares)):.6f}, "
                          f"in ok markets: {float(jnp.min(ok_min_shares)):.6f}" if ok_min_shares.size > 0 else
                          f"  [nan-diag] {n_nan_markets} markets, {n_nan_firms} firms have nan delta.")
                # Warm start went bad; restart from cold init
                delta = data.delta_init_batch
                for it2 in range(contraction_maxiter):
                    delta, err = _contraction_step(
                        delta, tau, qbar_batch, data.X_batch, data.D_batch,
                        log_s_data, gamma1, sigma_e_fixed, data.gamma0_scalar,
                    )
                    if float(err) <= contraction_tol:
                        break
                break
            if err_val <= contraction_tol:
                break
        # Only cache if delta is finite
        if not bool(jnp.any(jnp.isnan(delta))):
            delta_cache[0] = delta

        # --- Value + grad through JIT'd objective ---
        # delta is concrete here, so it's a constant in the JIT trace.
        val, grad_z = _vg_fn(z, delta)
        if not hasattr(value_and_grad_z, '_call_count'):
            value_and_grad_z._call_count = 0
        value_and_grad_z._call_count += 1
        if value_and_grad_z._call_count <= 5:
            print(f"  [debug] iter={value_and_grad_z._call_count}: val={float(val):.6f}, "
                  f"grad_norm={float(jnp.linalg.norm(grad_z)):.3e}, "
                  f"contraction_iters={it+1}, contraction_err={float(err):.3e}, "
                  f"delta_has_nan={bool(jnp.any(jnp.isnan(delta)))}")
        return val, grad_z

    return value_and_grad_z


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------


def distance_metrics(theta_hat: np.ndarray, baseline: Baseline) -> Dict[str, float]:
    J = baseline.delta.size
    norm_scale = np.sqrt(J) if J > 0 else 1.0

    delta_hat = theta_hat[5 : 5 + J]
    qbar_hat = theta_hat[5 + J :]
    delta_true = baseline.delta
    qbar_true = baseline.Qbar

    delta_rmse = float(np.sqrt(np.mean((delta_hat - delta_true) ** 2)))
    qbar_rmse = float(np.sqrt(np.mean((qbar_hat - qbar_true) ** 2)))
    delta_sd = float(np.std(delta_true)) if J > 0 else 1.0
    qbar_sd = float(np.std(qbar_true)) if J > 0 else 1.0
    delta_corr = float(np.corrcoef(delta_hat, delta_true)[0, 1]) if J > 1 else 0.0
    qbar_corr = float(np.corrcoef(qbar_hat, qbar_true)[0, 1]) if J > 1 else 0.0

    return {
        "tau_hat": float(theta_hat[0]),
        "tau_true": float(baseline.tau),
        "tau_error": float(theta_hat[0] - baseline.tau),
        "alpha_hat": float(theta_hat[1]),
        "alpha_true": float(baseline.alpha),
        "alpha_error": float(theta_hat[1] - baseline.alpha),
        "gamma1_hat": float(theta_hat[2]),
        "gamma1_true": float(baseline.gamma1),
        "gamma1_error": float(theta_hat[2] - baseline.gamma1),
        "sigma_e_hat": float(theta_hat[3]),
        "sigma_e_true": float(baseline.sigma_e_baseline),
        "sigma_e_error": float(theta_hat[3] - baseline.sigma_e_baseline),
        "lambda_e_hat": float(theta_hat[4]),
        "lambda_e_error": float(theta_hat[4] - baseline.lambda_e),
        "delta_rmse": delta_rmse,
        "delta_rmse_over_sd": delta_rmse / max(delta_sd, 1e-12),
        "delta_corr": delta_corr,
        "delta_l2": float(np.linalg.norm(delta_hat - delta_true) / norm_scale),
        "qbar_rmse": qbar_rmse,
        "qbar_rmse_over_sd": qbar_rmse / max(qbar_sd, 1e-12),
        "qbar_corr": qbar_corr,
        "qbar_l2": float(np.linalg.norm(qbar_hat - qbar_true) / norm_scale),
    }


def make_param_names_raw(J: int) -> list[str]:
    return (
        ["tau", "alpha", "gamma1", "sigma_e", "lambda_e"]
        + [f"delta_{idx + 1}" for idx in range(J)]
        + [f"qbar_{idx + 1}" for idx in range(J)]
    )


def make_param_names_transformed(J: int) -> list[str]:
    return (
        ["tau", "alpha", "gamma1_over_sigma_e", "sigma_e", "lambda_e"]
        + [f"delta_{idx + 1}" for idx in range(J)]
        + [f"qbar_tilde_{idx + 1}" for idx in range(J)]
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

    use_contraction = (args.delta_method == "contraction")

    if use_contraction:
        frozen_mask, frozen_names = build_frozen_mask_no_delta(args.freeze, data.J)
        if data.M == 0:
            raise ValueError("--delta_method contraction requires multi-market data")
        print(f"  Using BLP contraction mapping (tol={args.contraction_tol}, "
              f"maxiter={args.contraction_maxiter})")
        print(f"  Parameter space: 5+J = {5 + data.J} (was 5+2J = {5 + 2 * data.J})")
    else:
        frozen_mask, frozen_names = build_frozen_mask(args.freeze, data.J)

    frozen_vals = theta0 if jnp.any(frozen_mask) else None

    if frozen_names:
        print(f"Freezing parameters: {frozen_names}")
    if args.penalty_weight != 1.0:
        print(f"Penalty weight: {args.penalty_weight}")

    if use_contraction:
        components_fn = make_components_fn_batched_contraction(
            data, contraction_tol=args.contraction_tol,
            contraction_maxiter=args.contraction_maxiter,
        )
        print(f"  Using contraction vmap path: M={data.M}, J_per={data.J_per}, N_per={data.N_per}")
        # Contraction path: custom value_and_grad that runs contraction in Python
        value_and_grad_z = make_contraction_value_and_grad_z(
            data, transform,
            penalty_weight=args.penalty_weight,
            frozen_mask=frozen_mask,
            frozen_vals=frozen_vals,
            contraction_tol=args.contraction_tol,
            contraction_maxiter=args.contraction_maxiter,
        )
    elif data.M > 0:
        components_fn = make_components_fn_batched(data)
        print(f"  Using batched vmap path: M={data.M}, J_per={data.J_per}, N_per={data.N_per}")
    else:
        components_fn = make_components_fn(data)

    if not use_contraction:
        value_fn = make_value_fn(data, components_fn, penalty_weight=args.penalty_weight)
        value_and_grad_z = make_value_and_grad_z(value_fn, transform, frozen_mask, frozen_vals)
    build_time = time.perf_counter() - build_start

    solver = LBFGS(
        fun=value_and_grad_z,
        value_and_grad=True,
        maxiter=int(args.maxiter),
        tol=float(args.tol),
        jit=not use_contraction,  # contraction uses Python loop, can't be JIT'd
    )

    solve_start = time.perf_counter()
    res = solver.run(z0)
    jax.block_until_ready(res.state.value)
    solve_time = time.perf_counter() - solve_start
    total_time = build_time + solve_time

    z_hat = res.params
    theta_hat = transform.fwd(z_hat)
    theta_hat_np = np.asarray(theta_hat, dtype=float)

    if use_contraction:
        # Reconstruct full theta by solving delta one final time
        tau_hat = theta_hat_np[0]
        qbar_hat_all = theta_hat_np[5:]
        gamma1_hat = theta_hat_np[2]
        sigma_e_hat_val = theta_hat_np[3]
        qbar_hat_batch = jnp.asarray(qbar_hat_all.reshape(data.M, data.J_per), dtype=jnp.float64)
        sigma_e_fixed = jnp.array(1.0, dtype=jnp.float64)
        delta_final, conv_final = solve_delta_contraction_batched(
            data.delta_init_batch,
            jnp.asarray(tau_hat, dtype=jnp.float64),
            qbar_hat_batch,
            data.X_batch,
            data.D_batch,
            data.shares_emp_batch,
            jnp.asarray(gamma1_hat, dtype=jnp.float64),
            sigma_e_fixed,
            data.gamma0_scalar,
            args.contraction_tol,
            args.contraction_maxiter,
        )
        n_converged = int(jnp.sum(conv_final))
        print(f"  BLP contraction (final): {n_converged}/{data.M} markets converged")
        delta_hat_all = np.asarray(delta_final.reshape(-1), dtype=float)

        # Build full theta: [tau, alpha, gamma1, sigma_e, lambda_e, delta_1..J, ln_qbar_1..J]
        theta_hat_full = np.concatenate((
            theta_hat_np[:5],
            delta_hat_all,
            theta_hat_np[5:],
        ))
        theta_hat_tilde_np, jac_transform = transform_theta_and_jacobian(theta_hat_full, data.J)

        # Also build full theta0 for reporting
        theta0_np_raw = np.asarray(theta0, dtype=float)
        theta0_full = np.concatenate((
            theta0_np_raw[:5],
            baseline.delta,  # original deltas for init
            theta0_np_raw[5:],
        ))
        theta0_tilde_np, _ = transform_theta_and_jacobian(theta0_full, data.J)
    else:
        theta_hat_full = theta_hat_np
        theta_hat_tilde_np, jac_transform = transform_theta_and_jacobian(theta_hat_np, data.J)
        theta0_np_raw = np.asarray(theta0, dtype=float)
        theta0_tilde_np, _ = transform_theta_and_jacobian(theta0_np_raw, data.J)

    baseline_vec = baseline.vector()
    baseline_raw_np = np.asarray(baseline_vec, dtype=float)
    baseline_tilde_np, _ = transform_theta_and_jacobian(baseline_raw_np, data.J)

    obj = float(res.state.value)
    grad_norm = float(np.linalg.norm(np.asarray(res.state.grad)))
    nit = int(res.state.iter_num)

    P_hat, per_obs_nll, m_hat, L_hat, Q_hat = components_fn(theta_hat)
    nll_hat = float(jnp.sum(per_obs_nll))
    penalty_hat = float(0.5 * m_hat @ (data.weight_matrix @ m_hat))

    m_hat_np = np.asarray(m_hat, dtype=float)
    L_hat_np = np.asarray(L_hat, dtype=float)
    Q_hat_np = np.asarray(Q_hat, dtype=float)

    if P_hat is not None:
        shares_model = np.asarray(P_hat, dtype=float).mean(axis=0)
        labor_model_np = np.asarray(P_hat, dtype=float)[:, 1:].sum(axis=0)
    else:
        # Batched path: P_hat not available
        shares_model = np.zeros(data.J + 1)
        labor_model_np = L_hat_np

    print(
        f"[LBFGS] objective={obj:.6f} (nll={nll_hat:.6f}, penalty={penalty_hat:.6f}), "
        f"nit={nit}, grad_norm={grad_norm:.3e}, build={build_time:.2f}s, "
        f"solve={solve_time:.2f}s, total={total_time:.2f}s"
    )

    if P_hat is not None and data.J <= 200:
        print("Market shares (empirical vs model):")
        for idx in range(data.J + 1):
            name = "outside" if idx == 0 else f"firm_{idx}"
            print(
                f"  {name}: {data.shares_empirical[idx]:.6f} vs {shares_model[idx]:.6f}"
            )
    else:
        print(f"  (Skipping per-firm share printout for J={data.J} firms)")

    param_names_raw = make_param_names_raw(data.J)
    param_names_trans = make_param_names_transformed(data.J)

    hessian_full: np.ndarray | None = None
    jmicro_mat: np.ndarray | None = None
    jfirm_mat: np.ndarray | None = None
    cov_penalized_trans: np.ndarray | None = None

    if args.skip_statistics or use_contraction:
        cov_penalized = None
        cov_penalized_trans = None
        se_penalized = np.full(len(param_names_raw), np.nan, dtype=float)
        ci_radius = np.full(len(param_names_raw), np.nan, dtype=float)
        se_penalized_trans = np.full(len(param_names_trans), np.nan, dtype=float)
        ci_radius_trans = np.full(len(param_names_trans), np.nan, dtype=float)
        if use_contraction:
            print("Skipping covariance computation (contraction mode — SE not yet supported).")
        else:
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

            print("Standard errors (raw theta) via H^{-1} Omega H^{-1}:")
            for name, se in zip(param_names_raw, se_penalized):
                print(f"  {name}: {se:.6f}")
            print("Standard errors (transformed theta_tilde):")
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

    distance_dict = distance_metrics(theta_hat_full, baseline)

    print("\n=== Parameter Recovery ===")
    for k in ("tau", "alpha", "gamma1", "sigma_e"):
        print(f"  {k:10s}: hat={distance_dict[f'{k}_hat']:.6f}  "
              f"true={distance_dict[f'{k}_true']:.6f}  "
              f"err={distance_dict[f'{k}_error']:+.6f}")
    print(f"  {'delta':10s}: RMSE/SD={distance_dict['delta_rmse_over_sd']:.4f}  "
          f"corr={distance_dict['delta_corr']:.4f}  "
          f"RMSE={distance_dict['delta_rmse']:.4f}")
    print(f"  {'qbar':10s}: RMSE/SD={distance_dict['qbar_rmse_over_sd']:.4f}  "
          f"corr={distance_dict['qbar_corr']:.4f}  "
          f"RMSE={distance_dict['qbar_rmse']:.4f}")

    if args.skip_plot:
        theta_plot_path = None
        print("Skipping plot generation (--skip_plot).")
    else:
        theta_plot_path = out_dir / "mle_penalized_gamma_sigma_e_theta_ci_plot.png"
        fig, axes = plt.subplots(7, 1, figsize=(10, 21), constrained_layout=True)

        def plot_scalar(ax, index: int, label: str, estimate: float, truth: float) -> None:
            radius = ci_radius_trans[index] if np.isfinite(ci_radius_trans[index]) else 0.0
            ax.errorbar(
                [0],
                [estimate],
                yerr=[radius],
                fmt="o",
                capsize=4,
                label="Estimate +/-95% CI",
            )
            ax.scatter([0], [truth], marker="x", color="red", label="Baseline theta_0")
            ax.set_xticks([0])
            ax.set_xticklabels([label])
            ax.set_ylabel(label)
            ax.set_title(f"{label} estimate")
            ax.legend()

        plot_scalar(axes[0], 0, "tau", theta_hat_tilde_np[0], baseline_tilde_np[0])
        plot_scalar(axes[1], 1, "alpha", theta_hat_tilde_np[1], baseline_tilde_np[1])
        plot_scalar(axes[2], 2, "gamma1/sigma_e", theta_hat_tilde_np[2], baseline_tilde_np[2])
        plot_scalar(axes[3], 3, "sigma_e", theta_hat_tilde_np[3], baseline_tilde_np[3])
        plot_scalar(axes[4], 4, "lambda_e", theta_hat_tilde_np[4], baseline_tilde_np[4])

        v_indices = np.arange(data.J)
        ci_delta = np.where(
            np.isfinite(ci_radius_trans[5 : 5 + data.J]),
            ci_radius_trans[5 : 5 + data.J],
            0.0,
        )
        axes[5].errorbar(
            v_indices,
            theta_hat_tilde_np[5 : 5 + data.J],
            yerr=ci_delta,
            fmt="o",
            capsize=4,
            label="Estimate +/-95% CI",
        )
        axes[5].scatter(
            v_indices,
            baseline.delta,
            marker="x",
            color="red",
            label="Baseline theta_0",
        )
        axes[5].set_xticks(v_indices)
        axes[5].set_xticklabels([f"delta_{idx + 1}" for idx in range(data.J)], rotation=45, ha="right")
        axes[5].set_ylabel("delta")
        axes[5].set_title("Firm delta estimates")
        axes[5].legend()

        ci_Qbar = np.where(
            np.isfinite(ci_radius_trans[5 + data.J :]),
            ci_radius_trans[5 + data.J :],
            0.0,
        )
        qbar_indices = np.arange(data.J)
        axes[6].errorbar(
            qbar_indices,
            theta_hat_tilde_np[5 + data.J :],
            yerr=ci_Qbar,
            fmt="o",
            capsize=4,
            label="Estimate +/-95% CI",
        )
        axes[6].scatter(
            qbar_indices,
            baseline_tilde_np[5 + data.J :],
            marker="x",
            color="red",
            label="Baseline theta_0",
        )
        axes[6].set_xticks(qbar_indices)
        axes[6].set_xticklabels([f"qbar_{idx + 1}" for idx in range(data.J)], rotation=45, ha="right")
        axes[6].set_ylabel("qbar_tilde")
        axes[6].set_title("Firm cutoff estimates (tilde)")
        axes[6].legend()

        fig.savefig(theta_plot_path, dpi=200)
        plt.close(fig)

    out_path = out_dir / "mle_tau_alpha_gamma_sigma_e_lambda_e_delta_qbar_penalty_estimates_jax.json"

    out = {
        "solver": "LBFGS",
        "frozen_params": frozen_names,
        "penalty_weight": args.penalty_weight,
        "objective": obj,
        "objective_breakdown": {
            "neg_log_likelihood": nll_hat,
            "penalty": penalty_hat,
        },
        "nit": nit,
        "grad_norm": grad_norm,
        "theta0": theta0_np_raw.tolist(),
        "theta0_transformed": theta0_tilde_np.tolist(),
        "delta_method": args.delta_method,
        "theta_hat": theta_hat_full.tolist(),
        "theta_hat_transformed": theta_hat_tilde_np.tolist(),
        "moment_vector": m_hat_np.tolist(),
        "labor_supplied_data": data.labor_counts.tolist(),
        "labor_supplied_model": labor_model_np.tolist(),
        "average_skill": Q_hat_np.tolist(),
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
            "tau": baseline.tau,
            "alpha": baseline.alpha,
            "gamma1": baseline.gamma1,
            "sigma_e_baseline": baseline.sigma_e_baseline,
            "sigma_e": baseline.sigma_e,
            "lambda_e": baseline.lambda_e,
            "delta": baseline.delta.tolist(),
            "qbar": baseline.Qbar.tolist(),
        },
        "true_params_transformed": {
            "tau": baseline_tilde_np[0],
            "alpha": baseline_tilde_np[1],
            "gamma1_over_sigma_e": baseline_tilde_np[2],
            "sigma_e": baseline_tilde_np[3],
            "lambda_e": baseline_tilde_np[4],
            "delta": baseline_tilde_np[5 : 5 + data.J].tolist(),
            "qbar_tilde": baseline_tilde_np[5 + data.J :].tolist(),
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

    # --- Append summary to persistent run log ---
    import csv, datetime

    log_path = Path(args.out_dir).parent / "mle_run_log.csv"
    log_fields = [
        "timestamp", "M", "J_per", "J_total", "N_total", "K",
        "maxiter", "nit", "mle_converged", "grad_norm",
        "nll", "penalty_weight",
        "frozen_params",
        "tau_hat", "tau_true", "tau_error",
        "alpha_hat", "alpha_true",
        "delta_rmse", "delta_rmse_over_sd", "delta_corr",
        "qbar_rmse", "qbar_rmse_over_sd", "qbar_corr",
        "build_time_s", "solve_time_s", "total_time_s",
        "per_iter_s", "memory_gb",
        "eq_converged", "eq_converged_pct",
        "out_dir",
    ]
    write_header = not log_path.exists()

    import resource
    mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # KB -> GB

    # Read equilibrium solver convergence metadata
    firms_dir = Path(args.firms_path).parent
    eq_converged_str = ""
    eq_converged_pct = ""
    meta_files = sorted(firms_dir.glob("**/equilibrium_firms*.meta.json"))
    if not meta_files:
        # Try the markets subfolder
        meta_files = sorted((firms_dir / "markets").glob("equilibrium_firms_market_*.meta.json"))
    if meta_files:
        n_converged = 0
        n_total = 0
        for mf in meta_files:
            try:
                with open(mf) as _mf:
                    meta = json.load(_mf)
                if not meta.get("noscreening", False):
                    n_total += 1
                    if meta.get("converged", False):
                        n_converged += 1
            except Exception:
                pass
        if n_total > 0:
            eq_converged_str = f"{n_converged}/{n_total}"
            eq_converged_pct = f"{100 * n_converged / n_total:.0f}"

    log_row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "M": data.M if data.M > 0 else 1,
        "J_per": data.J_per if data.M > 0 else data.J,
        "J_total": data.J,
        "N_total": data.N,
        "K": (5 + data.J) if use_contraction else (5 + 2 * data.J),
        "maxiter": args.maxiter,
        "nit": nit,
        "mle_converged": nit < args.maxiter,
        "grad_norm": f"{grad_norm:.4e}",
        "nll": f"{nll_hat:.4f}",
        "penalty_weight": args.penalty_weight,
        "frozen_params": freeze_str if (freeze_str := getattr(args, 'freeze', None)) else "",
        "tau_hat": f"{distance_dict['tau_hat']:.6f}",
        "tau_true": f"{distance_dict['tau_true']:.6f}",
        "tau_error": f"{distance_dict['tau_error']:.6f}",
        "alpha_hat": f"{distance_dict['alpha_hat']:.6f}",
        "alpha_true": f"{distance_dict['alpha_true']:.6f}",
        "delta_rmse": f"{distance_dict['delta_rmse']:.4f}",
        "delta_rmse_over_sd": f"{distance_dict['delta_rmse_over_sd']:.4f}",
        "delta_corr": f"{distance_dict['delta_corr']:.4f}",
        "qbar_rmse": f"{distance_dict['qbar_rmse']:.4f}",
        "qbar_rmse_over_sd": f"{distance_dict['qbar_rmse_over_sd']:.4f}",
        "qbar_corr": f"{distance_dict['qbar_corr']:.4f}",
        "build_time_s": f"{build_time:.2f}",
        "solve_time_s": f"{solve_time:.2f}",
        "total_time_s": f"{total_time:.2f}",
        "per_iter_s": f"{solve_time / max(nit, 1):.3f}",
        "memory_gb": f"{mem_gb:.1f}",
        "eq_converged": eq_converged_str,
        "eq_converged_pct": eq_converged_pct,
        "out_dir": str(args.out_dir),
    }

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(log_row)

    print(f"Run log appended to {log_path}")


if __name__ == "__main__":
    main()
