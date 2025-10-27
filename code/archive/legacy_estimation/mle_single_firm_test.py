#!/usr/bin/env python3
"""Synthetic single-firm test harness for the JAX MLE solver.

This script mirrors the data structures of :mod:`code.estimation.run_mle_jax` but exposes
fine-grained control over the simulated environment:

- Fix a single firm's cutoff ``c`` and indirect utility ``V`` directly.
- Draw log distance and skill jointly from a bivariate normal distribution so
  distance in levels is strictly positive (lognormal).
- Generate worker choices using the same probability evaluator as the MLE solver.
- Run a constrained LBFGS search over any subset of ``{gamma, c, V}``.

Example
-------
Estimate ``gamma`` and ``V`` while holding ``c`` fixed::

    python code/mle_single_firm_test.py --fit gamma,V --cutoff 0.75 --gamma 0.08 --V 2.0

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGS
import matplotlib.pyplot as plt

try:  # Local package import when executed via ``python -m`` at repo root
    from .jax_model import enable_x64, compute_choice_probabilities_gamma_c_V_jax
    from .optimize_gmm import make_reparam
except ImportError:  # pragma: no cover - direct execution
    from jax_model import enable_x64, compute_choice_probabilities_gamma_c_V_jax  # type: ignore
    from optimize_gmm import make_reparam  # type: ignore


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def draw_joint_normal(
    n: int,
    mean_log_distance: float,
    mean_skill: float,
    std_log_distance: float,
    std_skill: float,
    corr: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw log distance and skill jointly from a bivariate normal, return levels."""

    mu_log = float(mean_log_distance)
    sigma_log = float(std_log_distance)
    mean_skill = float(mean_skill)
    std_skill = float(std_skill)

    if sigma_log < 0:
        raise ValueError("std_log_distance must be non-negative.")

    corr = float(np.clip(corr, -0.999, 0.999))
    cov = np.array(
        [
            [sigma_log ** 2, corr * sigma_log * std_skill],
            [corr * sigma_log * std_skill, std_skill ** 2],
        ]
    )
    mean_vec = np.array([mu_log, mean_skill])

    samples = rng.multivariate_normal(mean=mean_vec, cov=cov, size=int(n))
    log_dist = samples[:, 0]
    skills = samples[:, 1]
    distances = np.exp(log_dist)
    return distances, skills


def simulate_choices(theta: jnp.ndarray, X: jnp.ndarray, aux: Dict, rng: np.random.Generator) -> np.ndarray:
    """Draw worker choices according to the MLE probability evaluator."""
    P = np.asarray(compute_choice_probabilities_gamma_c_V_jax(theta, X, aux))
    u = rng.random(P.shape[0])[:, None]
    cumulative = np.cumsum(P, axis=1)
    choices = (u > cumulative).sum(axis=1).astype(np.int32)
    return choices


def _summary_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def save_binscatter(
    distances: np.ndarray,
    choices: np.ndarray,
    theta: jnp.ndarray,
    mean_skill: float,
    aux_template: Dict,
    path: Path,
    bins: int,
) -> None:
    """Save binscatter with empirical and mean-skill model shares vs distance."""

    if bins <= 0:
        raise ValueError("bins must be positive")

    distances = np.asarray(distances, dtype=float)
    choices = np.asarray(choices, dtype=int)
    indicator = (choices == 1).astype(float)

    edges = np.linspace(distances.min(), distances.max(), bins + 1)
    bin_ids = np.digitize(distances, edges[1:-1], right=False)

    x_vals: list[float] = []
    y_vals: list[float] = []
    model_vals: list[float] = []

    phi = float(aux_template.get("phi", 1.0))
    mu_a = float(aux_template.get("mu_a", 0.0))
    sigma_a = float(aux_template.get("sigma_a", 1.0))
    firm_ids = jnp.asarray([1], dtype=jnp.int32)

    theta = jnp.asarray(theta, dtype=jnp.float64)
    X_mean = jnp.asarray([mean_skill], dtype=jnp.float64)

    for b in range(bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        dist_mean = float(distances[mask].mean())
        x_vals.append(dist_mean)
        y_vals.append(float(indicator[mask].mean()))

        aux_single = {
            "D_nat": jnp.asarray([[dist_mean]], dtype=jnp.float64),
            "phi": phi,
            "mu_a": mu_a,
            "sigma_a": sigma_a,
            "firm_ids": firm_ids,
        }
        P_theory = compute_choice_probabilities_gamma_c_V_jax(theta, X_mean, aux_single)
        model_vals.append(float(P_theory[0, 1]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x_vals, y_vals, color="tab:blue", s=35, label="Empirical (bins)")
    ax.plot(x_vals, y_vals, color="tab:blue", linewidth=1.5)
    if model_vals:
        ax.plot(x_vals, model_vals, color="tab:red", linestyle="--", linewidth=1.5,
                label="Model (skill = mean)")
    ax.set_xlabel("Distance to firm")
    ax.set_ylabel("Share choosing firm")
    ax.set_title("Binscatter: choice vs distance")
    ax.grid(True, alpha=0.3)
    if model_vals:
        ax.legend()
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_aux(distances: np.ndarray, phi: float, mu_a: float, sigma_a: float) -> Dict:
    """Construct the auxiliary data dictionary consumed by the JAX evaluator."""
    distances = np.asarray(distances, dtype=np.float64).reshape(-1, 1)

    aux = {
        "D_nat": jnp.asarray(distances, dtype=jnp.float64),
        "phi": float(phi),
        "mu_a": float(mu_a),
        "sigma_a": float(sigma_a),
        "firm_ids": jnp.asarray([1], dtype=jnp.int32),
    }
    return aux


# ---------------------------------------------------------------------------
# Likelihood construction
# ---------------------------------------------------------------------------


def build_neg_log_likelihood(
    free_names: List[str],
    fixed_params: Dict[str, float],
    X: jnp.ndarray,
    choice_idx: jnp.ndarray,
    aux: Dict,
):
    """Create a JIT-compiled negative log-likelihood over the free parameters."""

    idx_gamma = free_names.index("gamma") if "gamma" in free_names else None
    idx_c = free_names.index("c") if "c" in free_names else None
    idx_V = free_names.index("V") if "V" in free_names else None

    fixed_gamma = float(fixed_params["gamma"])
    fixed_c = float(fixed_params["c"])
    fixed_V = float(fixed_params["V"])

    @jax.jit
    def neg_log_likelihood(theta_free: jnp.ndarray) -> jnp.ndarray:
        gamma_val = theta_free[idx_gamma] if idx_gamma is not None else fixed_gamma
        c_val = theta_free[idx_c] if idx_c is not None else fixed_c
        V_val = theta_free[idx_V] if idx_V is not None else fixed_V

        theta_full = jnp.array([gamma_val, V_val, c_val], dtype=jnp.float64)

        P = compute_choice_probabilities_gamma_c_V_jax(theta_full, X, aux)
        probs = jnp.take_along_axis(P, choice_idx[:, None], axis=1).squeeze(axis=1)
        safe_probs = jnp.clip(probs, 1e-12, 1.0)
        return -jnp.sum(jnp.log(safe_probs))

    return neg_log_likelihood


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic single-firm MLE test harness")
    parser.add_argument("--fit", type=str, default="gamma,c,V",
                        help="Comma-separated subset of parameters to estimate (choices: gamma,c,V)")
    parser.add_argument("--gamma", type=float, default=0.08, help="True gamma used for simulation")
    parser.add_argument("--cutoff", type=float, default=0.2, help="True firm cutoff c")
    parser.add_argument("--V", type=float, default=2.0, help="True firm indirect utility V")
    parser.add_argument("--gamma0", type=float, default=None, help="Initial guess for gamma")
    parser.add_argument("--cutoff0", type=float, default=None, help="Initial guess for cutoff c")
    parser.add_argument("--V0", type=float, default=None, help="Initial guess for V")
    parser.add_argument("--num-workers", type=int, default=4000, help="Number of workers to simulate")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--mean-distance", type=float, default=0.5,
                        help="Mean of log distance (before exponentiating)")
    parser.add_argument("--mean-skill", type=float, default=0.0, help="Mean skill in the joint normal")
    parser.add_argument("--std-distance", type=float, default=1,
                        help="Std dev of log distance (before exponentiating)")
    parser.add_argument("--std-skill", type=float, default=1.0, help="Std dev of skill draws")
    parser.add_argument("--corr", type=float, default=0.0, help="Correlation between distance and skill")
    parser.add_argument("--phi", type=float, default=1.0, help="Skill loading phi")
    parser.add_argument("--mu-a", type=float, default=0.0, help="Mean skill shock μ_a")
    parser.add_argument("--sigma-a", type=float, default=0.2, help="Std dev skill shock σ_a")
    parser.add_argument("--maxiter", type=int, default=200, help="LBFGS maximum iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="LBFGS convergence tolerance")
    parser.add_argument("--bins", type=int, default=20, help="Number of bins for the distance plot")
    parser.add_argument("--binscatter-path", type=str, default=None,
                        help="Output path for the binscatter PNG (default: output/mle_single_firm_binscatter.png)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    requested = set()
    if args.fit:
        for part in args.fit.split(','):
            token = part.strip().lower()
            if not token:
                continue
            if token in {"gamma"}:
                requested.add("gamma")
            elif token in {"c", "cutoff"}:
                requested.add("c")
            elif token in {"v"}:
                requested.add("V")
            else:
                raise ValueError(f"Unrecognised fit token '{part}'. Use any of gamma,c,V.")

    free_names = [name for name in ["gamma", "c", "V"] if name in requested]
    if not free_names:
        raise ValueError("At least one parameter must be selected in --fit.")

    enable_x64()

    rng = np.random.default_rng(args.seed)

    distances, skills = draw_joint_normal(
        n=args.num_workers,
        mean_log_distance=args.mean_distance,
        mean_skill=args.mean_skill,
        std_log_distance=args.std_distance,
        std_skill=args.std_skill,
        corr=args.corr,
        rng=rng,
    )  # defaults imply γ·E[d] ≈ 0.3 ≈ 15% of V=2.0

    aux = build_aux(
        distances=distances,
        phi=args.phi,
        mu_a=args.mu_a,
        sigma_a=args.sigma_a,
    )

    X = jnp.asarray(skills, dtype=jnp.float64)

    theta_true = jnp.array([args.gamma, args.V, args.cutoff], dtype=jnp.float64)

    choices = simulate_choices(theta_true, X, aux, rng)
    choice_idx = jnp.asarray(choices, dtype=jnp.int32)

    binscatter_path = Path(args.binscatter_path) if args.binscatter_path else (
        Path(__file__).parent.parent / "output" / "mle_single_firm_binscatter.png"
    )
    save_binscatter(
        distances=distances,
        choices=np.asarray(choices),
        theta=theta_true,
        mean_skill=float(args.mean_skill),
        aux_template=aux,
        path=binscatter_path,
        bins=int(args.bins),
    )

    dist_stats = _summary_stats(distances)
    skill_stats = _summary_stats(skills)

    fixed_params = {
        "gamma": float(args.gamma),
        "c": float(args.cutoff),
        "V": float(args.V),
    }

    params0 = {
        "gamma": float(args.gamma0) if args.gamma0 is not None else fixed_params["gamma"],
        "c": float(args.cutoff0) if args.cutoff0 is not None else fixed_params["c"],
        "V": float(args.V0) if args.V0 is not None else fixed_params["V"],
    }

    neg_loglik_theta = build_neg_log_likelihood(
        free_names=free_names,
        fixed_params=fixed_params,
        X=X,
        choice_idx=choice_idx,
        aux=aux,
    )

    theta0_vec = jnp.asarray([params0[name] for name in free_names], dtype=jnp.float64)

    lb_list: List[float] = []
    ub_list: List[float] = []
    for name in free_names:
        if name == "gamma":
            lb_list.append(0.0)
            ub_list.append(1.0)
        elif name == "c":
            lb_list.append(1e-6)
            ub_list.append(np.inf)
        elif name == "V":
            lb_list.append(-np.inf)
            ub_list.append(np.inf)
        else:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected parameter {name}")

    lb = jnp.asarray(lb_list, dtype=jnp.float64)
    ub = jnp.asarray(ub_list, dtype=jnp.float64)

    fwd, inv = make_reparam(lb, ub)

    def neg_loglik_z(z: jnp.ndarray) -> jnp.ndarray:
        theta_free = fwd(z)
        return neg_loglik_theta(theta_free)

    neg_loglik_z = jax.jit(neg_loglik_z)

    z0 = inv(theta0_vec)

    solver = LBFGS(fun=neg_loglik_z, value_and_grad=False, maxiter=int(args.maxiter), tol=args.tol)
    result = solver.run(z0)

    z_hat = result.params
    theta_free_hat = fwd(z_hat)

    est_params = fixed_params.copy()
    for idx, name in enumerate(free_names):
        est_params[name] = float(theta_free_hat[idx])

    hessian_free = np.asarray(jax.hessian(neg_loglik_theta)(theta_free_hat))
    hessian_free = 0.5 * (hessian_free + hessian_free.T)
    try:
        cov_free = np.linalg.inv(hessian_free)
    except np.linalg.LinAlgError:
        cov_free = np.linalg.pinv(hessian_free)
    cov_free = 0.5 * (cov_free + cov_free.T)
    if cov_free.size:
        se_free = np.sqrt(np.maximum(np.diag(cov_free), 0.0))
        se_map = {name: float(se) for name, se in zip(free_names, se_free)}
    else:
        se_free = np.array([])
        se_map = {}

    # Derived quantities
    theta_hat_full = jnp.array(
        [est_params["gamma"], est_params["V"], est_params["c"]],
        dtype=jnp.float64,
    )

    objective = float(neg_loglik_theta(theta_free_hat))
    nit = int(result.state.iter_num)
    grad_norm = float(jnp.linalg.norm(result.state.grad))

    P_hat = np.asarray(compute_choice_probabilities_gamma_c_V_jax(theta_hat_full, X, aux))
    avg_share_model = float(P_hat[:, 1].mean()) if P_hat.shape[1] > 1 else 0.0
    share_empirical = float((choices == 1).mean())

    print("Synthetic single-firm MLE test")
    print("-------------------------------")
    print(f"Workers simulated: {args.num_workers}")
    print(f"Objective (neg log-likelihood): {objective:.6f}")
    print(f"LBFGS iterations: {nit}")
    print(f"Gradient norm: {grad_norm:.3e}")

    def _fmt(values: Dict[str, float], names: Iterable[str]) -> str:
        parts = []
        for name in names:
            parts.append(f"{name}={values[name]:.6f}")
        return ", ".join(parts)

    true_subset = {"gamma": args.gamma, "c": args.cutoff, "V": args.V}
    init_subset = {name: params0[name] for name in free_names}
    est_subset = {name: est_params[name] for name in free_names}

    print("True params:")
    print("  " + _fmt(true_subset, ["gamma", "c", "V"]))
    print("Initial guess (free subset):")
    print("  " + _fmt(init_subset, free_names))
    print("Estimated (free subset):")
    print("  " + _fmt(est_subset, free_names))
    if se_free.size:
        print("Observed-info standard errors:")
        for name in free_names:
            print(f"  {name}: {se_map[name]:.6f}")

    print("Empirical firm share: {:.4f}".format(share_empirical))
    print("Model firm share at estimate: {:.4f}".format(avg_share_model))
    print(f"Binscatter saved to {binscatter_path}")
    print("Distance stats (mean, std, p10, p50, p90):" \
          f" {dist_stats['mean']:.3f}, {dist_stats['std']:.3f}, {dist_stats['p10']:.3f},"
          f" {dist_stats['p50']:.3f}, {dist_stats['p90']:.3f}")
    print("Skill stats (mean, std, p10, p50, p90):" \
          f" {skill_stats['mean']:.3f}, {skill_stats['std']:.3f}, {skill_stats['p10']:.3f},"
          f" {skill_stats['p50']:.3f}, {skill_stats['p90']:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
