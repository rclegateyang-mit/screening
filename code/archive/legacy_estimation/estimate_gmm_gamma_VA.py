#!/usr/bin/env python3
"""
GMM Estimation of (γ, V, A) using Worker-Level Data

This script extends estimate_gmm_gamma_V.py to include firm productivity A_j
as free parameters in θ. The parameter vector is:
    θ = [ γ, V_nat(1..J), A_nat(1..J) ]

Key changes vs. estimate_gmm_gamma_V.py:
- Parse extended θ containing γ, V, and A.
- Build a probability evaluator that uses the A_j guess to recompute cutoffs c.
- Compute Chamberlain optimal instruments at θ0 where A is initialized from
  equilibrium_firms.csv (unbounded for V and A, with γ ∈ [0,1]).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from .helpers import (
        build_choice_matrix,
        chamberlain_instruments_numeric,
        compute_choice_probabilities,
        compute_cutoffs,
        compute_order_maps,
        compute_worker_firm_distances,
        naive_theta_guess_gamma_V_A,
        read_firms_data,
        read_parameters,
        read_workers_data,
        suffix_sum,
    )
except ImportError:  # pragma: no cover - Allow running as script
    sys.path.append(str(Path(__file__).resolve().parent))
    from helpers import (  # type: ignore
        build_choice_matrix,
        chamberlain_instruments_numeric,
        compute_choice_probabilities,
        compute_cutoffs,
        compute_order_maps,
        compute_worker_firm_distances,
        naive_theta_guess_gamma_V_A,
        read_firms_data,
        read_parameters,
        read_workers_data,
        suffix_sum,
    )


def parse_theta0_extended(args, J: int, V0_nat: np.ndarray, A0_nat: np.ndarray) -> np.ndarray:
    """
    Parse θ0 from CLI arguments or construct baseline.

    θ = [γ, V_nat(1..J), A_nat(1..J)] with length 1 + 2J.

    If not provided, build baseline with γ0=0.5, V0_nat from data, and A0_nat
    from equilibrium_firms.csv.
    """
    K_expected = 1 + 2 * J

    # 1) CLI list
    if args.theta0_list is not None:
        theta0_str = args.theta0_list.replace("\n", " ").replace("\t", " ")
        values: list[float] = []
        for part in theta0_str.split(','):
            values.extend([float(x.strip()) for x in part.split() if x.strip()])
        theta0 = np.asarray(values, dtype=float)
        if theta0.size != K_expected:
            raise ValueError(f"θ0 length {theta0.size} != 1+2J={K_expected}. Provide γ, then V(1..J), then A(1..J).")
        return theta0

    # 2) CLI file
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
            raise ValueError(f"Unsupported file format: {file_path}. Use .json or .csv")

        if theta0.size != K_expected:
            raise ValueError(f"θ0 length {theta0.size} != 1+2J={K_expected}. Provide γ, then V(1..J), then A(1..J).")
        return theta0

    # 3) Baseline θ0
    gamma0 = 0.05
    theta0 = np.concatenate(([gamma0], V0_nat, A0_nat))
    return theta0


def create_probability_evaluator_gamma_V_A(
    V_baseline_nat: np.ndarray,
    distances_nat: np.ndarray,
    w_nat: np.ndarray,
    Y_nat: np.ndarray,
    A_baseline_nat: np.ndarray,
    c_nat: np.ndarray,
    x_skill: np.ndarray,
    firm_ids: np.ndarray,
    beta: float,
    phi: float,
    mu_a: float,
    sigma_a: float,
) -> Callable:
    """
    Build a probability evaluator that accepts θ = [γ, V_nat(1..J), A_nat(1..J)].

    The evaluator will:
    - Split θ into (γ, V_nat, A_nat)
    - Recompute cutoffs c using the guessed A_nat via compute_cutoffs(w, Y, A, β)
    - Compute choice probabilities using compute_choice_probabilities with V_nat
      overriding the baseline V.
    """
    J = len(firm_ids)

    def probs_worker_level(theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).ravel()
        K_expected = 1 + 2 * J
        if theta.size != K_expected:
            raise ValueError(f"θ length {theta.size} != 1+2J={K_expected}. Expected [γ, V(1..J), A(1..J)].")

        gamma = float(theta[0])
        V_nat = theta[1:1 + J]
        A_nat = theta[1 + J:]

        # Recompute cutoffs using the guessed A
        c_from_A = compute_cutoffs(w_nat, Y_nat, A_nat, beta)

        # Use compute_choice_probabilities with V_nat override and A_nat
        return compute_choice_probabilities(
            gamma,
            V_baseline_nat,           # baseline V (unused when V_nat is provided)
            distances_nat,
            w_nat, Y_nat, A_nat,      # pass guessed A
            c_from_A,                 # not used internally; compute_choice_probabilities recomputes c
            x_skill,
            firm_ids, beta, phi, mu_a, sigma_a,
            V_nat=V_nat,
        )

    return probs_worker_level


def chamberlain_moments_and_scores(theta, G_theta0, prob_eval, Y_full):
    """
    Chamberlain moments using pre-computed instruments G(θ0).

    Parameters:
    -----------
    theta : np.ndarray
        (K,) vector (γ, V₁..V_J, A₁..A_J)
    G_theta0 : np.ndarray
        (N, J, K) array with s_ij(θ0) = ∂θ log(P_ij/P_i0)|_{θ0}
    prob_eval : callable
        Function returning P_full(θ) with shape (N, J+1) (outside in col 0)
    Y_full : np.ndarray
        (N, J+1) one-hot including outside
        
    Returns:
    --------
    m_vec : np.ndarray
        (K,) sample average moment
    Psi : np.ndarray
        (N, K) per-observation moment contributions
    """
    theta = np.asarray(theta, float).ravel()
    P_full = prob_eval(theta)              # (N, J+1)

    assert P_full.ndim == 2 and P_full.shape[1] == Y_full.shape[1]
    P_full = np.clip(P_full, 1e-300, 1.0)

    N, J_plus_1 = P_full.shape
    J = J_plus_1 - 1
    K = len(theta)

    P_firms = P_full[:, 1:]
    Y_firms = Y_full[:, 1:]
    R = Y_firms - P_firms

    # Vectorize: Psi[n, k] = Σ_j R[n, j] * G_theta0[n, j, k]
    # Equivalent to batched dot-product over j for each (n, k)
    # Shapes: R (N,J), G_theta0 (N,J,K) -> Psi (N,K)
    Psi = np.einsum('nj,njk->nk', R, G_theta0)

    m_vec = Psi.mean(axis=0)
    return m_vec, Psi


def gmm_obj_theta(theta, G_theta0, prob_eval, Y_full, W=None):
    """GMM objective with Chamberlain moments."""
    m, _ = chamberlain_moments_and_scores(theta, G_theta0, prob_eval, Y_full)
    if W is None:
        return float(m @ m)
    return float(m @ (W @ m))


def create_gmm_objective(G_theta0, prob_eval, Y_full, W=None):
    def gmm_obj_theta_closure(theta: np.ndarray) -> float:
        return gmm_obj_theta(theta, G_theta0, prob_eval, Y_full, W)
    return gmm_obj_theta_closure


def optimize_theta(G_theta0: np.ndarray, prob_eval: Callable, Y_full: np.ndarray,
                  theta0: np.ndarray, J: int, N: int) -> Dict:
    """
    Joint optimization over θ = (γ, V_1..V_J, A_1..A_J) using Chamberlain moments.
    """
    start_time = time.time()

    K = len(theta0)
    print(f"Initial θ₀: γ₀={theta0[0]:.4f}, V₀ range=[{np.min(theta0[1:1+J]):.3f}, {np.max(theta0[1:1+J]):.3f}], "
          f"A₀ range=[{np.min(theta0[1+J:]):.3f}, {np.max(theta0[1+J:]):.3f}]")
    print(f"θ₀ shape: {theta0.shape} (1 + 2J with J={J})")

    # Bounds: γ ∈ [0,1], remaining K-1 free (V_j and A_j unbounded)
    bounds = [(0.0, 1.0)] + [(-np.inf, np.inf)] * (K - 1)

    # Objective
    obj = create_gmm_objective(G_theta0, prob_eval, Y_full, W=None)
    obj_start = obj(theta0)
    print(f"Initial objective: Q(θ₀) = {obj_start:.6f}")

    print("Optimizing θ using BFGS...")
    result = minimize(
        obj,
        theta0,
        method="L-BFGS-B",
        bounds = bounds,
        options={"maxiter": 50000},
    )

    theta_hat = result.x
    obj_hat = result.fun

    gamma_hat = theta_hat[0]
    V_hat = theta_hat[1:1+J]
    A_hat = theta_hat[1+J:]

    elapsed_time = time.time() - start_time

    bounds_hit = (gamma_hat <= 0.001) or (gamma_hat >= 0.999)

    print("Optimization complete:")
    print(f"  θ₀: γ₀={theta0[0]:.4f}, Q₀={obj_start:.6f}")
    print(f"  θ̂: γ̂={gamma_hat:.4f}, Q̂={obj_hat:.6f}")
    print(f"  V̂ range: [{np.min(V_hat):.3f}, {np.max(V_hat):.3f}]")
    print(f"  Â range: [{np.min(A_hat):.3f}, {np.max(A_hat):.3f}]")
    print(f"  Bounds hit (γ): {bounds_hit}")
    print(f"  Converged: {result.success}")
    print(f"  Time: {elapsed_time:.2f}s")

    return {
        'theta_hat': theta_hat.tolist(),
        'objective': float(obj_hat),
        'theta0_used': theta0.tolist(),
        'instrument_file': "chamberlain_G_theta0.npz",
    }


def main():
    parser = argparse.ArgumentParser(description='GMM estimation of (γ, V, A) using worker-level data')
    parser.add_argument('--workers_path', type=str, default='output/workers_dataset.csv',
                       help='Path to workers dataset CSV')
    parser.add_argument('--firms_path', type=str, default='output/equilibrium_firms.csv',
                       help='Path to equilibrium firms CSV')
    parser.add_argument('--params_path', type=str, default='output/parameters_effective.csv',
                       help='Path to parameters CSV')
    parser.add_argument('--out_dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed for reproducibility')
    parser.add_argument('--theta0_list', type=str, default=None,
                       help='Comma- or space/newline-separated numbers for θ0 (K=1+2J long)')
    parser.add_argument('--theta0_file', type=str, default=None,
                       help='JSON or CSV file with θ0 array (either {"theta":[...]} or single row)')
    parser.add_argument('--two_step', action='store_true',
                       help='[Deprecated] Enable two-step Chamberlain instruments (use --k_step=2)')
    parser.add_argument('--k_step', type=int, default=1,
                       help='K-step GMM with Chamberlain instruments (K≥1). K=1 uses instruments at θ₀ only.')
    parser.add_argument('--theta0_from_helper', action='store_true',
                       help='Build θ0 via helpers.naive_theta_guess_gamma_V_A using worker choices and firms CSV (Y,L).')

    args = parser.parse_args()

    print("GMM Estimation of (γ, V, A) using Worker-Level Data")
    print("=" * 60)

    # Load params and data
    print("Reading data...")
    params = read_parameters(args.params_path)
    firm_ids, w, Y, A, xi, loc_firms, c = read_firms_data(args.firms_path)
    x_skill, ell_x, ell_y, chosen_firm = read_workers_data(args.workers_path)

    # Extract parameters
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 0.5)
    phi = params.get('varphi', params.get('phi', 1.0))
    mu_a = params.get('mu_a', 0.0)
    sigma_a = params.get('sigma_a', 0.12)
    # True gamma (if available from parameters file)
    gamma_true = params.get('gamma', None)

    print(f"Parameters: α={alpha:.3f}, β={beta:.3f}, φ={phi:.3f}, μ_a={mu_a:.3f}, σ_a={sigma_a:.3f}")

    # Dimensions
    N = len(x_skill)
    J = len(w)
    print(f"Data: N={N} workers, J={J} firms")

    # Baseline V and A from data (natural order)
    V0_nat = alpha * np.log(w) + xi
    A0_nat = A.copy()
    print(f"Baseline V range: [{np.min(V0_nat):.3f}, {np.max(V0_nat):.3f}]")
    print(f"Baseline A range: [{np.min(A0_nat):.3f}, {np.max(A0_nat):.3f}]")
    if gamma_true is not None:
        try:
            print(f"True γ (from parameters): {float(gamma_true):.4f}")
        except Exception:
            print("True γ available in parameters but not a float; skipping display.")

    # Distances
    print("Computing worker-firm distances...")
    distances = compute_worker_firm_distances(ell_x, ell_y, loc_firms)

    # θ0
    if args.theta0_from_helper:
        print("Building θ0 from helper (naive MNL for γ,V and A from logY − (1−β)·log(x̄·L))...")
        theta0 = naive_theta_guess_gamma_V_A(
            x_skill=x_skill,
            ell_x=ell_x,
            ell_y=ell_y,
            chosen_firm=chosen_firm,
            loc_firms=loc_firms,
            firm_ids=firm_ids,
            beta=beta,
            firms_csv_path=args.firms_path,
        )
    else:
        print("Parsing θ0 (expects [γ, V(1..J), A(1..J)])...")
        theta0 = parse_theta0_extended(args, J, V0_nat, A0_nat)
    K = len(theta0)
    print(f"θ0 length: {K} (should be 1 + 2J = {1 + 2*J})")
    print(f"θ0 preview: γ0={theta0[0]:.4f}, V0_1={theta0[1]:.3f}, A0_1={theta0[1+J]:.3f}")

    # Probability evaluator that accepts extended θ
    print("Creating probability evaluator over (γ, V, A)...")
    probs_evaluator = create_probability_evaluator_gamma_V_A(
        V_baseline_nat=V0_nat,
        distances_nat=distances,
        w_nat=w, Y_nat=Y,
        A_baseline_nat=A0_nat,
        c_nat=c,
        x_skill=x_skill,
        firm_ids=firm_ids,
        beta=beta, phi=phi, mu_a=mu_a, sigma_a=sigma_a,
    )

    # Choice matrix including outside option
    print("Building choice matrix...")
    Y_choices = build_choice_matrix(chosen_firm, firm_ids)

    # Bounds for approx_derivative and sanity checks
    print("Building parameter bounds...")
    lb = np.full(K, -np.inf)
    ub = np.full(K, np.inf)
    lb[0] = 0.0
    ub[0] = 1.0
    bounds = (np.array(lb), np.array(ub))
    print(f"Bounds: γ ∈ [{lb[0]:.1f}, {ub[0]:.1f}], (V_j, A_j) free")

    # Determine number of K-steps
    k_steps = int(args.k_step) if args.k_step and args.k_step > 0 else 1
    if args.two_step and k_steps == 1:
        print("[Note] --two_step provided; using k_step=2. Use --k_step going forward.")
        k_steps = 2
    print(f"K-step Chamberlain GMM: K={k_steps}")

    # Storage for per-step results
    step_results = []
    G_current = None
    theta_current = theta0.copy()

    # Ensure out_dir exists
    os.makedirs(args.out_dir, exist_ok=True)

    for s in range(1, k_steps + 1):
        if s == 1:
            print("Computing Chamberlain instruments at θ₀...")
            G_current = chamberlain_instruments_numeric(theta_current, probs_evaluator, bounds=bounds)
            # Persist instruments
            np.savez(os.path.join(args.out_dir, "chamberlain_G_theta0.npz"),
                     G=G_current, theta0=theta_current, shape=np.array(G_current.shape))
            print("Chamberlain instruments saved to chamberlain_G_theta0.npz")
        else:
            print(f"Step {s}: Recomputing Chamberlain instruments at θ_{s-1}...")
            G_current = chamberlain_instruments_numeric(theta_current, probs_evaluator, bounds=bounds)
            np.savez(os.path.join(args.out_dir, f"chamberlain_G_theta{s-1}.npz"),
                     G=G_current, theta_prev=theta_current, shape=np.array(G_current.shape))
            print(f"Step {s} instruments saved to chamberlain_G_theta{s-1}.npz")

        # Basic checks
        assert G_current.shape == (N, J, K), f"Expected shape (N={N}, J={J}, K={K}), got {G_current.shape}"
        assert np.isfinite(G_current).all(), "G contains non-finite values"

        # Objective and quick diagnostics
        gmm_obj_theta_closure = create_gmm_objective(G_theta0=G_current, prob_eval=probs_evaluator, Y_full=Y_choices, W=None)
        test_obj_theta0 = gmm_obj_theta_closure(theta_current)
        theta_test = theta_current.copy(); theta_test[0] = min(0.1, max(0.0, theta_current[0]))
        test_obj_theta = gmm_obj_theta_closure(theta_test)
        print(f"Test objective at θ_{s-1}: {test_obj_theta0:.6f}; at γ tweak: {test_obj_theta:.6f}")

        # Optimize
        print(f"Optimizing θ at step {s}...")
        res_s = optimize_theta(G_current, probs_evaluator, Y_choices, theta_current, J, N)
        theta_hat_s = np.asarray(res_s['theta_hat'])
        obj_hat_s = float(res_s['objective'])
        m_norm_s = np.linalg.norm(chamberlain_moments_and_scores(theta_hat_s, G_current, probs_evaluator, Y_choices)[0])
        print(f"[Chamb Step {s}] ||m(θ̂_{s})||₂={m_norm_s:.6f}, obj={obj_hat_s:.6f}, γ̂={theta_hat_s[0]:.4f}")

        step_results.append({
            'step': s,
            'theta_hat': theta_hat_s.tolist(),
            'objective': obj_hat_s,
        })
        theta_current = theta_hat_s

    # Final estimates from last step
    theta_hat = theta_current
    obj_hat = step_results[-1]['objective']
    gamma_hat = theta_hat[0]
    V_hat = theta_hat[1:1+J]
    A_hat = theta_hat[1+J:]

    m_norm = np.linalg.norm(chamberlain_moments_and_scores(theta_hat, G_current, probs_evaluator, Y_choices)[0])
    print(f"[Chamb Final] N={N}, J={J}, K={K}, ||m(θ̂)||₂={m_norm:.6f}, obj={obj_hat:.6f}, γ̂={gamma_hat:.4f}, "
          f"||V̂||∞={np.max(np.abs(V_hat)):.3f}, ||Â||∞={np.max(np.abs(A_hat)):.3f}")

    # Compute standard errors using robust mode
    print("\nComputing GMM standard errors...")
    
    # Create moments function for standard errors
    def moments_fn(th):
        # Use final-step instruments fixed at θ̂ for robust SEs
        return chamberlain_moments_and_scores(th, G_current, probs_evaluator, Y_choices)
    
    # Import compute_gmm_standard_errors from helpers
    from helpers import compute_gmm_standard_errors
    
    # Compute standard errors using robust mode
    se_results = compute_gmm_standard_errors(
        theta_hat,
        moments_fn,
        mode="robust",
        bounds=bounds,
        h_abs=1e-5,
        h_rel=1e-6,
        ridge=1e-12
    )
    
    # Print standard errors
    print("Standard Errors:")
    print(f"  γ̂: {theta_hat[0]:.6f} ± {se_results['se'][0]:.6f}")
    for j in range(len(V_hat)):
        print(f"  V̂_{j+1}: {theta_hat[j+1]:.6f} ± {se_results['se'][j+1]:.6f}")
    for j in range(len(A_hat)):
        print(f"  Â_{j+1}: {theta_hat[1+J+j]:.6f} ± {se_results['se'][1+J+j]:.6f}")

    # Compute implied market shares at θ̂ and observed shares
    print("\nComputing implied market shares at θ̂...")
    P_hat_full = probs_evaluator(theta_hat)  # (N, J+1)
    implied_shares_vec = P_hat_full.mean(axis=0)
    observed_shares_vec = Y_choices.mean(axis=0)
    # Map firm columns 1..J to firm_ids
    implied_by_firm = {int(firm_ids[j-1]): float(implied_shares_vec[j]) for j in range(1, J+1)}
    observed_by_firm = {int(firm_ids[j-1]): float(observed_shares_vec[j]) for j in range(1, J+1)}
    print(f"Outside share (implied): {implied_shares_vec[0]:.4f}; inside sum: {implied_shares_vec[1:].sum():.4f}")
    
    # Export results
    output_json = {
        "theta_hat": theta_hat.tolist(),
        "objective": float(obj_hat),
        "theta0_used": theta0.tolist(),
        "k_step": k_steps,
        "steps": step_results,
        "standard_errors": se_results['se'].tolist(),
        "vcov_matrix": se_results['vcov'].tolist(),
        "se_mode": se_results['mode'],
        "N": se_results['N'],
        "K": se_results['K']
    }
    # Include true parameters in the estimates JSON for convenience
    try:
        output_json["true_params"] = {
            "gamma": (None if gamma_true is None else float(gamma_true)),
            "V": V0_nat.tolist(),
            "A": A0_nat.tolist(),
            "firm_ids": [int(x) for x in firm_ids.tolist()],
            "notes": "V and A reported in natural firm order matching firm_ids."
        }
    except Exception:
        # Keep output robust even if something unexpected occurs
        pass
    # Attach implied and observed market shares
    output_json["implied_shares"] = {
        "vector": [float(x) for x in implied_shares_vec.tolist()],
        "outside": float(implied_shares_vec[0]),
        "by_firm_id": {str(k): v for k, v in implied_by_firm.items()},
        "note": "vector order: [outside, firm_ids in natural order]"
    }
    output_json["observed_shares"] = {
        "vector": [float(x) for x in observed_shares_vec.tolist()],
        "outside": float(observed_shares_vec[0]),
        "by_firm_id": {str(k): float(v) for k, v in observed_by_firm.items()}
    }
    output_json["two_step_enabled"] = (k_steps == 2 and args.two_step)
    json_path = os.path.join(args.out_dir, "gmm_gamma_VA_estimates.json")
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Results exported to {json_path}")

    # Also write a separate JSON file containing only the true parameters (γ, V, A)
    true_out = {
        "gamma": (None if gamma_true is None else float(gamma_true)),
        "V": V0_nat.tolist(),
        "A": A0_nat.tolist(),
        "firm_ids": [int(x) for x in firm_ids.tolist()],
        "description": "True/observed parameters used to generate data. V = alpha*log(w) + xi."
    }
    true_json_path = os.path.join(args.out_dir, "gmm_gamma_VA_truth.json")
    with open(true_json_path, 'w') as f:
        json.dump(true_out, f, indent=2)
    print(f"True parameters exported to {true_json_path}")

    # Optional: separate JSON with market shares only
    shares_out = {
        "implied": {
            "vector": [float(x) for x in implied_shares_vec.tolist()],
            "outside": float(implied_shares_vec[0]),
            "by_firm_id": {str(k): v for k, v in implied_by_firm.items()}
        },
        "observed": {
            "vector": [float(x) for x in observed_shares_vec.tolist()],
            "outside": float(observed_shares_vec[0]),
            "by_firm_id": {str(k): float(v) for k, v in observed_by_firm.items()}
        },
        "firm_ids": [int(x) for x in firm_ids.tolist()],
        "note": "Shares include outside at index 0; firm columns follow natural firm_ids order."
    }
    shares_json_path = os.path.join(args.out_dir, "gmm_gamma_VA_shares.json")
    with open(shares_json_path, 'w') as f:
        json.dump(shares_out, f, indent=2)
    print(f"Market shares exported to {shares_json_path}")

    print("\n✅ Optimization finished and results exported")


if __name__ == '__main__':
    main()
