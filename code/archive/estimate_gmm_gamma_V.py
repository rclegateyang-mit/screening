#!/usr/bin/env python3
"""
GMM Estimation of γ using Worker-Level Data

This script estimates the distance decay parameter γ ∈ [0,1] using GMM
with worker-level choice data and firm equilibrium information.
"""

import argparse
import csv
import json
import os
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize
from scipy.optimize import approx_fprime
from typing import Dict, Tuple, Callable
import matplotlib.pyplot as plt

# Add code directory to path for imports
sys.path.append(str(Path(__file__).parent))
from helpers import (read_parameters, read_firms_data, read_workers_data,
                     compute_worker_firm_distances, compute_cutoffs, suffix_sum,
                     compute_choice_probabilities, create_probability_evaluator, build_choice_matrix,
                     compute_order_maps, chamberlain_instruments_numeric, compute_gmm_standard_errors)


def parse_theta0(args, J) -> np.ndarray:
    """
    Parse θ0 from CLI arguments or construct baseline.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command line arguments
    J : int
        Number of firms
        
    Returns:
    --------
    theta0 : np.ndarray
        Initial parameter vector θ0 with length 1+J
    """
    # If args.theta0_list present: split on commas/whitespace, map to float
    if args.theta0_list is not None:
        # Split on commas and whitespace, filter out empty strings
        theta0_str = args.theta0_list.replace('\n', ' ').replace('\t', ' ')
        # First split on commas, then on whitespace
        theta0_list = []
        for part in theta0_str.split(','):
            theta0_list.extend([float(x.strip()) for x in part.split() if x.strip()])
        theta0 = np.array(theta0_list, dtype=float)
        
    # Else if args.theta0_file present: read JSON or CSV
    elif args.theta0_file is not None:
        file_path = args.theta0_file
        if file_path.endswith('.json'):
            # JSON: expects {"theta":[...]}
            with open(file_path, 'r') as f:
                data = json.load(f)
            theta0 = np.array(data['theta'], dtype=float)
        elif file_path.endswith('.csv'):
            # CSV: read first row, flatten numbers
            df = pd.read_csv(file_path)
            theta0 = df.iloc[0].values.astype(float)
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Use .json or .csv")
            
    # Else: construct θ0 = [γ0] + list(V0) using current baseline
    else:
        # This will be constructed later in the main flow
        # For now, return None to indicate baseline should be used
        return None
    
    # Validate: len(theta0) == 1 + J
    if len(theta0) != 1 + J:
        raise ValueError(f"θ0 length {len(theta0)} != 1+J={1+J}. Check that V has one entry per firm.")
    
    return theta0


def split_theta(theta: np.ndarray, J: int) -> tuple[float, np.ndarray]:
    """Split parameter vector θ into gamma and V_nat."""
    theta = np.asarray(theta, dtype=np.float64).ravel()
    assert theta.size == 1 + J, f"theta length {theta.size} != 1+J"
    gamma = float(theta[0])
    V_nat = theta[1:].astype(np.float64)
    return gamma, V_nat


def moments_and_scores(theta: np.ndarray, probs_worker_level: Callable, 
                      Y_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build UNCONDITIONAL moments with outside option.
    
    Parameters:
    -----------
    theta : np.ndarray
        Parameter vector θ[0]=gamma, θ[1:]=V_nat
    probs_worker_level : Callable
        Function returning choice probabilities P(θ)  
    Y_full : np.ndarray
        Choice matrix N×(J+1) with outside column at index 0
        
    Returns:
    --------
    m_vec : np.ndarray
        Moment vector (J+1,) 
    scores : np.ndarray
        Individual score contributions (N×(J+1))
    """
    # Get model probabilities P_full: N×(J+1) matching column order
    P_full = probs_worker_level(theta)  # (N, J+1) including outside option
    
    # Individual scores: Y_full - P_full
    scores = Y_full - P_full  # (N×(J+1))
    
    # Moment vector: average of scores
    m_vec = scores.mean(axis=0)  # ((J+1),)
    
    return m_vec, scores





def create_gmm_objective(G_theta0, prob_eval, Y_full, W=None):
    """
    Create GMM objective function using Chamberlain moments.
    
    Parameters:
    -----------
    G_theta0 : np.ndarray
        (N, J, K) Chamberlain instruments pre-computed at θ0
    prob_eval : callable
        Function returning P_full(θ) with shape (N, J+1)
    Y_full : np.ndarray
        (N, J+1) choice matrix including outside option
    W : np.ndarray, optional
        (K, K) weighting matrix. If None, uses identity.
        
    Returns:
    --------
    gmm_obj_theta_closure : Callable
        Objective function that takes θ as input
    """
    def gmm_obj_theta_closure(theta: np.ndarray) -> float:
        return gmm_obj_theta(theta, G_theta0, prob_eval, Y_full, W)
    
    return gmm_obj_theta_closure


def chamberlain_moments_and_scores(theta, G_theta0, prob_eval, Y_full):
    """
    Chamberlain moments using pre-computed instruments G(θ0).
    
    Parameters:
    -----------
    theta : np.ndarray
        (K,) vector (γ, V₁..V_J)
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
    
    # Basic guards:
    assert P_full.ndim == 2 and P_full.shape[1] == Y_full.shape[1]
    P_full = np.clip(P_full, 1e-300, 1.0)
    
    # Get dimensions
    N, J_plus_1 = P_full.shape
    J = J_plus_1 - 1  # Exclude outside option
    K = len(theta)
    
    # Extract firms-only parts:
    P_firms = P_full[:, 1:]               # (N, J)
    Y_firms = Y_full[:, 1:]               # (N, J)
    
    # Residuals:
    R = Y_firms - P_firms                 # (N, J)
    
    # Chamberlain moment per obs (K-dim): ψ_i = Σ_j R_ij * G_ij·  (dot over j)
    # Compute explicitly: Psi[n, k] = Σ_j R[n, j] * G_theta0[n, j, k]
    Psi = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            Psi[n, k] = np.sum(R[n, :] * G_theta0[n, :, k])
    
    # Average:
    m_vec = Psi.mean(axis=0)                           # (K,)
    
    return m_vec, Psi


def gmm_obj_theta(theta, G_theta0, prob_eval, Y_full, W=None):
    """
    GMM objective function using Chamberlain moments.
    
    Parameters:
    -----------
    theta : np.ndarray
        Parameter vector θ[0]=gamma, θ[1:]=V_nat
    G_theta0 : np.ndarray
        (N, J, K) Chamberlain instruments pre-computed at θ0
    prob_eval : callable
        Function returning P_full(θ) with shape (N, J+1)
    Y_full : np.ndarray
        (N, J+1) choice matrix including outside option
    W : np.ndarray, optional
        (K, K) weighting matrix. If None, uses identity.
        
    Returns:
    --------
    obj_value : float
        Objective Q(θ; W) = m(θ)ᵀ W m(θ)
    """
    m, _ = chamberlain_moments_and_scores(theta, G_theta0, prob_eval, Y_full)
    if W is None:
        return float(m @ m)
    return float(m @ (W @ m))


def create_chamberlain_gmm_objective(G_theta0, prob_eval, Y_full, W=None):
    """
    Create GMM objective function using Chamberlain moments.
    
    Parameters:
    -----------
    G_theta0 : np.ndarray
        (N, J, K) Chamberlain instruments pre-computed at θ0
    prob_eval : callable
        Function returning P_full(θ) with shape (N, J+1)
    Y_full : np.ndarray
        (N, J+1) choice matrix including outside option
    W : np.ndarray, optional
        (K, K) weighting matrix. If None, uses identity.
        
    Returns:
    --------
    gmm_obj_theta_closure : Callable
        Objective function that takes θ as input
    """
    def gmm_obj_theta_closure(theta: np.ndarray) -> float:
        return gmm_obj_theta(theta, G_theta0, prob_eval, Y_full, W)
    
    return gmm_obj_theta_closure


def optimize_theta(G_theta0: np.ndarray, prob_eval: Callable, Y_full: np.ndarray, 
                  theta0: np.ndarray, J: int, N: int) -> Dict:
    """
    Joint optimization over θ = (γ, V_1, ..., V_J) using Chamberlain moments.
    
    Parameters:
    -----------
    G_theta0 : np.ndarray
        (N, J, K) Chamberlain instruments pre-computed at θ0
    prob_eval : Callable
        Function returning choice probabilities P(θ)
    Y_full : np.ndarray
        Choice matrix (N, J+1) including outside option
    theta0 : np.ndarray
        Initial parameter vector θ0
    J : int
        Number of firms
    N : int
        Number of workers
        
    Returns:
    --------
    results : Dict
        Optimization results including θ̂, objective value, and diagnostics
    """
    start_time = time.time()
    
    print(f"Initial θ₀: γ₀={theta0[0]:.4f}, V₀ range=[{np.min(theta0[1:]):.3f}, {np.max(theta0[1:]):.3f}]")
    print(f"θ₀ shape: {theta0.shape} (1 + {J} firms)")
    
    # Bounds: gamma ∈ [0,1], V_j unbounded
    bounds = [(0.0, 1.0)] + [(-np.inf, np.inf)] * J
    
    # Create GMM objective with identity weighting using Chamberlain moments
    obj = create_gmm_objective(G_theta0, prob_eval, Y_full, W=None)
    
    # Test objective at initial point
    obj_start = obj(theta0)
    print(f"Initial objective: Q(θ₀) = {obj_start:.6f}")
    
    # Optimize using L-BFGS-B
    print("Optimizing θ using L-BFGS-B...")
    result = minimize(
        obj, 
        theta0, 
        method="L-BFGS-B", 
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10}
    )
    
    theta_hat = result.x
    obj_hat = result.fun
    
    # Extract γ̂ and V̂ from θ̂
    gamma_hat = theta_hat[0]
    V_hat = theta_hat[1:]
    
    elapsed_time = time.time() - start_time
    
    # Diagnostics
    bounds_hit = (gamma_hat <= 0.001) or (gamma_hat >= 0.999)
    
    print(f"Optimization complete:")
    print(f"  θ₀: γ₀={theta0[0]:.4f}, Q₀={obj_start:.6f}")
    print(f"  θ̂: γ̂={gamma_hat:.4f}, Q̂={obj_hat:.6f}")
    print(f"  V̂ range: [{np.min(V_hat):.3f}, {np.max(V_hat):.3f}]")
    print(f"  Bounds hit: {bounds_hit}")
    print(f"  Converged: {result.success}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    return {
        'theta_hat': theta_hat.tolist(),
        'objective': float(obj_hat),
        'theta0_used': theta0.tolist(),
        'instrument_file': "chamberlain_G_theta0.npz"
    }








def save_results(results: Dict, out_dir: str):
    """Save JSON + TXT per earlier prompt."""
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    
    # Save estimates JSON
    estimates_path = out_path / 'gmm_gamma_estimates.json'
    with open(estimates_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Estimates saved to: {estimates_path}")
    
    # Save log TXT
    log_path = out_path / 'gmm_gamma_V_log.txt'
    with open(log_path, 'w') as f:
        f.write(f"GMM Estimation of θ = (γ, V_1, ..., V_J)\n")
        f.write(f"========================================\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data: N={results['N']} workers, J={results['J']} firms\n")
        f.write(f"Initial γ: {results['gamma_start']:.4f} (obj: {results['obj_start']:.6f})\n")
        f.write(f"Final γ: {results['gamma_hat']:.4f} (obj: {results['objective']:.6f})\n")
        f.write(f"V̂ range: [{min(results['V_hat']):.3f}, {max(results['V_hat']):.3f}]\n")
        f.write(f"Bounds hit: {results['bounds_hit']}\n")
        f.write(f"Converged: {results['converged']}\n")
        f.write(f"Elapsed time: {results['elapsed_time']:.2f}s\n")
        f.write(f"Final θ̂: {results['theta_hat']}\n")
    print(f"Log saved to: {log_path}")

def main():
    parser = argparse.ArgumentParser(description='GMM estimation of γ using worker-level data')
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
                       help='Comma- or space/newline-separated numbers for θ0 (K long)')
    parser.add_argument('--theta0_file', type=str, default=None,
                       help='JSON or CSV file with θ0 array (either {"theta":[...]} or single row)')



    
    args = parser.parse_args()
    
    print("GMM Estimation of γ using Worker-Level Data")
    print("=" * 50)
    
    # Load params → dict (floats)
    print("Reading data...")
    params = read_parameters(args.params_path)
    firm_ids, w, Y, A, xi, loc_firms, c = read_firms_data(args.firms_path)
    x_skill, ell_x, ell_y, chosen_firm = read_workers_data(args.workers_path)
    
    # Extract parameters
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 0.5)
    phi = params.get('varphi', params.get('phi', 1.0))  # take varphi=1 if absent
    mu_a = params.get('mu_a', 0.0)
    sigma_a = params.get('sigma_a', 0.12)
    
    print(f"Parameters: α={alpha:.3f}, β={beta:.3f}, φ={phi:.3f}, μ_a={mu_a:.3f}, σ_a={sigma_a:.3f}")
    
    # N and J from data
    N = len(x_skill)
    J = len(w)
    print(f"Data: N={N} workers, J={J} firms")
    
    # Compute V_j = α·log w_j + ξ_j
    V = alpha * np.log(w) + xi
    print(f"V range: [{np.min(V):.3f}, {np.max(V):.3f}]")
    
    # Precompute worker→firm distances D_nat (N×J)
    print("Computing worker-firm distances...")
    distances = compute_worker_firm_distances(ell_x, ell_y, loc_firms)
    print(f"Distance range: [{np.min(distances):.3f}, {np.max(distances):.3f}]")
    
    # Build order maps from current c. Reuse `compute_order_maps`
    order_maps = compute_order_maps(c)
    print(f"Order maps computed from cutoffs c")
    
    # Parse θ0 from CLI arguments or construct baseline
    print("Parsing θ0...")
    theta0 = parse_theta0(args, J)
    
    if theta0 is None:
        # Construct baseline θ0 = [γ0] + list(V0)
        gamma0 = 0.05  # Pilot value for γ
        V0_nat = alpha * np.log(w) + xi  # V0 = α*log w + ξ in natural order
        theta0 = np.concatenate(([gamma0], V0_nat))
        print(f"Using baseline θ0: γ0={gamma0:.4f}, V0 range=[{np.min(V0_nat):.3f}, {np.max(V0_nat):.3f}]")
    else:
        print(f"Using CLI-provided θ0: length K={len(theta0)}")
    
    # Print θ0 details
    K = len(theta0)
    print(f"θ0 length: {K} (1 + {J} firms)")
    print(f"θ0 first few entries: γ0={theta0[0]:.4f}, V0_1={theta0[1]:.3f}, V0_2={theta0[2]:.3f}")
    
    # Create probability evaluator
    print("Creating probability evaluator...")
    probs_evaluator = create_probability_evaluator(V, distances, w, Y, A, c, x_skill,
                                                 firm_ids, beta, phi, mu_a, sigma_a)
    
    # Build choice matrix Y_choices (N × max_firm_id+1) one-hot including outside option
    print("Building choice matrix...")
    Y_choices = build_choice_matrix(chosen_firm, firm_ids)
    
    # STEP 2: Build bounds and compute Chamberlain instruments at θ0
    print("Building parameter bounds...")
    lb = np.full(K, -np.inf)  # K = 1 + J
    ub = np.full(K, np.inf)
    lb[0] = 0.0   # γ ∈ [0, 1]
    ub[0] = 1.0
    bounds = (np.array(lb), np.array(ub))
    print(f"Bounds: γ ∈ [{lb[0]:.1f}, {ub[0]:.1f}], V_j ∈ [{lb[1]:.1f}, {ub[1]:.1f}]")
    
    # Compute Chamberlain instruments at θ0
    print("Computing Chamberlain instruments at θ0...")
    G_theta0 = chamberlain_instruments_numeric(
        theta0, probs_evaluator, bounds=bounds,
        rel_step=1e-6, abs_step=1e-5
    )
    
    # Sanity checks
    print(f"G_theta0 shape: {G_theta0.shape}")
    assert G_theta0.shape == (N, J, K), f"Expected shape (N={N}, J={J}, K={K}), got {G_theta0.shape}"
    assert np.isfinite(G_theta0).all(), "G_theta0 contains non-finite values"
    print("✅ Chamberlain instruments computed successfully")
    
    # Save to disk for reproducibility
    np.savez(os.path.join(args.out_dir, "chamberlain_G_theta0.npz"),
             G=G_theta0, theta0=theta0, shape=np.array(G_theta0.shape))
    print(f"Chamberlain instruments saved to chamberlain_G_theta0.npz")
    
    # Keep G_theta0 in memory for moment computation
    print(f"G_theta0 kept in memory, shape: {G_theta0.shape}")
    print(f"Choice matrix shape: {Y_choices.shape}")
    print(f"Choice distribution: {np.sum(Y_choices, axis=0)}")
    print(f"Outside choices: {np.sum(Y_choices[:, 0])}")
    print(f"Firm choices by firm_id: {np.sum(Y_choices[:, 1:], axis=0)}")
    print(f"Firm IDs: {firm_ids}")
    
    # STEP 5: Add minimal diagnostics to catch indexing issues
    print("\nRunning diagnostics...")
    
    # Check P at θ0: ensure probabilities sum to 1
    print("Checking probability evaluator at θ0...")
    P0 = probs_evaluator(theta0)
    row_sums = P0.sum(1)
    max_deviation = np.max(np.abs(row_sums - 1))
    print(f"Max probability row sum deviation: {max_deviation:.2e}")
    assert max_deviation <= 1e-10, f"Probabilities don't sum to 1: max deviation {max_deviation}"
    print("✅ Probability row sums check passed")
    
    # Dimension checks: ensure G_theta0 has correct shape
    print("Checking dimension consistency...")
    expected_shape = (Y_choices.shape[0], Y_choices.shape[1]-1, len(theta0))
    print(f"G_theta0 shape: {G_theta0.shape}")
    print(f"Expected shape: {expected_shape}")
    assert G_theta0.shape == expected_shape, f"G_theta0 shape mismatch: got {G_theta0.shape}, expected {expected_shape}"
    print("✅ Dimension consistency check passed")
    
    # Create GMM objective with identity weighting (W=None) using Chamberlain moments
    print("Creating GMM objective using Chamberlain moments...")
    gmm_obj_gamma = create_gmm_objective(G_theta0=G_theta0, prob_eval=probs_evaluator, Y_full=Y_choices, W=None)
    
    # Test objective with sample gamma
    test_gamma = 0.1
    test_obj = gmm_obj_gamma(test_gamma)
    print(f"Test objective at γ = {test_gamma}: Q = {test_obj:.6f}")
    
    # STEP 5: Optimize using optimize_theta and export to JSON
    print("\nOptimizing θ using Chamberlain moments...")
    results = optimize_theta(G_theta0, probs_evaluator, Y_choices, theta0, J, N)
    
    # Extract results
    theta_hat = np.array(results['theta_hat'])
    obj_hat = results['objective']
    
    # Print one-line summary
    gamma_hat = theta_hat[0]
    V_hat = theta_hat[1:]
    m_norm = np.linalg.norm(chamberlain_moments_and_scores(theta_hat, G_theta0, probs_evaluator, Y_choices)[0])
    print(f"[Chamb] N={N}, J={J}, K={len(theta0)}, ||m(θ̂)||₂={m_norm:.6f}, obj={obj_hat:.6f}, γ̂={gamma_hat:.4f}, ||V̂||∞={np.max(np.abs(V_hat)):.3f}")
    
    # Compute standard errors
    print("\nComputing GMM standard errors...")
    
    # Create moments function for standard errors
    def moments_fn(th):
        return chamberlain_moments_and_scores(th, G_theta0, probs_evaluator, Y_choices)
    
  
    # Compute standard errors using efficient mode (Fisher information)
    se_results = compute_gmm_standard_errors(
        theta_hat=theta_hat,
        moments_fn=moments_fn,
        mode="robust",
        chamberlain_builder=lambda th: chamberlain_instruments_numeric(th, probs_evaluator, bounds=bounds, rel_step=1e-6, abs_step=1e-5),
        prob_eval=probs_evaluator,
        Y_full=Y_choices,
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
    
    # Export results to JSON
    output_json = {
        "theta_hat": results['theta_hat'],
        "objective": results['objective'],
        "theta0_used": results['theta0_used'],
        "instrument_file": results['instrument_file'],
        "standard_errors": se_results['se'].tolist(),
        "vcov_matrix": se_results['vcov'].tolist(),
        "se_mode": se_results['mode'],
        "N": se_results['N'],
        "K": se_results['K']
    }
    
    json_path = os.path.join(args.out_dir, "gmm_gamma_V_estimates.json")
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Results exported to {json_path}")
    
    print("\n✅ STEP 5 COMPLETED: Optimization finished and results exported")


if __name__ == '__main__':
    main()
