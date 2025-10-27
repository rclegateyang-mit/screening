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
from scipy.optimize import minimize_scalar
from scipy.optimize import approx_fprime
from scipy.special import ndtr
from typing import Dict, Tuple, Callable
import matplotlib.pyplot as plt

# Add code directory to path for imports
sys.path.append(str(Path(__file__).parent))
from helpers import compute_order_maps


def read_parameters(path: str) -> Dict[str, float]:
    """Load params → dict (floats)."""
    df = pd.read_csv(path)
    params = {}
    for _, row in df.iterrows():
        try:
            params[row['parameter']] = float(row['value'])
        except (ValueError, TypeError):
            # Skip non-numeric parameters
            continue
    return params


def read_firms_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From firms CSV, read arrays (firm_ids, w, Y, A or logA, xi, loc_firms, c)."""
    df = pd.read_csv(path)
    
    # Sort by firm_id to ensure consistent ordering
    df = df.sort_values('firm_id').reset_index(drop=True)
    
    # Extract firm IDs (economic firm identifiers: 1, 2, 3, ..., J)
    firm_ids = df['firm_id'].values
    
    # Extract arrays
    w = df['w'].values  # wages
    xi = df['xi'].values  # firm fixed effects
    
    # Handle A (TFP) - could be A or logA
    if 'A' in df.columns:
        A = df['A'].values
    elif 'logA' in df.columns:
        A = np.exp(df['logA'].values)
    else:
        raise ValueError("Neither 'A' nor 'logA' column found in firms data")
    
    # Handle Y (output) - could be Y or computed from other fields
    if 'Y' in df.columns:
        Y = df['Y'].values
    else:
        raise ValueError("'Y' column not found in firms data")
    
    # Firm locations
    loc_firms = df[['x', 'y']].values
    
    # Cutoffs c
    c = df['c'].values
    
    return firm_ids, w, Y, A, xi, loc_firms, c


def read_workers_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read worker data."""
    df = pd.read_csv(path)
    
    x_skill = df['x_skill'].values
    ell_x = df['ell_x'].values
    ell_y = df['ell_y'].values
    chosen_firm = df['chosen_firm'].values
    
    return x_skill, ell_x, ell_y, chosen_firm


def compute_worker_firm_distances(ell_x: np.ndarray, ell_y: np.ndarray, 
                                loc_firms: np.ndarray) -> np.ndarray:
    """Precompute worker→firm distances D_nat (N×J) using worker (ell_x, ell_y) and firm locations (x_j, y_j)."""
    N = len(ell_x)
    J = len(loc_firms)
    
    # Worker locations (N, 2)
    worker_locs = np.column_stack([ell_x, ell_y])
    
    # Firm locations (J, 2)
    firm_locs = loc_firms
    
    # Compute distances (N, J)
    distances = np.zeros((N, J))
    for i in range(N):
        for j in range(J):
            distances[i, j] = np.linalg.norm(worker_locs[i] - firm_locs[j])
    
    return distances


def compute_cutoffs(w: np.ndarray, Y: np.ndarray, A: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute cutoffs from firms (natural order):
    log c_j = log w_j − log A_j − log(1−β) + (β/(1−β))·(log Y_j − log A_j)
    c_j = exp(log c_j)
    """
    # Numerical guards: ensure positive values for log operations
    w_safe = np.maximum(w, 1e-10)
    Y_safe = np.maximum(Y, 1e-10)
    A_safe = np.maximum(A, 1e-10)
    
    log_c = (np.log(w_safe) - np.log(A_safe) - np.log(1 - beta) + 
             (beta / (1 - beta)) * (np.log(Y_safe) - np.log(A_safe)))
    
    c = np.exp(log_c)
    return c


def suffix_sum(arr: np.ndarray) -> np.ndarray:
    """Compute suffix sum: sum from index j to end for each j."""
    return np.flip(np.cumsum(np.flip(arr)))


def compute_choice_probabilities(gamma: float, V: np.ndarray, distances: np.ndarray,
                               c_nat: np.ndarray, order_idx: np.ndarray,
                               x_skill: np.ndarray, firm_ids: np.ndarray, 
                               phi: float, mu_a: float, sigma_a: float,
                               distances_sorted: np.ndarray | None = None) -> np.ndarray:
    """
    Probability kernel per worker i for a given γ.
    
    Returns P (N × J), rows workers, cols natural firm IDs.
    """
    N, J = distances.shape
    
    # Assert precomputed values have correct shapes
    assert c_nat.shape == (J,), f"c_nat shape {c_nat.shape} != (J={J},)"
    assert order_idx.shape == (J,), f"order_idx shape {order_idx.shape} != (J={J},)"
    
    # 1) Use precomputed cutoffs (natural order)
    # c_nat is already provided
    
    # 2) Use precomputed order maps
    # order_idx is already provided
    # Compute inverse order and sorted cutoffs
    inv_order = np.argsort(order_idx)
    c_sorted = c_nat[order_idx]
    
    # 3) Reindex arrays to by-c order
    V_sorted = V[order_idx]
    if distances_sorted is None:
        distances_sorted = distances[:, order_idx]  # (N, J) in by-c order
    
    # 4) Compute v_ij for each worker i - VECTORIZED
    # v_0 ≡ 1 (outside), v_{ij} = exp(−γ·d_{ij} + V_{(j)}) for j≥1
    v_out = np.ones((N, 1))  # (N, 1)
    v_in = np.exp(-gamma * distances_sorted + V_sorted[None, :])  # (N, J)
    v_all = np.concatenate((v_out, v_in), axis=1)  # (N, J+1)
    
    # 5) Compute denominators denom_k(i) = Σ_{m=0}^k v_{im}
    denom = np.cumsum(v_all, axis=1)  # (N, J+1)
    
    # 6) Build p_x(i) for each worker - FULLY VECTORIZED
    # Sentinels: c_0 = −∞, c_{J+1} = +∞
    c_pad = np.concatenate([[-np.inf], c_sorted, [np.inf]])  # (J+2,)
    
    # Compute transformed skill for all workers
    s = phi * x_skill + mu_a  # (N,)
    
    # Broadcast the z-scores in one shot
    z_hi = (c_pad[1:][None, :] - s[:, None]) / sigma_a  # (N, J+1)
    z_lo = (c_pad[:-1][None, :] - s[:, None]) / sigma_a  # (N, J+1)
    
    # Use vectorized normal CDF
    p_x = ndtr(z_hi) - ndtr(z_lo)  # (N, J+1)
    
    # Clip tiny negatives and renormalize rows once (vectorized)
    p_x = np.clip(p_x, 0.0, 1.0)
    row_sums = p_x.sum(axis=1, keepdims=True)
    p_x /= np.maximum(row_sums, 1e-300)
       
    # 7) Apply suffix-sum U·(·) operator - FULLY VECTORIZED
    # q = p_x / denom (vectorized)
    q = p_x / np.maximum(denom, 1e-300)  # (N, J+1)
    
    # Suffix sums for all rows at once
    Q = np.flip(np.cumsum(np.flip(q, axis=1), axis=1), axis=1)  # (N, J+1)
    
    # Final by-c probabilities
    P_byc = v_all * Q  # (N, J+1)
    
    # 9) Map back to natural order using firm IDs
    # Natural order: index 0=outside, index firm_id=firm
    P_nat = np.zeros((N, J + 1))

    # Front-append 0 to inv_order to include outside option at index 0
    inv_order_with_outside = np.concatenate([[0], inv_order+1])
    P_all_nat = P_byc[:, inv_order_with_outside]

    return P_all_nat


def create_probability_evaluator(V: np.ndarray, distances: np.ndarray, w: np.ndarray, 
                               Y: np.ndarray, A: np.ndarray, c: np.ndarray, x_skill: np.ndarray, 
                               firm_ids: np.ndarray, beta: float, phi: float, mu_a: float, sigma_a: float) -> Callable[[float], np.ndarray]:
    """
    Return an evaluator:
    def probs_worker_level(gamma: float) -> np.ndarray:
        return P (N × J), rows workers, cols natural firm IDs.
    """
    # Precompute cutoffs and order maps once
    c_nat = compute_cutoffs(w, Y, A, beta)
    order_maps = compute_order_maps(c_nat)
    order_idx = order_maps['order_idx']
    
    # Precompute sorted distances once
    distances_sorted = distances[:, order_idx]
    
    def probs_worker_level(gamma: float) -> np.ndarray:
        start_time = time.perf_counter()
        result = compute_choice_probabilities(gamma, V, distances, c_nat, order_idx,
                                          x_skill, firm_ids, phi, mu_a, sigma_a,
                                          distances_sorted)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"Profiling: N={len(x_skill)}, J={len(firm_ids)}, elapsed={elapsed_ms:.1f}ms")
        return result
    
    return probs_worker_level


def build_choice_matrix(chosen_firm: np.ndarray, firm_ids: np.ndarray) -> np.ndarray:
    """
    Build Y (N × J+1) one-hot: Y[i, j_nat] = 1{chosen_firm_i == j_nat}.
    
    Natural order indexing:
    - Y[:, 0] = outside option (chosen_firm == 0)
    - Y[:, j] = firm with firm_id == j (chosen_firm == j for j in firm_ids)
    
    Parameters:
    -----------
    chosen_firm : np.ndarray
        Worker choices: 0=outside, firm_id=1,2,3,...,J for firms
    firm_ids : np.ndarray
        Economic firm identifiers [1, 2, 3, ..., J] from equilibrium_firms.csv
    """
    N = len(chosen_firm)
    max_firm_id = int(np.max(firm_ids))
    Y = np.zeros((N, max_firm_id + 1))  # +1 for outside option at index 0
    
    for i in range(N):
        choice = chosen_firm[i]  # 0=outside, or firm_id for firms
        if choice == 0:
            Y[i, 0] = 1.0  # Outside option
        else:
            # Find position of this firm_id in natural order
            Y[i, int(choice)] = 1.0
    
    return Y


def moments_and_scores(gamma: float, probs_worker_level: Callable[[float], np.ndarray], 
                      Y_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build UNCONDITIONAL moments with outside option.
    
    Parameters:
    -----------
    gamma : float
        Parameter value
    probs_worker_level : Callable
        Function returning choice probabilities P(γ)  
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
    P_full = probs_worker_level(gamma)  # (N, J+1) including outside option
    
    # Individual scores: Y_full - P_full
    scores = Y_full - P_full  # (N×(J+1))
    
    # Moment vector: average of scores
    m_vec = scores.mean(axis=0)  # ((J+1),)
    
    return m_vec, scores


def gmm_obj_gamma(gamma: float, probs_worker_level: Callable[[float], np.ndarray], 
                 Y_full: np.ndarray, W: np.ndarray = None) -> float:
    """
    GMM objective function with optional weighting matrix.
    
    Parameters:
    -----------
    gamma : float
        Parameter value
    probs_worker_level : Callable
        Function returning choice probabilities P(γ)
    Y_full : np.ndarray
        Choice matrix (N×(J+1)) including outside option
    W : np.ndarray, optional
        Weighting matrix ((J+1)×(J+1)). If None, uses identity.
        
    Returns:
    --------
    obj_value : float
        Objective Q(γ; W) = m(γ)ᵀ W m(γ)
    """
    m, _ = moments_and_scores(gamma, probs_worker_level, Y_full)
    
    if W is None:
        # Identity weighting
        return float(m @ m)
    
    return float(m @ (W @ m))


def create_gmm_objective(probs_worker_level: Callable[[float], np.ndarray], 
                        Y: np.ndarray, W: np.ndarray = None) -> Callable[[float], float]:
    """
    Create GMM objective function with optional weighting matrix.
    
    Parameters:
    -----------
    probs_worker_level : Callable
        Function returning choice probabilities P(γ)
    Y : np.ndarray
        Choice matrix (N, M) including outside option
    W : np.ndarray, optional
        Weighting matrix (M, M). If None, uses identity matrix.
        
    Returns:
    --------
    gmm_obj_gamma_closure : Callable
        Objective function that takes only gamma as input
    """
    def gmm_obj_gamma_closure(gamma: float) -> float:
        return gmm_obj_gamma(gamma, probs_worker_level, Y, W)
    
    return gmm_obj_gamma_closure


def grid_search_gamma(gmm_obj_gamma: Callable[[float], float], 
                     n_grid: int = 51) -> Tuple[float, float]:
    """
    If --grid_search: evaluate on a coarse grid to get a warm start.
    """
    gamma_grid = np.linspace(0.0, 1.0, n_grid)
    obj_values = np.zeros(n_grid)
    
    print(f"Grid search over {n_grid} points...")
    for i, gamma in enumerate(gamma_grid):
        obj_values[i] = gmm_obj_gamma(gamma)
        if i % 10 == 0:
            print(f"  γ = {gamma:.3f}, Q = {obj_values[i]:.6f}")
    
    # Find best point
    best_idx = np.argmin(obj_values)
    gamma_best = gamma_grid[best_idx]
    obj_best = obj_values[best_idx]
    
    print(f"Grid search best: γ = {gamma_best:.4f}, Q = {obj_best:.6f}")
    return gamma_best, obj_best


def optimize_gamma(gmm_obj_gamma: Callable[[float], float], 
                  probs_worker_level: Callable[[float], np.ndarray],
                  Y: np.ndarray, grid_search: bool = False) -> Dict:
    """
    Optimization over γ ∈ [0,1].
    Record gamma_hat, obj_hat, and m_hat at optimum.
    """
    start_time = time.time()
    
    # Grid search warm start if requested
    if grid_search:
        gamma_start, obj_start = grid_search_gamma(gmm_obj_gamma)
    else:
        gamma_start = 0.5
        obj_start = gmm_obj_gamma(gamma_start)
        print(f"Starting from γ = {gamma_start:.4f}, Q = {obj_start:.6f}")
    
    # Local optimization
    print("Local optimization...")
    result = minimize_scalar(
        gmm_obj_gamma,
        bounds=(0.0, 1.0),
        method="bounded"
    )
    
    gamma_hat = result.x
    obj_hat = result.fun
    
    # Compute final moment conditions
    P_hat = probs_worker_level(gamma_hat)
    m_hat = (Y - P_hat).mean(axis=0)
    
    elapsed_time = time.time() - start_time
    
    # Diagnostics
    bounds_hit = (gamma_hat <= 0.001) or (gamma_hat >= 0.999)
    m_inf_norm = np.max(np.abs(m_hat))
    
    print(f"Optimization complete:")
    print(f"  γ_start = {gamma_start:.4f}, Q_start = {obj_start:.6f}")
    print(f"  γ_hat = {gamma_hat:.4f}, Q_hat = {obj_hat:.6f}")
    print(f"  ||m_hat||∞ = {m_inf_norm:.6f}")
    print(f"  Bounds hit: {bounds_hit}")
    print(f"  Converged: {result.success}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    return {
        'gamma_hat': float(gamma_hat),
        'obj_hat': float(obj_hat),
        'm_hat': m_hat.tolist(),
        'gamma_start': float(gamma_start),
        'obj_start': float(obj_start),
        'm_inf_norm': float(m_inf_norm),
        'bounds_hit': bool(bounds_hit),
        'converged': bool(result.success),
        'elapsed_time': float(elapsed_time),
        'N': int(Y.shape[0]),
        'J': int(Y.shape[1])
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
    log_path = out_path / 'gmm_gamma_log.txt'
    with open(log_path, 'w') as f:
        f.write(f"GMM Estimation of γ\n")
        f.write(f"==================\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data: N={results['N']} workers, J={results['J']} firms\n")
        f.write(f"Grid search: {'Yes' if 'gamma_start' in results else 'No'}\n")
        f.write(f"Initial γ: {results['gamma_start']:.4f} (obj: {results['obj_start']:.6f})\n")
        f.write(f"Final γ: {results['gamma_hat']:.4f} (obj: {results['obj_hat']:.6f})\n")
        f.write(f"||m_hat||∞: {results['m_inf_norm']:.6f}\n")
        f.write(f"Bounds hit: {results['bounds_hit']}\n")
        f.write(f"Converged: {results['converged']}\n")
        f.write(f"Elapsed time: {results['elapsed_time']:.2f}s\n")
        f.write(f"Moment conditions: {results['m_hat']}\n")
        
        # Standard error information
        if 'se_gamma' in results:
            f.write(f"\nGMM Standard Errors:\n")
            f.write(f"Standard Error of γ: {results['se_gamma']:.6f}\n")
            f.write(f"Variance of γ: {results['var_gamma']:.8f}\n")
            f.write(f"t-statistic: {results['t_statistic']:.4f}\n")
            f.write(f"p-value (two-sided): {results['p_value']:.6f}\n")
            f.write(f"95% Confidence Interval: [{results['ci_lower_95']:.4f}, {results['ci_upper_95']:.4f}]\n")
            f.write(f"Gradient of objective: {results['gradient_obj']:.8f}\n")
            f.write(f"Hessian of objective: {results['hessian_obj']:.8f}\n")
    print(f"Log saved to: {log_path}")


def plot_gmm_objective(gmm_obj_gamma: Callable[[float], float], 
                      gamma_max: float = 0.2, 
                      grid_fineness: float = 0.05,
                      save_path: str = None,
                      gamma_hat: float = None,
                      se_gamma: float = None) -> None:
    """
    Plot the GMM objective over a range of gamma values.
    
    Parameters:
    -----------
    gmm_obj_gamma : Callable
        The GMM objective function that takes gamma and returns Q(gamma)
    gamma_max : float, default 0.5
        Maximum gamma value to plot
    grid_fineness : float, default 0.001
        Grid spacing for gamma values
    save_path : str, optional
        Path to save the plot. If None, displays the plot.
    gamma_hat : float, optional
        Estimated gamma value to display in legend
    se_gamma : float, optional
        Standard error of gamma estimate to display in legend
    """
    # Create gamma grid
    gamma_grid = np.arange(0, gamma_max + grid_fineness, grid_fineness)
    n_points = len(gamma_grid)
    
    print(f"Computing GMM objective over {n_points} points from γ=0 to γ={gamma_max}")
    print(f"Grid fineness: {grid_fineness}")
    
    # Compute objective values
    obj_values = np.zeros(n_points)
    for i, gamma in enumerate(gamma_grid):
        obj_values[i] = gmm_obj_gamma(gamma)
        if i % 100 == 0:  # Progress indicator
            print(f"  Progress: {i}/{n_points} (γ={gamma:.3f}, Q={obj_values[i]:.6f})")
    
    # Find minimum
    min_idx = np.argmin(obj_values)
    gamma_min = gamma_grid[min_idx]
    obj_min = obj_values[min_idx]
    
    print(f"Minimum found at γ={gamma_min:.4f}, Q={obj_min:.6f}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_grid, obj_values, 'b-', linewidth=1.5, label='GMM Objective Q(γ)')
    
    # Add minimum from grid search
    plt.axvline(x=gamma_min, color='r', linestyle='--', alpha=0.7, 
                label=f'Grid min at γ={gamma_min:.4f}')
    plt.axhline(y=obj_min, color='r', linestyle='--', alpha=0.7)
    
    # Add estimated gamma if provided
    if gamma_hat is not None:
        label_text = f'γ̂ = {gamma_hat:.4f}'
        if se_gamma is not None:
            label_text += f' (SE = {se_gamma:.4f})'
        plt.axvline(x=gamma_hat, color='green', linestyle='-', linewidth=2, alpha=0.8,
                    label=label_text)
    
    plt.xlabel('γ (Distance Decay Parameter)', fontsize=12)
    plt.ylabel('GMM Objective Q(γ)', fontsize=12)
    plt.title('GMM Objective Function vs Distance Decay Parameter γ', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add text box with statistics
    stats_text = f'Grid: {n_points} points\nFineness: {grid_fineness}\nGrid min Q: {obj_min:.6f}\nGrid min γ: {gamma_min:.4f}'
    if gamma_hat is not None:
        stats_text += f'\nEstimate γ̂: {gamma_hat:.4f}'
    if se_gamma is not None:
        stats_text += f'\nStd Error: {se_gamma:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    return gamma_grid, obj_values, gamma_min, obj_min


def debug_worker_choice_probabilities(worker_idx: int, gamma: float, 
                                    probs_worker_level: Callable[[float], np.ndarray],
                                    out_dir: str) -> None:
    """
    Debug dump of choice-probability calculation for a single worker.
    Calls probs_worker_level(gamma) and saves the worker's row to a CSV file.
    """
    print(f"\n{'='*80}")
    print(f"DEBUG CHOICE PROBABILITY CALCULATION - WORKER {worker_idx}")
    print(f"{'='*80}")
    
    # Call probs_worker_level to get choice probabilities for all workers
    P = probs_worker_level(gamma)
    P_worker = P[worker_idx, :]  # Get the worker_idx-th row
    
    print(f"\n1) INPUTS:")
    print(f"   Worker index: {worker_idx}")
    print(f"   γ (gamma): {gamma:.10f}")
    print(f"   Number of choices: {P_worker.shape[0]}")
    
    print(f"\n2) CHOICE PROBABILITIES FROM probs_worker_level:")
    print(f"   P_worker (row {worker_idx}): {P_worker}")
    print(f"   Sum of probabilities: {np.sum(P_worker):.10f}")
    
    # 3) Save to CSV file
    print(f"\n3) SAVING TO CSV:")
    
    # Create CSV data
    csv_data = []
    csv_data.append(['worker_idx', worker_idx])
    csv_data.append(['gamma', gamma])
    csv_data.append(['choice', 'probability'])
    
    # Add choice probabilities
    for j in range(P_worker.shape[0]):
        if j == 0:
            choice_name = 'outside'
        else:
            choice_name = f'firm_{j}'
        csv_data.append([choice_name, P_worker[j]])
    
    # Write CSV
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    
    csv_path = out_path / f'worker_{worker_idx}_choice_probs_gamma_{gamma:.6f}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    print(f"   CSV saved to: {csv_path}")
    
    print(f"\n{'='*80}")
    print(f"DEBUG DUMP COMPLETE")
    print(f"{'='*80}\n")


def compute_gmm_standard_errors(gamma_hat: float, 
                               gmm_obj_gamma: Callable[[float], float],
                               probs_worker_level: Callable[[float], np.ndarray],
                               Y: np.ndarray,
                               N: int, J: int,
                               eps: float = 1e-6) -> Dict:
    """
    Compute GMM standard errors for the gamma estimate.
    
    Parameters:
    -----------
    gamma_hat : float
        The estimated gamma parameter
    gmm_obj_gamma : Callable
        The GMM objective function
    probs_worker_level : Callable
        Function that computes choice probabilities for given gamma
    Y : np.ndarray
        Choice matrix (N × J)
    N : int
        Number of workers
    J : int
        Number of firms
    eps : float
        Step size for numerical derivatives
        
    Returns:
    --------
    Dict with standard error information
    """
    print("Computing GMM standard errors...")
    
    # 1. Compute gradient of objective function at gamma_hat
    def obj_wrapper(gamma):
        return gmm_obj_gamma(gamma)
    
    # Numerical gradient
    grad_obj = approx_fprime([gamma_hat], obj_wrapper, eps)[0]
    print(f"Gradient of objective at γ_hat: {grad_obj:.8f}")
    
    # 2. Compute Hessian (second derivative) of objective function using finite differences
    # f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h^2
    f_plus = gmm_obj_gamma(gamma_hat + eps)
    f_minus = gmm_obj_gamma(gamma_hat - eps)
    f_center = gmm_obj_gamma(gamma_hat)
    
    hessian_obj = (f_plus - 2*f_center + f_minus) / (eps**2)
    print(f"Hessian of objective at γ_hat: {hessian_obj:.8f}")
    
    # 3. Compute moment conditions and their derivatives
    m_hat, _ = moments_and_scores(gamma_hat, probs_worker_level, Y)
    
    # Compute gradient of moment conditions
    def moment_wrapper(gamma):
        m, _ = moments_and_scores(gamma, probs_worker_level, Y)
        return m
    
    # Gradient of moment conditions (J+1 × 1 matrix)
    grad_moments = approx_fprime([gamma_hat], moment_wrapper, eps)
    grad_moments = grad_moments.reshape(J+1, 1)  # (J+1, 1)
    
    print(f"Moment conditions: {m_hat}")
    print(f"Gradient of moments shape: {grad_moments.shape}")
    
    # 4. Compute sample variance of moment conditions
    # For each worker i, compute moment contribution: Y[i,:] - P[i,:]
    _, moment_contributions = moments_and_scores(gamma_hat, probs_worker_level, Y)  # (N, J+1)
    
    # Sample variance-covariance matrix of moment conditions
    S_hat = np.cov(moment_contributions.T)  # (J+1, J+1)
    print(f"Moment variance matrix S_hat shape: {S_hat.shape}")
    print(f"S_hat condition number: {np.linalg.cond(S_hat):.2e}")
    
    # 5. GMM variance formula: Var(γ_hat) = (G' * W * G)^(-1) * G' * W * S * W * G * (G' * W * G)^(-1)
    # For identity weighting W = I, this simplifies to: Var(γ_hat) = (G' * G)^(-1) * G' * S * G * (G' * G)^(-1)
    # where G = grad_moments (J+1 × 1)
    
    G = grad_moments  # (J+1, 1)
    W = np.eye(J+1)     # Identity weighting matrix (J+1, J+1)
    
    # Compute (G' * W * G)^(-1) = (G' * G)^(-1) for identity weighting
    GtWG = G.T @ W @ G  # (1, 1) scalar
    GtWG_inv = 1.0 / GtWG[0, 0] if GtWG[0, 0] != 0 else np.inf
    
    # Compute G' * W * S * W * G = G' * S * G for identity weighting
    GtWSWG = G.T @ W @ S_hat @ W @ G  # (1, 1) scalar
    
    # GMM variance
    var_gamma = GtWG_inv * GtWSWG[0, 0] * GtWG_inv / N
    
    # Standard error
    se_gamma = np.sqrt(var_gamma) if var_gamma > 0 else np.inf
    
    print(f"GMM variance of γ_hat: {var_gamma:.8f}")
    print(f"GMM standard error of γ_hat: {se_gamma:.6f}")
    
    # 6. Compute t-statistic and p-value (two-sided test)
    t_stat = gamma_hat / se_gamma if se_gamma > 0 else np.inf
    p_value = 2 * (1 - norm.cdf(abs(t_stat))) if se_gamma > 0 else 0.0
    
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value (two-sided): {p_value:.6f}")
    
    # 7. Confidence interval (95%)
    z_critical = norm.ppf(0.975)  # 1.96 for 95% CI
    ci_lower = gamma_hat - z_critical * se_gamma
    ci_upper = gamma_hat + z_critical * se_gamma
    
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return {
        'gamma_hat': gamma_hat,
        'se_gamma': se_gamma,
        'var_gamma': var_gamma,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper,
        'gradient_obj': grad_obj,
        'hessian_obj': hessian_obj,
        'moment_conditions': m_hat.tolist(),
        'gradient_moments': grad_moments.flatten().tolist(),
        'moment_variance_matrix': S_hat.tolist(),
        'N': N,
        'J': J
    }


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
    parser.add_argument('--grid_search', action='store_true',
                       help='Perform grid search before local optimization')
    parser.add_argument('--plot_objective', action='store_true',
                       help='Plot GMM objective function over gamma range')
    parser.add_argument('--plot_gamma_max', type=float, default=0.5,
                       help='Maximum gamma value for plotting (default: 0.5)')
    parser.add_argument('--plot_fineness', type=float, default=0.001,
                       help='Grid fineness for plotting (default: 0.001)')
    parser.add_argument('--debug_worker_index', type=int, default=None,
                       help='0-based row index into workers dataset for detailed debug dump')
    parser.add_argument('--debug_gamma', type=float, default=None,
                       help='Gamma value for debug dump (if absent, uses current estimate)')
    parser.add_argument('--two_step_gmm', action='store_true', default=False,
                       help='Use two-step GMM with efficient weighting matrix (default: False)')
    
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
    
    # Create probability evaluator
    print("Creating probability evaluator...")
    probs_evaluator = create_probability_evaluator(V, distances, w, Y, A, c, x_skill,
                                                 firm_ids, beta, phi, mu_a, sigma_a)
    
    # Build choice matrix Y_choices (N × max_firm_id+1) one-hot including outside option
    print("Building choice matrix...")
    Y_choices = build_choice_matrix(chosen_firm, firm_ids)
    print(f"Choice matrix shape: {Y_choices.shape}")
    print(f"Choice distribution: {np.sum(Y_choices, axis=0)}")
    print(f"Outside choices: {np.sum(Y_choices[:, 0])}")
    print(f"Firm choices by firm_id: {np.sum(Y_choices[:, 1:], axis=0)}")
    print(f"Firm IDs: {firm_ids}")
    
    # Create GMM objective with identity weighting (W=None)
    print("Creating GMM objective...")
    gmm_obj_gamma = create_gmm_objective(probs_evaluator, Y_choices, W=None)
    
    # Test objective with sample gamma
    test_gamma = 0.1
    test_obj = gmm_obj_gamma(test_gamma)
    print(f"Test objective at γ = {test_gamma}: Q = {test_obj:.6f}")
    
    # First-step optimization with identity weighting
    print("\nOptimizing γ (first-step, identity weighting)...")
    results = optimize_gamma(gmm_obj_gamma, probs_evaluator, Y_choices, grid_search=args.grid_search)
    
    # Compute first-step moments and scores
    gamma_hat_1 = results['gamma_hat']
    m1, Psi1 = moments_and_scores(gamma_hat_1, probs_evaluator, Y_choices)
    Q1 = float(m1 @ m1)
    
    # Store first-step results
    res_step1 = {
        "gamma_hat_1": float(gamma_hat_1),
        "Q1": float(Q1), 
        "m1": m1.tolist(),
        "N": int(N),
        "moment_dim": int(m1.size)
    }
    
    # Print one-line summary
    m1_norm = float(np.linalg.norm(m1))
    print(f"[GMM-1] gamma_hat_1={gamma_hat_1:.4f}, Q1={Q1:.6f}, ||m1||2={m1_norm:.6f}")
    
    # Compute efficient weighting matrix from first-step results
    print("\nComputing efficient weighting matrix...")
    
    # 1) Compute S_hat_raw = (Psi1.T @ Psi1) / N
    S_hat_raw = (Psi1.T @ Psi1) / N
    
    # 2) Symmetrize: S_hat = 0.5 * (S_hat_raw + S_hat_raw.T)
    S_hat = 0.5 * (S_hat_raw + S_hat_raw.T)
    
    # 3) Ridge regularization
    ridge = 1e-8
    trace_S = np.trace(S_hat)
    ridge_amount = ridge * trace_S / S_hat.shape[0]
    S_hat = S_hat + ridge_amount * np.eye(S_hat.shape[0])
    
    # 4) Try to invert S_hat for W_hat = S_hat^{-1}
    cond_S = np.linalg.cond(S_hat)
    fallback = 0
    
    if cond_S > 1e12:
        # Use diagonal fallback
        W_hat = np.diag(1.0 / np.maximum(np.diag(S_hat), 1e-12))
        fallback = 1
    else:
        try:
            W_hat = np.linalg.inv(S_hat)
        except np.linalg.LinAlgError:
            # Inversion failed, use diagonal fallback
            W_hat = np.diag(1.0 / np.maximum(np.diag(S_hat), 1e-12))
            fallback = 1
    
    # 5) Ensure W_hat is symmetric
    W_hat = 0.5 * (W_hat + W_hat.T)
    
    # Store in res_step1
    res_step1["S_hat"] = S_hat.tolist()
    res_step1["W_hat"] = W_hat.tolist()
    
    # Print summary
    print(f"[GMM-W] cond(S_hat)={cond_S:.2e}, ridge={ridge:.2e}, fallback={fallback}")
    
    # Decide whether to run two-step GMM
    if args.two_step_gmm:
        # Second-step optimization with efficient weighting matrix
        print("\nOptimizing γ (second-step, efficient weighting)...")
        
        # Create GMM objective with efficient weighting matrix W_hat
        gmm_obj_gamma_step2 = create_gmm_objective(probs_evaluator, Y_choices, W=W_hat)
        
        # Test objective with first-step estimate
        test_obj_step2 = gmm_obj_gamma_step2(gamma_hat_1)
        print(f"Test objective at γ₁ = {gamma_hat_1:.4f}: Q(γ₁; Ŵ) = {test_obj_step2:.6f}")
        
        # Optimize with efficient weighting
        results_step2 = optimize_gamma(gmm_obj_gamma_step2, probs_evaluator, Y_choices, grid_search=False)
        
        # Compute second-step moments and scores
        gamma_hat_2 = results_step2['gamma_hat']
        m2, Psi2 = moments_and_scores(gamma_hat_2, probs_evaluator, Y_choices)
        Q2 = float(m2 @ (W_hat @ m2))  # Use efficient weighting for objective
        
        # Print second-step summary
        m2_norm = float(np.linalg.norm(m2))
        print(f"[GMM-2] gamma_hat_2={gamma_hat_2:.4f}, Q2={Q2:.6f}, ||m2||2={m2_norm:.6f}")
        
        # Use second-step results as final results
        results = results_step2
        results.update({
            'method': 'two_step_gmm',
            'step1_results': res_step1,
            'step2_results': {
                'gamma_hat_2': float(gamma_hat_2),
                'Q2': float(Q2),
                'm2': m2.tolist(),
                'N': int(N),
                'moment_dim': int(m2.size)
            }
        })
        
        # Save compact two-step JSON
        two_step_json = {
            "two_step": True,
            "gamma_hat_1": float(gamma_hat_1),
            "Q1": float(Q1),
            "gamma_hat_2": float(gamma_hat_2),
            "Q2": float(Q2),
            "moment_dim": int(m1.size),
            "N": int(N)
        }
        json_path = os.path.join(args.out_dir, "gmm_gamma_two_step.json")
        with open(json_path, 'w') as f:
            json.dump(two_step_json, f, indent=2)
        print(f"Two-step results saved to {json_path}")
        
    else:
        # Use first-step results as final results (one-step GMM)
        results.update({
            'method': 'one_step_gmm',
            'step1_results': res_step1
        })
        
        # Save compact one-step JSON
        one_step_json = {
            "two_step": False,
            "gamma_hat_1": float(gamma_hat_1),
            "Q1": float(Q1),
            "moment_dim": int(m1.size),
            "N": int(N)
        }
        json_path = os.path.join(args.out_dir, "gmm_gamma_one_step.json")
        with open(json_path, 'w') as f:
            json.dump(one_step_json, f, indent=2)
        print(f"One-step results saved to {json_path}")
    
    # Compute GMM standard errors
    print("\nComputing GMM standard errors...")
    se_results = compute_gmm_standard_errors(
        results['gamma_hat'], 
        gmm_obj_gamma, 
        probs_evaluator, 
        Y_choices, 
        N, 
        Y_choices.shape[1] - 1  # J = number of firms (Y_choices.shape[1] includes outside option)
    )
    
    # Merge results
    results.update(se_results)
    
    # Save results
    print("\nSaving results...")
    save_results(results, args.out_dir)
    
    # Plot objective function if requested
    if args.plot_objective:
        print(f"\nPlotting GMM objective function...")
        plot_path = Path(args.out_dir) / 'gmm_objective_plot.png'
        plot_gmm_objective(gmm_obj_gamma, 
                          gamma_max=args.plot_gamma_max,
                          grid_fineness=args.plot_fineness,
                          save_path=str(plot_path),
                          gamma_hat=results.get('gamma_hat'),
                          se_gamma=results.get('se_gamma'))
    
    # Debug worker choice probabilities if requested
    if args.debug_worker_index is not None:
        print(f"\nDebug dump for worker {args.debug_worker_index}...")
        
        # Determine gamma for debug
        debug_gamma = args.debug_gamma if args.debug_gamma is not None else results['gamma_hat']
        
        # Worker location data is already loaded
        # (ell_x and ell_y are already available from main data loading)
        
        # Call debug function
        debug_worker_choice_probabilities(
            worker_idx=args.debug_worker_index,
            gamma=debug_gamma,
            probs_worker_level=probs_evaluator, # Pass the evaluator directly
            out_dir=args.out_dir
        )
    
    print("\nGMM estimation complete!")


if __name__ == '__main__':
    main()
