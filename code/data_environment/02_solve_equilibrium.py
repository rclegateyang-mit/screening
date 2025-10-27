#!/usr/bin/env python3
"""
Worker Screening Equilibrium Solver

This script reads firm data and worker quadrature points, then solves the fixed-point
equilibrium over wages (w_j) and cutoff costs (c_j) for firms j=1..J.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import root

try:
    from .. import get_data_dir
except ImportError:  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from code import get_data_dir  # type: ignore

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from code.estimation.helpers import (  # type: ignore
        compute_order_maps,
        conditional_LM_at_loc,
        inclusive_values_at_loc,
        reorder_to_natural,
        safe_logs,
    )
else:  # pragma: no cover - executed when running as package module
    from ..estimation.helpers import (
        compute_order_maps,
        conditional_LM_at_loc,
        inclusive_values_at_loc,
        reorder_to_natural,
        safe_logs,
    )

# Optional numba import
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# =============================================================================
# CORE SOLVER FUNCTIONS
# =============================================================================




def truncated_normal_column_terms(
    c_sorted: np.ndarray, mu_s: float, sigma_s: float, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    f"""
    Returns (DeltaF, M) for k = 0,...,J using sentinels c_(0)=-inf, c_(J+1)=+inf.
    DeltaF[k] = Φ(z_k+1) − Φ(z_k), M[k] = μ_s + σ_s * (φ(z_k+1) − φ(z_k)) / DeltaF[k],
    where z_k = (c_(k) − μ_s)/σ_s and Φ,φ are standard normal CDF/PDF.
    Enforces Φ(z_0)=0 and Φ(z_J+1)=1 exactly; φ at sentinels is 0.
    
    Economic interpretation: Intervals (c_(k), c_(k+1)] represent skill ranges where
    workers are eligible for firms with rank k. A firm with rank r contributes to
    intervals k ≥ r+1 (strict inequality, excluding own upper-bound interval).
    
    Args:
        c_sorted: Sorted cutoff costs (J,)
        mu_s: Mean of skill distribution
        sigma_s: Standard deviation of skill distribution
        eps: Numerical safety floor
    
    Returns:
        Tuple of (DeltaF, M) arrays, both of length J+1
    """
    J = int(c_sorted.shape[0])
    
    # Pad with sentinels
    c_pad = np.empty(J + 2, dtype=np.float64)
    c_pad[0] = -np.inf
    c_pad[1:-1] = c_sorted
    c_pad[-1] = np.inf
    
    # Standardize
    z = (c_pad - mu_s) / sigma_s
    
    # Compute CDF and PDF
    Phi = np.empty_like(z)
    Phi[0] = 0.0
    Phi[-1] = 1.0
    if J > 0:
        Phi[1:-1] = special.ndtr(z[1:-1])
    
    phi = np.zeros_like(z)
    if J > 0:
        phi[1:-1] = np.exp(-0.5 * z[1:-1]**2) / np.sqrt(2 * np.pi)
    
    # Compute interval masses and conditional means
    DeltaF = Phi[1:] - Phi[:-1]  # length J+1
    DeltaF = np.maximum(DeltaF, eps)  # floor for numerical stability
    DeltaF = DeltaF / DeltaF.sum()  # renormalize to exactly 1
    
    # Conditional mean on each interval
    M = mu_s + sigma_s * (phi[:-1] - phi[1:]) / np.maximum(DeltaF, eps)
    
    # Clip extremes when DeltaF is tiny
    M = np.clip(M, mu_s - 20.0 * sigma_s, mu_s + 20.0 * sigma_s)
    
    return DeltaF, M





def aggregate_over_locations(support_points: np.ndarray, weights: np.ndarray, w: np.ndarray, 
                           c: np.ndarray, xi: np.ndarray, loc: np.ndarray, alpha: float, 
                           gamma: float, mu_s: float, sigma_s: float) -> dict:
    """
    Aggregate over worker locations to produce L, M, S vectors in by-c order and natural order.
    
    Args:
        support_points: Worker support points (S, 2)
        weights: Worker weights (S,)
        w: Wages (J,)
        c: Cutoff costs (J,)
        xi: Firm amenity shocks (J,)
        loc: Firm locations (J, 2)
        alpha: Wage elasticity parameter
        gamma: Distance decay parameter
        mu_s: Mean of skill distribution
        sigma_s: Standard deviation of skill distribution
        
    Returns:
        Dictionary containing:
        - L_byc, M_byc, S_byc: (J+1,) vectors in by-c order (including outside)
        - L_firms_nat, M_firms_nat, S_firms_nat: (J,) vectors in natural order (firms only)
        - order_idx, inv_order: Ordering mappings
    """
    J, S = len(w), len(support_points)
    
    # Step 1: Build order maps from c and compute sorted arrays
    order_maps = compute_order_maps(c)
    order_idx = order_maps['order_idx']
    inv_order = order_maps['inv_order']
    c_sorted = order_maps['c_sorted']
    
    # Sort other arrays by c order
    w_sorted = w[order_idx]
    xi_sorted = xi[order_idx]
    loc_sorted = loc[order_idx]
    
    # Step 2: Compute (p, m) once from c_sorted
    DeltaF, M_conditional = truncated_normal_column_terms(c_sorted, mu_s, sigma_s, eps=1e-12)
    
    # p and m vectors (J+1,) including outside option
    p = np.zeros(J + 1, dtype=np.float64)
    m = np.zeros(J + 1, dtype=np.float64)
    p = DeltaF 
    m = DeltaF * M_conditional 
    
    # Ensure m sums to mu_s (normalize if needed)
    m_sum = m.sum()
    if abs(m_sum - mu_s) > 1e-12:
        # Scale m to sum to mu_s, but keep outside option at 0
        if abs(m_sum) > 1e-12:
            m[1:] = m[1:] * (mu_s / m_sum)
        else:
            # If m_sum is zero, set m to zero (this shouldn't happen in practice)
            m[1:] = 0.0
    
    # Step 3: Aggregate over locations
    L_byc = np.zeros(J + 1, dtype=np.float64)
    M_byc = np.zeros(J + 1, dtype=np.float64)
    
    for s in range(S):
        ell = support_points[s]
        omega_ell = weights[s]
        
        # Compute inclusive values at this location
        v = inclusive_values_at_loc(ell, w_sorted, xi_sorted, loc_sorted, alpha, gamma)
        
        # Compute location-conditional L and M
        L_loc, M_loc = conditional_LM_at_loc(v, p, m)
        
        # Weight and accumulate
        L_byc += omega_ell * L_loc
        M_byc += omega_ell * M_loc
    
    # Step 4: Compute S_byc = M_byc / L_byc (guard division)
    S_byc = np.zeros(J + 1, dtype=np.float64)
    S_byc[L_byc > 1e-300] = M_byc[L_byc > 1e-300] / L_byc[L_byc > 1e-300]
    
    # Step 5: Drop outside index 0 → firms_byc = slice(1:)
    L_firms_byc = L_byc[1:]
    M_firms_byc = M_byc[1:]
    S_firms_byc = S_byc[1:]
    
    # Step 6: Map firms_byc back to natural using reorder_to_natural
    L_firms_nat = L_firms_byc[inv_order] 
    M_firms_nat = M_firms_byc[inv_order]
    S_firms_nat = S_firms_byc[inv_order] 
    
    return {
        'L_byc': L_byc,
        'M_byc': M_byc,
        'S_byc': S_byc,
        'L_firms_nat': L_firms_nat,
        'M_firms_nat': M_firms_nat,
        'S_firms_nat': S_firms_nat,
        'order_idx': order_idx,
        'inv_order': inv_order
    }





def fixed_point_map(
    logw: np.ndarray, logc: np.ndarray, A: np.ndarray, xi: np.ndarray,
    loc_firms: np.ndarray, support_points: np.ndarray, support_weights: np.ndarray,
    mu_s: float, sigma_s: float, alpha: float, beta: float, gamma: float,
    eps: float = 1e-12, conduct_mode: int = 1, N_workers: float = 1.0,
    eps_L_behavioral: Optional[np.ndarray] = None, eps_S_behavioral: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute one step of the fixed-point map using corrected by-c model with U-matrix via suffix sums.
    
    Args:
        logw: Current log wages (J,)
        logc: Current log cutoff costs (J,)
        A: Firm TFP (J,)
        xi: Firm amenity shocks (J,)
        loc_firms: Firm locations (J, 2)
        support_points: Worker support points (S, 2)
        support_weights: Worker weights (S,)
        mu_s: Mean of skill distribution
        sigma_s: Standard deviation of skill distribution
        alpha: Wage elasticity parameter
        beta: Production function parameter
        gamma: Distance decay parameter
        eps: Numerical safety floor
        conduct_mode: 0=monopsonistic, 1=status quo, 2=behavioral elasticities
        
    Returns:
        Tuple of (logw_new, logc_new, diagnostics)
    """
    J, S = len(A), len(support_points)
    
    # Convert to levels
    w = np.exp(logw)
    c = np.exp(logc)
    
    # Use the corrected by-c model with U-matrix via suffix sums
    # Aggregate over locations to get L, M, S in by-c order
    agg_result = aggregate_over_locations(
        support_points, support_weights, w, c, xi, loc_firms, 
        alpha, gamma, mu_s, sigma_s
    )
    
    # Extract results
    L_byc = agg_result['L_byc']  # (J+1,) including outside
    M_byc = agg_result['M_byc']  # (J+1,) including outside  
    S_byc = agg_result['S_byc']  # (J+1,) including outside
    L_firms_nat = agg_result['L_firms_nat']  # (J,) firms only, natural order
    M_firms_nat = agg_result['M_firms_nat']  # (J,) firms only, natural order
    S_firms_nat = agg_result['S_firms_nat']  # (J,) firms only, natural order
    order_idx = agg_result['order_idx']
    inv_order = agg_result['inv_order']
  
    # Get rank for each firm (position in by-c order)
    rank = np.argsort(order_idx)  # rank[j] = position of firm j in sorted array
    
    # Extract firms-only vectors in by-c order (drop outside option)
    L_firms_byc = L_byc[1:]  # (J,) firms only, by-c order
    M_firms_byc = M_byc[1:]  # (J,) firms only, by-c order
    S_firms_byc = S_byc[1:]  # (J,) firms only, by-c order
    
    # Scale labor by market size (L_firms_byc are already shares)
    L_share = L_firms_byc  # (J,) in by-c order
    L = N_workers * L_share  # (J,) in by-c order
    S = S_firms_byc  # (J,) in by-c order
    
    # Output using scaled labor (in by-c order)
    Y = A[order_idx] * (L * S) ** (1 - beta)  # (J,) in by-c order
    
    # Wage equation depends on conduct mode (compute in by-c order)
    w_byc = w[order_idx]  # Wages in by-c order
    A_byc = A[order_idx]  # TFP in by-c order
    
    if conduct_mode == 1:
        # Monopsonistic: log w_j = log(1-β) + log Y_j - log L_j + log(α/(α+1))
        logw_new_byc = np.log(1 - beta) + safe_logs(Y, eps) - safe_logs(L, eps) + np.log(alpha / (alpha + 1))
    elif conduct_mode == 2:
        # Behavioral elasticities: use firm-specific eps_L_behavioral and eps_S_behavioral
        if eps_L_behavioral is None or eps_S_behavioral is None:
            raise ValueError("Behavioral elasticities required for conduct_mode=2")
        
        # Get behavioral elasticities in by-c order
        eps_L_behavioral_byc = eps_L_behavioral[order_idx]
        eps_S_behavioral_byc = eps_S_behavioral[order_idx]
        
        # Compute wage equation with behavioral elasticities
        numerator = eps_L_behavioral_byc + eps_S_behavioral_byc
        denominator = eps_L_behavioral_byc + 1
        logw_new_byc = np.log(1 - beta) + safe_logs(Y, eps) - safe_logs(L, eps) + safe_logs(numerator / np.maximum(denominator, eps), eps)
    else:
        # conduct_mode == 0: For now, use status quo as placeholder
        logw_new_byc = np.log(1 - beta) + safe_logs(Y, eps) - safe_logs(L, eps) + safe_logs((L_elas_byc + S_elas_byc) / (L_elas_byc + 1))
    
    # c-update: log c_j = log w_j - log A_j - log(1-beta) + beta/(1-beta) *(log Y_j - log A_j)
    logc_new_byc = logw_new_byc - np.log(A_byc) - np.log(1 - beta) + beta/(1 - beta) * (safe_logs(Y, eps) - np.log(A_byc))

    # Map back to natural order
    logw_new = np.empty(J, dtype=np.float64)
    logc_new = np.empty(J, dtype=np.float64)
    logw_new[inv_order] = logw_new_byc
    logc_new[inv_order] = logc_new_byc
    
    # Diagnostics
    diagnostics = {
        'L': L_firms_nat, 'L_share': L_firms_nat, 'S': S_firms_nat, 'Y': Y[inv_order], 'rank': rank, 'order_idx': order_idx,
        'L_byc': L_byc, 'M_byc': M_byc, 'S_byc': S_byc,
        'L_firms_byc': L_firms_byc, 'M_firms_byc': M_firms_byc, 'S_firms_byc': S_firms_byc,
        'L_firms_nat': L_firms_nat, 'M_firms_nat': M_firms_nat, 'S_firms_nat': S_firms_nat,
        'inv_order': inv_order
    }
    
    if conduct_mode == 0:
        # TODO: Add monopsonistic diagnostics when implemented
        pass
    elif conduct_mode == 2:
        # Add behavioral elasticities to diagnostics
        diagnostics['eps_L_behavioral'] = eps_L_behavioral
        diagnostics['eps_S_behavioral'] = eps_S_behavioral
    
    return logw_new, logc_new, diagnostics

def f_eqm(
    logx: np.ndarray, A: np.ndarray, xi: np.ndarray,
    loc_firms: np.ndarray, support_points: np.ndarray, support_weights: np.ndarray,
    mu_s: float, sigma_s: float, alpha: float, beta: float, gamma: float,
    eps: float = 1e-12, conduct_mode: int = 1, N_workers: float = 1.0,
    eps_L_behavioral: Optional[np.ndarray] = None, eps_S_behavioral: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute one step of the fixed-point map using corrected by-c model with U-matrix via suffix sums.
    
    Args:
        [logw, logc] (2J,)
        A: Firm TFP (J,)
        xi: Firm amenity shocks (J,)
        loc_firms: Firm locations (J, 2)
        support_points: Worker support points (S, 2)
        support_weights: Worker weights (S,)
        mu_s: Mean of skill distribution
        sigma_s: Standard deviation of skill distribution
        alpha: Wage elasticity parameter
        beta: Production function parameter
        gamma: Distance decay parameter
        eps: Numerical safety floor
        conduct_mode: 0=monopsonistic, 1=status quo, 2=behavioral elasticities
        
    Returns:
        [logw_new, logc_new] (2J,)
    """
    J, S = len(A), len(support_points)
    
    # Convert to levels
    w = np.exp(logx[0:J])
    c = np.exp(logx[J:])
    
    # Use the corrected by-c model with U-matrix via suffix sums
    # Aggregate over locations to get L, M, S in by-c order
    agg_result = aggregate_over_locations(
        support_points, support_weights, w, c, xi, loc_firms, 
        alpha, gamma, mu_s, sigma_s
    )
    
    # Extract results
    L_byc = agg_result['L_byc']  # (J+1,) including outside
    M_byc = agg_result['M_byc']  # (J+1,) including outside  
    S_byc = agg_result['S_byc']  # (J+1,) including outside
    L_firms_nat = agg_result['L_firms_nat']  # (J,) firms only, natural order
    M_firms_nat = agg_result['M_firms_nat']  # (J,) firms only, natural order
    S_firms_nat = agg_result['S_firms_nat']  # (J,) firms only, natural order
    order_idx = agg_result['order_idx']
    inv_order = agg_result['inv_order']
   
    # Extract firms-only vectors in by-c order (drop outside option)
    L_firms_byc = L_byc[1:]  # (J,) firms only, by-c order
    M_firms_byc = M_byc[1:]  # (J,) firms only, by-c order
    S_firms_byc = S_byc[1:]  # (J,) firms only, by-c order
    
    # Scale labor by market size (L_firms_byc are already shares)
    L_share = L_firms_byc  # (J,) in by-c order
    L = N_workers * L_share  # (J,) in by-c order
    S = S_firms_byc  # (J,) in by-c order
    M = M_firms_byc * N_workers
   
    # Output using scaled labor (in by-c order)
    Y = A[order_idx] * (M) ** (1 - beta)  # (J,) in by-c order
    
    # Wage equation depends on conduct mode (compute in by-c order)
    w_byc = w[order_idx]  # Wages in by-c order
    A_byc = A[order_idx]  # TFP in by-c order
    
    if conduct_mode == 1:
        # Monopsonistic: log w_j = log(1-β) + log Y_j - log L_j + log(α/(α+1))
        logw_new_byc = np.log(1 - beta) + safe_logs(Y, eps) - safe_logs(L, eps) + np.log(alpha / (alpha + 1))
    elif conduct_mode == 2:
        # Behavioral elasticities: use firm-specific eps_L_behavioral and eps_S_behavioral
        if eps_L_behavioral is None or eps_S_behavioral is None:
            raise ValueError("Behavioral elasticities required for conduct_mode=2")
        
        # Get behavioral elasticities in by-c order
        eps_L_behavioral_byc = eps_L_behavioral[order_idx]
        eps_S_behavioral_byc = eps_S_behavioral[order_idx]
        
        # Compute wage equation with behavioral elasticities
        numerator = eps_L_behavioral_byc + eps_S_behavioral_byc
        denominator = eps_L_behavioral_byc + 1
        logw_new_byc = np.log(1 - beta) + safe_logs(Y, eps) - safe_logs(L, eps) + safe_logs(numerator / np.maximum(denominator, eps), eps)
    else:
        # conduct_mode == 0: For now, use status quo as placeholder
        logw_new_byc = np.log(1 - beta) + safe_logs(Y, eps) - safe_logs(L, eps) + safe_logs((L_elas_byc + S_elas_byc) / (L_elas_byc + 1))
    
    # c-update: log c_j = log w_j - log A_j - log(1-beta) + beta/(1-beta) *(log Y_j - log A_j)
    logc_new_byc = logw_new_byc - np.log(A_byc) - np.log(1 - beta) + beta/(1 - beta) * (safe_logs(Y, eps) - np.log(A_byc))

    # Map back to natural order
    logw_new = np.empty(J, dtype=np.float64)
    logc_new = np.empty(J, dtype=np.float64)
    logw_new = logw_new_byc[inv_order]
    logc_new = logc_new_byc[inv_order]
    
    logx_new = np.concatenate([logw_new, logc_new])
    delta_logx = logx_new - logx
    return delta_logx



def anderson_update(x: np.ndarray, Fx: np.ndarray, store: dict, m: int, 
                   reg: float, damping: float) -> np.ndarray:
    """
    Anderson acceleration update.
    
    Args:
        x: Current iterate
        Fx: Function evaluation at x
        store: Storage for residuals and updates
        m: Anderson memory
        reg: Tikhonov regularization
        damping: Fallback damping factor
        
    Returns:
        Updated iterate
    """
    r = Fx - x  # residual
    
    # Store history
    if 'residuals' not in store:
        store['residuals'] = []
        store['updates'] = []
    
    store['residuals'].append(r)
    store['updates'].append(x)
    
    # Anderson update
    if len(store['residuals']) > 1 and m > 0:
        k = int(min(m, len(store['residuals']) - 1))
        
        # Build least-squares system
        dr = np.column_stack([store['residuals'][-i-1] - store['residuals'][-i-2] 
                             for i in range(k)])
        dx = np.column_stack([store['updates'][-i-1] - store['updates'][-i-2] 
                             for i in range(k)])
        
        # Solve with regularization
        try:
            coeffs = np.linalg.lstsq(dr.T @ dr + reg * np.eye(k), 
                                   dr.T @ r, rcond=None)[0]
            x_new = x - r + dx @ coeffs
        except:
            # Fallback to damping
            x_new = x + damping * r
    else:
        # Simple damping
        x_new = x + damping * r
    
    # Clip updates to prevent extreme jumps
    max_update = 5.0
    update = x_new - x
    
    # Check for invalid values (NaN or inf)
    if np.any(~np.isfinite(update)):
        # Fallback to damping if update contains invalid values
        x_new = x + damping * (Fx - x)
    elif np.any(np.abs(update) > max_update):
        scale = max_update / np.max(np.abs(update))
        x_new = x + scale * update
    
    return x_new


def solve_equilibrium(
    A: np.ndarray, xi: np.ndarray, loc_firms: np.ndarray,
    support_points: np.ndarray, support_weights: np.ndarray,
    mu_s: float, sigma_s: float, alpha: float, beta: float, gamma: float,
    N_workers: float = 1.0, conduct_mode: int = 1, eps_L_behavioral: Optional[np.ndarray] = None,
    eps_S_behavioral: Optional[np.ndarray] = None, max_iter: int = 2000, tol: float = 1e-8,
    damping: float = 0.5, anderson_m: int = 5, anderson_reg: float = 1e-8,
    eps: float = 1e-12, check_every: int = 10
) -> Dict[str, Any]:
    """
    Solve the fixed-point equilibrium.
    
    Args:
        A: Firm TFP (J,)
        xi: Firm amenity shocks (J,)
        loc_firms: Firm locations (J, 2)
        support_points: Worker support points (S, 2)
        support_weights: Worker weights (S,)
        mu_s: Mean of skill distribution
        sigma_s: Standard deviation of skill distribution
        alpha: Wage elasticity parameter
        beta: Production function parameter
        gamma: Distance decay parameter
        N_workers: Total number of workers in the market
        conduct_mode: 0=monopsonistic, 1=status quo, 2=behavioral elasticities
        eps_L_behavioral: Behavioral labor elasticities (J,) for conduct_mode=2
        eps_S_behavioral: Behavioral skill elasticities (J,) for conduct_mode=2
        max_iter: Maximum iterations
        tol: Convergence tolerance
        damping: Fallback damping factor
        anderson_m: Anderson memory
        anderson_reg: Tikhonov regularization
        eps: Numerical safety floor
        check_every: Frequency of diagnostics
        
    Returns:
        Dictionary with solution and diagnostics
    """
    J = len(A)
    
    # Initialize with economically reasonable values
    # Start with wages that give positive employment
    logw = np.log(1 - beta) + np.log(A) + np.log(alpha / (alpha + 1)) + 2.0  # Large offset for stability
    logc = -np.log(1 - beta) + logw - np.log(A) + 1.0  # Large offset to cutoffs
    
    # Storage for Anderson acceleration
    store = {}
    
    # Iteration
    start_time = time.time()
    for iter_num in range(max_iter):
        iter_start = time.time()
        
        # Fixed-point map
        logw_new, logc_new, diagnostics = fixed_point_map(
            logw, logc, A, xi, loc_firms, support_points, support_weights,
            mu_s, sigma_s, alpha, beta, gamma, eps, conduct_mode, N_workers,
            eps_L_behavioral, eps_S_behavioral
        )
        
        # Anderson acceleration
        x = np.concatenate([logw, logc])
        Fx = np.concatenate([logw_new, logc_new])
        
        x_new = anderson_update(x, Fx, store, anderson_m, anderson_reg, damping)
        
        # Extract new values
        logw_new = x_new[:J]
        logc_new = x_new[J:]
        
        # Check convergence
        residual = np.max(np.abs(x_new - x))
        
        # Update with damping to prevent instability
        # Use CLI-provided damping parameter instead of hard-coded value
        logw = (1 - damping) * logw + damping * logw_new
        logc = (1 - damping) * logc + damping * logc_new
        
        # Diagnostics
        if iter_num % check_every == 0:
            iter_time = time.time() - iter_start
            print(f"Iteration {iter_num}: residual = {residual:.2e}, time = {iter_time:.3f}s")
        
        # Check convergence
        if residual <= tol:
            break
    
    # Final diagnostics
    total_time = time.time() - start_time
    converged = residual <= tol
    
    # Convert to levels
    w = np.exp(logw)
    c = np.exp(logc)
    
    # Final fixed-point map for diagnostics
    _, _, final_diagnostics = fixed_point_map(
        logw, logc, A, xi, loc_firms, support_points, support_weights,
        mu_s, sigma_s, alpha, beta, gamma, eps, conduct_mode, N_workers,
        eps_L_behavioral, eps_S_behavioral
    )
    
    # Compute output
    Y = final_diagnostics['Y']
    
    return {
        'w': w, 'c': c, 'L': final_diagnostics['L'], 'S': final_diagnostics['S'], 'Y': Y,
        'logw': logw, 'logc': logc, 'rank': final_diagnostics['rank'],
        'iters': iter_num + 1, 'converged': converged, 'residual': residual,
        'timing': {'total_time': total_time},
        'diagnostics': final_diagnostics
    }


# =============================================================================
# I/O FUNCTIONS
# =============================================================================

def read_firms_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read firm data from CSV.
    
    Args:
        path: Path to firms CSV file
        
    Returns:
        Tuple of (A, xi, loc_firms, firm_id, comp, eps_L_behavioral, eps_S_behavioral)
        eps_L_behavioral and eps_S_behavioral are None if not present in CSV
    """
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ['firm_id', 'logA', 'A', 'xi', 'comp', 'x', 'y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by firm_id
    df = df.sort_values('firm_id').reset_index(drop=True)
    
    # Extract arrays
    firm_id = df['firm_id'].values
    A = df['A'].values
    xi = df['xi'].values
    loc_firms = df[['x', 'y']].values
    comp = df['comp'].values
    
    # Validate A and logA consistency
    if 'logA' in df.columns:
        logA = df['logA'].values
        max_rel_error = np.max(np.abs(A - np.exp(logA)) / A)
        if max_rel_error > 1e-8:
            print(f"Warning: A and logA inconsistent, max relative error: {max_rel_error:.2e}")
    
    # Check for behavioral elasticities
    eps_L_behavioral = None
    eps_S_behavioral = None
    if 'eps_L_behavioral' in df.columns and 'eps_S_behavioral' in df.columns:
        eps_L_behavioral = df['eps_L_behavioral'].values
        eps_S_behavioral = df['eps_S_behavioral'].values
        print(f"Found behavioral elasticities: eps_L_behavioral (mean={eps_L_behavioral.mean():.4f}, std={eps_L_behavioral.std():.4f})")
        print(f"  eps_S_behavioral (mean={eps_S_behavioral.mean():.4f}, std={eps_S_behavioral.std():.4f})")
    
    return A, xi, loc_firms, firm_id, comp, eps_L_behavioral, eps_S_behavioral


def read_support_points_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read support points from CSV.
    
    Args:
        path: Path to support points CSV file
        
    Returns:
        Tuple of (support_points, support_weights)
    """
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ['x', 'y', 'weight']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract arrays
    support_points = df[['x', 'y']].values
    support_weights = df['weight'].values
    
    # Validate weights
    if not np.all(support_weights >= 0):
        raise ValueError("Support weights must be non-negative")
    
    # Normalize weights if needed
    weight_sum = support_weights.sum()
    if not np.isclose(weight_sum, 1.0, atol=1e-12):
        print(f"Warning: Normalizing support weights from {weight_sum:.12f} to 1.0")
        support_weights = support_weights / weight_sum

    # Later scaling handled after reading parameters (requires N_workers)

    return support_points, support_weights


def read_parameters_csv(path: str) -> Dict[str, Any]:
    """
    Read parameters from CSV.
    
    Args:
        path: Path to parameters CSV file
        
    Returns:
        Dictionary of parameter values
    """
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ['parameter', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert to dictionary with type casting
    params = {}
    for _, row in df.iterrows():
        param_name = row['parameter']
        param_value = row['value']
        
        # Type casting based on parameter name
        try:
            if param_name in ['conduct_mode', 'max_iter', 'max_plots', 'quad_n_x', 'quad_n_y', 'seed']:
                params[param_name] = int(param_value)
            elif param_name in ['quad_normalize', 'quad_write_csv']:
                params[param_name] = bool(param_value)
            elif param_name == 'quad_kind':
                params[param_name] = str(param_value)
            else:
                params[param_name] = float(param_value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for parameter {param_name}: {param_value}")
    
    return params


def write_equilibrium_csv(
    path: str,
    firm_id: np.ndarray,
    comp: np.ndarray,
    A: np.ndarray,
    xi: np.ndarray,
    loc_firms: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    L: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    logw: np.ndarray,
    logc: np.ndarray,
    rank: np.ndarray,
    firm_id_original: Optional[np.ndarray] = None,
) -> str:
    """
    Write equilibrium results to CSV.
    
    Args:
        path: Output file path
        firm_id: Firm IDs
        comp: Component labels
        A: Firm TFP
        xi: Firm amenity shocks
        loc_firms: Firm locations
        w: Wages
        c: Cutoff costs
        L: Labor supply
        S: Skill supply
        Y: Output
        logw: Log wages
        logc: Log cutoff costs
        rank: Firm ranks
        firm_id_original: Optional array of original firm IDs (before dropping)
        
    Returns:
        Path to written file
    """
    data = {
        'firm_id': firm_id,
        'w': w,
        'c': c,
        'L': L,
        'S': S,
        'Y': Y,
        'logw': logw,
        'logc': logc,
        'rank': rank,
        'A': A,
        'xi': xi,
        'x': loc_firms[:, 0],
        'y': loc_firms[:, 1],
        'comp': comp,
    }
    if firm_id_original is not None:
        data['firm_id_original'] = firm_id_original

    df = pd.DataFrame(data)
    # Ensure original IDs appear next to the new IDs for readability
    if firm_id_original is not None:
        cols = ['firm_id', 'firm_id_original'] + [col for col in df.columns if col not in {'firm_id', 'firm_id_original'}]
        df = df[cols]

    df.to_csv(path, index=False)
    return path


# =============================================================================
# PROFIT SURFACE FUNCTIONS
# =============================================================================

def profit_surface_for_firm(
    j: int, logw_eq: np.ndarray, logc_eq: np.ndarray,
    A: np.ndarray, xi: np.ndarray, loc_firms: np.ndarray,
    support_points: np.ndarray, support_weights: np.ndarray,
    mu_s: float, sigma_s: float, alpha: float, beta: float, gamma: float,
    B: np.ndarray, N_workers: float = 1.0, grid_n: int = 40, grid_log_span: float = 0.5, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Compute profit surface for firm j.
    
    Args:
        j: Firm index
        logw_eq: Equilibrium log wages
        logc_eq: Equilibrium log cutoff costs
        A: Firm TFP
        xi: Firm amenity shocks
        loc_firms: Firm locations
        support_points: Worker support points
        support_weights: Worker weights
        mu_s: Mean of skill distribution
        sigma_s: Standard deviation of skill distribution
        alpha: Wage elasticity parameter
        beta: Production function parameter
        gamma: Distance decay parameter
        B: Precomputed base intensities
        grid_n: Grid resolution
        grid_log_span: Log span for grid
        eps: Numerical safety floor
        
    Returns:
        Tuple of (W, C, Pi, logw_star, logc_star, logw_argmax, logc_argmax)
    """
    J = len(A)
    
    # Equilibrium values for firm j
    logw_star = logw_eq[j]
    logc_star = logc_eq[j]
    
    # Create grid
    grid_n_int = int(grid_n)
    logw_grid = np.linspace(logw_star - grid_log_span, logw_star + grid_log_span, grid_n_int)
    logc_grid = np.linspace(logc_star - grid_log_span, logc_star + grid_log_span, grid_n_int)
    
    W, C = np.meshgrid(np.exp(logw_grid), np.exp(logc_grid))
    Pi = np.zeros_like(W)
    
    # Evaluate profit at each grid point
    for i in range(grid_n_int):
        for k in range(grid_n_int):
            # Create modified vectors with only firm j changed
            logw_mod = logw_eq.copy()
            logc_mod = logc_eq.copy()
            logw_mod[j] = logw_grid[k]
            logc_mod[j] = logc_grid[i]
            
            # Compute quantities using the same helper functions
            w_mod = np.exp(logw_mod)
            c_mod = np.exp(logc_mod)
            
            # Sort by cutoff cost
            idx = np.argsort(c_mod)
            c_sorted = c_mod[idx]
            rank = np.argsort(idx)
            
            # Compute truncated normal terms
            DeltaF, M = truncated_normal_column_terms(c_sorted, mu_s, sigma_s, eps)
            
            # Compute choice structure
            NUM = B * (w_mod[None, :] ** alpha)
            NUM_sorted = NUM[:, idx]
            DEN = np.cumsum(NUM_sorted, axis=1)
            DEN = np.maximum(DEN, eps)
            DEN_full = np.concatenate([DEN, DEN[:, -1:]], axis=1)
            f = support_weights[:, None] / np.maximum(DEN_full, eps)
            Gmat = NUM.T @ f
            
            # Weighted matrices with column-aligned mapping
            vL_cols = np.empty(J + 1, dtype=np.float64)
            vS_cols = np.empty(J + 1, dtype=np.float64)
            vL_cols[0:J] = DeltaF[1:J+1]  # columns 0..J-1 use intervals k = 1..J
            vS_cols[0:J] = DeltaF[1:J+1] * M[1:J+1]
            vL_cols[J] = DeltaF[J]  # column J duplicates interval k = J
            vS_cols[J] = DeltaF[J] * M[J]
            
            H_L = Gmat * vL_cols[None, :]
            H_S = Gmat * vS_cols[None, :]
            
            # Suffix sums with strict eligibility
            CumL = np.flip(np.cumsum(np.flip(H_L, axis=1), axis=1), axis=1)
            CumS = np.flip(np.cumsum(np.flip(H_S, axis=1), axis=1), axis=1)
            
            # Select for firm j (shares)
            j_new_pos = rank[j]
            start_col = j_new_pos + 1  # strict eligibility
            L_j_share = np.maximum(CumL[j, start_col], eps)
            S_j = np.maximum(CumS[j, start_col], eps)
            
            # Scale labor and compute profit
            L_j = N_workers * L_j_share
            Y_j = A[j] * (L_j * S_j) ** (1 - beta)
            w_j = np.exp(logw_grid[k])
            c_j = np.exp(logc_grid[i])
            Pi[i, k] = Y_j - w_j * L_j
    
    # Find argmax
    argmax_idx = np.unravel_index(np.argmax(Pi), Pi.shape)
    logw_argmax = logw_grid[argmax_idx[1]]
    logc_argmax = logc_grid[argmax_idx[0]]
    
    return W, C, Pi, logw_star, logc_star, logw_argmax, logc_argmax


def plot_profit_surface(firm_id: int, W: np.ndarray, C: np.ndarray, Pi: np.ndarray,
                       w_star: float, c_star: float, w_hat: float, c_hat: float,
                       out_path: str) -> str:
    """
    Plot profit surface for a firm.
    
    Args:
        firm_id: Firm ID
        W: Wage grid
        C: Cutoff cost grid
        Pi: Profit grid
        w_star: Equilibrium wage
        c_star: Equilibrium cutoff cost
        w_hat: Grid argmax wage
        c_hat: Grid argmax cutoff cost
        out_path: Output file path
        
    Returns:
        Path to written file
    """
    plt.figure(figsize=(10, 8))
    
    # Contour plot
    levels = np.linspace(Pi.min(), Pi.max(), 20)
    contour = plt.contourf(W, C, Pi, levels=levels, cmap='viridis')
    plt.colorbar(contour, label='Profit')
    
    # Mark equilibrium and argmax
    plt.plot(w_star, c_star, 'ro', markersize=10, label='Equilibrium')
    plt.plot(w_hat, c_hat, 'k*', markersize=15, label='Grid argmax')
    
    plt.xlabel('Wage (w_j)')
    plt.ylabel('Cutoff Cost (c_j)')
    plt.title(f'Profit Surface for Firm {firm_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return out_path


def make_profit_plots_for_firms(
    w: np.ndarray, c: np.ndarray, logw: np.ndarray, logc: np.ndarray,
    A: np.ndarray, xi: np.ndarray, loc_firms: np.ndarray,
    support_points: np.ndarray, support_weights: np.ndarray,
    mu_s: float, sigma_s: float, alpha: float, beta: float, gamma: float,
    firm_id: np.ndarray, out_dir: str, N_workers: float = 1.0, grid_n: int = 40, grid_log_span: float = 0.5,
    max_plots: Optional[int] = 12
) -> List[str]:
    """
    Generate profit surface plots for multiple firms.
    
    Args:
        w: Wages
        c: Cutoff costs
        logw: Log wages
        logc: Log cutoff costs
        A: Firm TFP
        xi: Firm amenity shocks
        loc_firms: Firm locations
        support_points: Worker support points
        support_weights: Worker weights
        mu_s: Mean of skill distribution
        sigma_s: Standard deviation of skill distribution
        alpha: Wage elasticity parameter
        beta: Production function parameter
        gamma: Distance decay parameter
        firm_id: Firm IDs
        out_dir: Output directory
        grid_n: Grid resolution
        grid_log_span: Log span for grid
        max_plots: Maximum number of plots
        
    Returns:
        List of written file paths
    """
    # Precompute base intensities
    B = precompute_bases(xi, loc_firms, support_points, gamma)
    
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which firms to plot
    if max_plots is None:
        firms_to_plot = range(len(firm_id))
    else:
        firms_to_plot = range(min(max_plots, len(firm_id)))
    
    plot_paths = []
    
    for j in firms_to_plot:
        print(f"Generating profit surface for firm {firm_id[j]}...")
        
        # Compute profit surface
        W, C, Pi, logw_star, logc_star, logw_argmax, logc_argmax = profit_surface_for_firm(
            j, logw, logc, A, xi, loc_firms, support_points, support_weights,
            mu_s, sigma_s, alpha, beta, gamma, B, N_workers, grid_n, grid_log_span
        )
        
        # Plot
        w_star = np.exp(logw_star)
        c_star = np.exp(logc_star)
        w_hat = np.exp(logw_argmax)
        c_hat = np.exp(logc_argmax)
        
        out_path = out_dir_path / f"firm_{firm_id[j]}_profit_surface.png"
        plot_path = plot_profit_surface(
            firm_id[j], W, C, Pi, w_star, c_star, w_hat, c_hat, out_path
        )
        
        print(f"  Saved to: {plot_path}")
        print(f"  Profit range: [{Pi.min():.6f}, {Pi.max():.6f}]")
        print(f"  Equilibrium: w={w_star:.4f}, c={c_star:.4f}")
        print(f"  Grid argmax: w={w_hat:.4f}, c={c_hat:.4f}")
        
        plot_paths.append(plot_path)
    
    return plot_paths


def solve_equilibrium_broyden(
    A: np.ndarray, xi: np.ndarray, loc_firms: np.ndarray,
    support_points: np.ndarray, support_weights: np.ndarray,
    mu_s: float, sigma_s: float, alpha: float, beta: float, gamma: float,
    N_workers: float = 1.0, conduct_mode: int = 1, eps_L_behavioral: Optional[np.ndarray] = None,
    eps_S_behavioral: Optional[np.ndarray] = None, max_iter: int = 2000, tol: float = 1e-8,
    eps: float = 1e-12
) -> Dict[str, Any]:
    """
    Solve the fixed-point equilibrium using Broyden's method from scipy.optimize.root.
    
    Args:
        A: Firm TFP (J,)
        xi: Firm amenity shocks (J,)
        loc_firms: Firm locations (J, 2)
        support_points: Worker support points (S, 2)
        support_weights: Worker weights (S,)
        mu_s: Mean of skill distribution
        sigma_s: Standard deviation of skill distribution
        alpha: Wage elasticity parameter
        beta: Production function parameter
        gamma: Distance decay parameter
        N_workers: Total number of workers in the market
        conduct_mode: 0=monopsonistic, 1=status quo, 2=behavioral elasticities
        eps_L_behavioral: Behavioral labor elasticities (J,) for conduct_mode=2
        eps_S_behavioral: Behavioral skill elasticities (J,) for conduct_mode=2
        max_iter: Maximum iterations
        tol: Convergence tolerance
        eps: Numerical safety floor
        
    Returns:
        Dictionary with solution and diagnostics
    """
    J = len(A)
    
    # Initialize with economically reasonable values
    # Start with wages that give positive employment
    logw_init = np.log(1 - beta) + np.log(A) + np.log(alpha / (alpha + 1)) + 2.0  # Large offset for stability
    logc_init = -np.log(1 - beta) + logw_init - np.log(A) + 1.0  # Large offset to cutoffs
    
    # Initial guess: concatenated [logw, logc]
    x0 = np.concatenate([logw_init, logc_init])
    
    # Define the function for scipy.optimize.root
    def objective_function(logx):
        return f_eqm(
            logx, A, xi, loc_firms, support_points, support_weights,
            mu_s, sigma_s, alpha, beta, gamma, eps, conduct_mode, N_workers,
            eps_L_behavioral, eps_S_behavioral
        )
    
    # Solve using Broyden's method
    print(f"Starting Broyden solver with {len(x0)} variables...")
    start_time = time.time()
    
    result = root(
        objective_function, 
        x0, 
        method='broyden1',
        options={'maxiter': max_iter, 'xtol': tol, 'disp': True}
    )
    
    total_time = time.time() - start_time
    
    if result.success:
        print(f"Broyden solver converged successfully!")
        converged = True
        final_residual = np.max(np.abs(result.fun))
        iterations = result.nit
    else:
        print(f"Broyden solver failed to converge: {result.message}")
        converged = False
        final_residual = np.max(np.abs(result.fun)) if result.fun is not None else np.inf
        iterations = result.nit
    
    # Extract solution
    logw_final = result.x[:J]
    logc_final = result.x[J:]
    w_final = np.exp(logw_final)
    c_final = np.exp(logc_final)
    
    # Compute final diagnostics using the last function evaluation
    try:
        _, _, final_diagnostics = fixed_point_map(
            logw_final, logc_final, A, xi, loc_firms, support_points, support_weights,
            mu_s, sigma_s, alpha, beta, gamma, eps, conduct_mode, N_workers,
            eps_L_behavioral, eps_S_behavioral
        )
        
        L_final = final_diagnostics['L']
        S_final = final_diagnostics['S']
        Y_final = final_diagnostics['Y']
        rank_final = final_diagnostics['rank']
        
    except Exception as e:
        print(f"Warning: Could not compute final diagnostics: {e}")
        L_final = np.zeros(J)
        S_final = np.zeros(J)
        Y_final = np.zeros(J)
        rank_final = np.arange(J)
        final_diagnostics = {}
    
    return {
        'converged': converged,
        'iters': iterations,
        'residual': final_residual,
        'time': total_time,
        'method': 'broyden',
        'L': L_final,
        'S': S_final,
        'w': w_final,
        'c': c_final,
        'Y': Y_final,
        'logw': logw_final,
        'logc': logc_final,
        'rank': rank_final,
        'diagnostics': final_diagnostics
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    data_dir = get_data_dir(create=True)
    parser = argparse.ArgumentParser(description="Worker Screening Equilibrium Solver")
    
    # Input files
    parser.add_argument(
        "--firms_path",
        type=str,
        default=str(data_dir / "firms.csv"),
        help="Path to firms CSV file",
    )
    parser.add_argument(
        "--support_path",
        type=str,
        default=str(data_dir / "support_points.csv"),
        help="Path to support points CSV file",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default=str(data_dir / "parameters_effective.csv"),
        help="Path to parameters CSV file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(data_dir),
        help="Output directory for equilibrium data (defaults to project data/ folder)",
    )
    
    # Solver parameters
    parser.add_argument("--conduct_mode", type=int, choices=[0, 1, 2], help="0=monopsonistic (model elasticities), 1=status quo, 2=behavioral elasticities ~ N(alpha,1) (overrides CSV)")
    parser.add_argument("--max_iter", type=int, help="Maximum solver iterations")
    parser.add_argument("--tol", type=float, help="Convergence tolerance")
    parser.add_argument("--damping", type=float, help="Fallback damping factor")
    parser.add_argument("--anderson_m", type=int, help="Anderson acceleration memory")
    parser.add_argument("--anderson_reg", type=float, help="Tikhonov regularization")
    parser.add_argument("--eps", type=float, help="Numerical safety floor")
    parser.add_argument("--method", type=str, choices=["fixed_point", "broyden"], default="fixed_point", 
                       help="Solver method: 'fixed_point' (default) or 'broyden'")
    
    # Profit surface parameters
    parser.add_argument("--plot_profits", action="store_true", help="Generate profit surface plots")
    parser.add_argument(
        "--drop_share_below",
        type=float,
        default=None,
        help="Drop firms whose worker share (L/N_workers) falls below this threshold; outside option is retained implicitly.",
    )
    parser.add_argument("--grid_n", type=int, help="Grid resolution for profit surfaces")
    parser.add_argument("--grid_log_span", type=float, help="Log span for profit surface grid")
    parser.add_argument("--max_plots", type=int, help="Maximum number of profit surface plots")
    
    args = parser.parse_args()
    
    print("Worker Screening Equilibrium Solver")
    print("=" * 50)
    
    # Read data
    print("Reading data...")
    firms_path = Path(args.firms_path)
    support_path = Path(args.support_path)
    params_path = Path(args.params_path)
    out_dir = Path(args.out_dir)

    try:
        A, xi, loc_firms, firm_id, comp, eps_L_behavioral, eps_S_behavioral = read_firms_csv(firms_path)
        firm_id_original = firm_id.copy()
        support_points, support_weights = read_support_points_csv(support_path)
        params = read_parameters_csv(params_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required CSV files exist:")
        print(f"  - {firms_path}")
        print(f"  - {support_path}")
        print(f"  - {params_path}")
        return 1
    
    print(f"Loaded {len(A)} firms and {len(support_points)} support points")
    print(f"Support weight sum: {support_weights.sum():.12f}")
    
    # Extract parameters
    mu_s = params['mu_s']
    sigma_s = params['sigma_s']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']

    # Extract N_workers with backward compatibility
    N_workers = float(params.get("N_workers", 1.0))
    if not np.isfinite(N_workers) or N_workers <= 0:
        raise ValueError(f"N_workers must be positive; got {N_workers}")

    support_counts = np.rint(support_weights * N_workers)
    positive_mask = support_counts > 0
    support_points = support_points[positive_mask]
    support_counts = support_counts[positive_mask]
    total_counts = support_counts.sum()
    if total_counts <= 0:
        raise ValueError("Rounded support weights sum to zero; adjust N_workers or quadrature configuration.")
    support_weights = support_counts / total_counts
    print(f"After rounding: {len(support_weights)} support points with positive mass (sum={support_weights.sum():.12f}, total_count={int(total_counts)})")
    
    # Solver parameters (with CLI overrides)
    max_iter = args.max_iter if args.max_iter is not None else params.get('max_iter', 2000)
    tol = args.tol if args.tol is not None else params.get('tol', 1e-8)
    damping = args.damping if args.damping is not None else params.get('damping', 0.5)
    anderson_m = args.anderson_m if args.anderson_m is not None else params.get('anderson_m', 5)
    anderson_reg = args.anderson_reg if args.anderson_reg is not None else params.get('anderson_reg', 1e-8)
    eps = args.eps if args.eps is not None else params.get('eps', 1e-12)
    
    # Conduct mode (CLI overrides CSV)
    conduct_mode = args.conduct_mode if args.conduct_mode is not None else params.get('conduct_mode', 1)
    
    # Solve equilibrium
    print(f"\nSolving equilibrium (conduct_mode={conduct_mode}, method={args.method})...")
    
    if args.method == "broyden":
        result = solve_equilibrium_broyden(
            A, xi, loc_firms, support_points, support_weights,
            mu_s, sigma_s, alpha, beta, gamma,
            N_workers=N_workers,
            conduct_mode=conduct_mode,
            eps_L_behavioral=eps_L_behavioral,
            eps_S_behavioral=eps_S_behavioral,
            max_iter=max_iter, tol=tol, eps=eps
        )
    else:  # args.method == "fixed_point"
        result = solve_equilibrium(
            A, xi, loc_firms, support_points, support_weights,
            mu_s, sigma_s, alpha, beta, gamma,
            N_workers=N_workers,
            conduct_mode=conduct_mode,
            eps_L_behavioral=eps_L_behavioral,
            eps_S_behavioral=eps_S_behavioral,
            max_iter=max_iter, tol=tol, damping=damping,
            anderson_m=anderson_m, anderson_reg=anderson_reg, eps=eps
        )
    
    # Print results
    print(f"\n=== EQUILIBRIUM RESULTS ===")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iters']}")
    print(f"Final residual: {result['residual']:.2e}")
    
    L, S, w, c, Y = result['L'], result['S'], result['w'], result['c'], result['Y']
    rank = result['rank']

    drop_threshold = args.drop_share_below
    drop_applied = False
    if drop_threshold is not None and drop_threshold > 0:
        total_workers = max(float(N_workers), 1e-12)
        shares = L 
        mask = shares >= drop_threshold
        dropped = int(np.size(L) - np.count_nonzero(mask))
        if dropped >= np.size(L):
            print(
                f"Warning: drop_share_below={drop_threshold:.2e} would remove all firms; skipping drop."
            )
        elif dropped > 0:
            print(
                f"Dropping {dropped} firms with worker share below {drop_threshold:.2e}."
            )
            w_masked = w[mask]
            c_masked = c[mask]

            A = A[mask]
            xi = xi[mask]
            loc_firms = loc_firms[mask]
            comp = comp[mask]
            firm_id_original = firm_id_original[mask]
            if eps_L_behavioral is not None:
                eps_L_behavioral = eps_L_behavioral[mask]
            if eps_S_behavioral is not None:
                eps_S_behavioral = eps_S_behavioral[mask]

            logw_masked = np.log(np.maximum(w_masked, 1e-300))
            logc_masked = np.log(np.maximum(c_masked, 1e-300))

            _, _, diagnostics_refit = fixed_point_map(
                logw_masked,
                logc_masked,
                A,
                xi,
                loc_firms,
                support_points,
                support_weights,
                mu_s,
                sigma_s,
                alpha,
                beta,
                gamma,
                eps,
                conduct_mode,
                N_workers,
                eps_L_behavioral,
                eps_S_behavioral,
            )

            w = np.exp(logw_masked)
            c = np.exp(logc_masked)
            L = diagnostics_refit['L']
            S = diagnostics_refit['S']
            Y = diagnostics_refit['Y']
            rank = diagnostics_refit['rank']

            shares = L
            firm_id = np.arange(1, L.size + 1, dtype=int)
            order_idx_new = np.argsort(c)
            rank = np.argsort(order_idx_new)

            diagnostics_refit['order_idx'] = order_idx_new
            diagnostics_refit['inv_order'] = np.argsort(order_idx_new)
            diagnostics_refit['rank'] = rank

            result['diagnostics'] = diagnostics_refit
            result['logw'] = logw_masked
            result['logc'] = logc_masked
            result['rank'] = rank
            result['L'] = L
            result['S'] = S
            result['w'] = w
            result['c'] = c
            result['Y'] = Y

            print(
                f"  Remaining firms: {L.size}; min share={shares.min():.3e}, max share={shares.max():.3e}"
            )
            drop_applied = True

    print(f"\nSolution bounds:")
    print(f"  L: [{np.min(L):.4f}, {np.max(L):.4f}]")
    print(f"  S: [{np.min(S):.4f}, {np.max(S):.4f}]")
    print(f"  w: [{np.min(w):.4f}, {np.max(w):.4f}]")
    print(f"  c: [{np.min(c):.4f}, {np.max(c):.4f}]")
    print(f"  Y: [{np.min(Y):.4f}, {np.max(Y):.4f}]")
    
    if conduct_mode == 0 and 'eps_L' in result['diagnostics']:
        eps_L = result['diagnostics']['eps_L']
        eps_S = result['diagnostics']['eps_S']
        print(f"\nElasticities:")
        print(f"  eps_L: [{np.min(eps_L):.4f}, {np.max(eps_L):.4f}]")
        print(f"  eps_S: [{np.min(eps_S):.4f}, {np.max(eps_S):.4f}]")
    elif conduct_mode == 2 and 'eps_L_behavioral' in result['diagnostics']:
        eps_L_behavioral = result['diagnostics']['eps_L_behavioral']
        eps_S_behavioral = result['diagnostics']['eps_S_behavioral']
        print(f"\nBehavioral Elasticities:")
        print(f"  eps_L_behavioral: [{np.min(eps_L_behavioral):.4f}, {np.max(eps_L_behavioral):.4f}]")
        print(f"  eps_S_behavioral: [{np.min(eps_S_behavioral):.4f}, {np.max(eps_S_behavioral):.4f}]")
    
    # Employment diagnostics
    from scipy.stats import norm
    c_sorted = np.sort(c)
    mass_above_min = 1.0 - norm.cdf((c_sorted[0] - mu_s)/sigma_s)
    weight_sum = float(support_weights.sum())
    print(f"\n[CHECK] Σ L (counts) = {L.sum():.6f}  vs expected ≈ {N_workers * mass_above_min * weight_sum:.6f}")
    print(f"[INFO] N_workers = {N_workers:.0f}, employment rate = {L.sum()/N_workers:.6%}")
    
    # Write results
    print(f"\nWriting results...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    equilibrium_path = write_equilibrium_csv(
        out_dir / "equilibrium_firms.csv",
        firm_id,
        comp,
        A,
        xi,
        loc_firms,
        w,
        c,
        L,
        S,
        Y,
        result['logw'],
        result['logc'],
        result['rank']
        #,firm_id_original=firm_id_original if drop_applied else None,
    )
    print(f"Equilibrium results written to: {equilibrium_path}")
    
    # Generate profit surface plots
    if args.plot_profits:
        print(f"\nGenerating profit surface plots...")
        print("NOTE: Profit surface plotting temporarily disabled due to integration changes.")
        print("TODO: Update profit surface functions to use new by-c model.")
        
        # TODO: Re-enable profit surface plotting after updating functions
        # plot_paths = make_profit_plots_for_firms(
        #     w, c, result['logw'], result['logc'],
        #     A, xi, loc_firms, support_points, support_weights,
        #     mu_s, sigma_s, alpha, beta, gamma, firm_id,
        #     out_dir / "profit_surfaces",
        #     N_workers=N_workers,
        #     grid_n=grid_n, grid_log_span=grid_log_span, max_plots=max_plots
        # )
        # print(f"Generated {len(plot_paths)} profit surface plots")
    
    # Print package versions
    print(f"\n=== PACKAGE VERSIONS ===")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"numba available: {NUMBA_AVAILABLE}")
    if NUMBA_AVAILABLE:
        print(f"numba: {numba.__version__}")
    
    return 0


if __name__ == "__main__":
    exit(main())
