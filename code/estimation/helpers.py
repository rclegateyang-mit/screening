#!/usr/bin/env python3
"""
Common Helper Functions for Worker Screening Simulations

This module contains helper functions that are shared across multiple components
of the worker screening simulation project.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Any, Callable
import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import norm # Added for norm.cdf
from scipy.optimize import approx_fprime


# =============================================================================
# ORDER MAPPING FUNCTIONS
# =============================================================================

def compute_order_maps(c: np.ndarray) -> dict:
    """
    Create mapping between natural firm index and "by-c" order (ascending).
    
    Args:
        c: Cutoff costs array (J,)
        
    Returns:
        Dictionary containing:
        - order_idx (J,): natural → by-c index in ascending order
        - inv_order (J,): by-c position → natural firm id  
        - c_sorted (J,): c[order_idx]
    """
    J = len(c)
    
    # Sort by c (ascending)
    order_idx = np.argsort(c)  # natural → by-c index
    c_sorted = c[order_idx]    # c values in sorted order
    
    # Inverse mapping: by-c position → natural firm id
    inv_order = np.argsort(order_idx)  # by-c position → natural firm id
    
    return {
        'order_idx': order_idx,
        'inv_order': inv_order, 
        'c_sorted': c_sorted
    }


def reorder_to_natural(vec_sorted: np.ndarray, inv_order: np.ndarray) -> np.ndarray:
    """
    Reorder a vector from sorted-by-c order back to natural firm index order.
    
    Args:
        vec_sorted: Vector in sorted-by-c order (J,)
        inv_order: Inverse order mapping (J,) - by-c position → natural firm id
        
    Returns:
        Vector in natural firm index order (J,)
    """
    J = len(vec_sorted)
    assert len(inv_order) == J, f"Length mismatch: vec_sorted={J}, inv_order={len(inv_order)}"
    
    # Use inv_order to map back to natural order
    # inv_order[i] gives the natural firm ID for position i in sorted array
    vec_natural = vec_sorted[inv_order]
    
    return vec_natural


# =============================================================================
# INCLUSIVE VALUES AND CHOICE PROBABILITY FUNCTIONS
# =============================================================================

def inclusive_values_at_loc(ell: np.ndarray, w_sorted: np.ndarray, xi_sorted: np.ndarray, 
                           loc_sorted: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Compute inclusive values at location ℓ in by-c order.
    
    Args:
        ell: Worker location (2,)
        w_sorted: Wages in by-c order (J,)
        xi_sorted: Amenity shocks in by-c order (J,)
        loc_sorted: Firm locations in by-c order (J, 2)
        alpha: Wage elasticity parameter
        gamma: Distance decay parameter
        
    Returns:
        Inclusive values vector (J+1,) with v[0]=1.0 (outside option)
    """
    J = len(w_sorted)
    
    # Initialize inclusive values vector (J+1,)
    v = np.ones(J + 1, dtype=np.float64)
    
    # Compute distances from worker location to firm locations (vectorized)
    distances = np.linalg.norm(ell[None, :] - loc_sorted, axis=1)
    
    # Compute log wages with guards
    log_w = np.log(np.maximum(w_sorted, 1e-300))  # Guard against log(w) = -inf
    
    # Compute inclusive values for firms (j ≥ 1)
    # v_j = exp(-γ * dist + α * log(w_sorted) + xi_sorted)
    v[1:] = np.exp(-gamma * distances + alpha * log_w + xi_sorted)
    
    return v


def conditional_LM_at_loc(v: np.ndarray, p: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute location-conditional L and M vectors using suffix sums.
    
    Args:
        v: Inclusive values vector (J+1,) with v[0]=1.0
        p: Probability weights (J+1,)
        m: Mass values (J+1,)
        
    Returns:
        Tuple of (L_loc, M_loc) vectors, both (J+1,)
    """
    J_plus_1 = len(v)
    assert len(p) == J_plus_1 and len(m) == J_plus_1, f"Length mismatch: v={J_plus_1}, p={len(p)}, m={len(m)}"
    
    # Compute cumulative denominators
    denom = np.cumsum(v)  # length J+1
    
    # Compute ratios
    q = p / np.maximum(denom, 1e-300)  # Guard against division by zero
    r = m / np.maximum(denom, 1e-300)
    
    # Compute suffix cumulative sums
    # Q[k] = Σ_{i=k}^{J} q[i], R[k] = Σ_{i=k}^{J} r[i]
    Q = np.flip(np.cumsum(np.flip(q)))  # suffix sum of q
    R = np.flip(np.cumsum(np.flip(r)))  # suffix sum of r
    
    # Compute final vectors
    L_loc = v * Q
    M_loc = v * R
    
    return L_loc, M_loc


def compute_choice_probabilities_at_loc(v: np.ndarray, p_x: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """
    Compute choice probabilities P(x,ℓ) using suffix sums.
    
    Args:
        v: Inclusive values vector (J+1,) with v[0]=1.0
        p_x: Skill-based probabilities (J+1,)
        denom: Cumulative denominators (J+1,)
        
    Returns:
        Choice probabilities vector (J+1,)
    """
    J_plus_1 = len(v)
    assert len(p_x) == J_plus_1 and len(denom) == J_plus_1, f"Length mismatch: v={J_plus_1}, p_x={len(p_x)}, denom={len(denom)}"
    
    # Compute ratios
    q = p_x / np.maximum(denom, 1e-300)  # Guard against division by zero
    
    # Compute suffix cumulative sums
    Q = np.flip(np.cumsum(np.flip(q)))  # suffix sum of q
    
    # Compute final choice probabilities
    P = v * Q
    
    return P


def compute_p_x_probabilities(x_skill: float, c_sorted: np.ndarray, phi: float, mu_a: float, sigma_a: float) -> np.ndarray:
    """
    Compute skill-based probabilities p_x using truncated normal CDF differences.
    
    Args:
        x_skill: Worker skill level
        c_sorted: Sorted cutoff costs (J,)
        phi: Skill scaling parameter
        mu_a: Mean of preference shock
        sigma_a: Standard deviation of preference shock
        
    Returns:
        Probability vector (J+1,) for outside option and firms
    """
    J = len(c_sorted)
    
    # Build padded cutoff array with sentinels
    c_pad = np.empty(J + 2, dtype=np.float64)
    c_pad[0] = -np.inf
    c_pad[1:-1] = c_sorted
    c_pad[-1] = np.inf
    
    # Compute a(x) = φ·x + μ_a
    a_x = phi * x_skill + mu_a
    
    # Compute standardized thresholds
    z = (c_pad - a_x) / sigma_a
    
    # Compute CDF values
    Phi = special.ndtr(z)
    
    # Compute interval probabilities
    p_x = Phi[1:] - Phi[:-1]  # length J+1
    
    # Normalize to ensure sum = 1
    p_x = p_x / np.maximum(p_x.sum(), 1e-300)
    
    return p_x


def _numeric_jacobian_afp(theta, mvec_fn, *, rel_step=1e-6, abs_step=1e-5, bounds=None):
    """
    Compute D = ∂ m(θ) / ∂ θ' at 'theta' using scipy.optimize.approx_fprime.
    Forward differences with per-parameter epsilon; optional bounds shrink steps to remain feasible.
    """
    theta = np.asarray(theta, float).ravel()
    K = theta.size

    # Per-parameter epsilons
    eps = np.maximum(abs_step, rel_step * np.maximum(1.0, np.abs(theta)))

    # Bounds handling (θ + ε ≤ ub − δ); supports (lb, ub) arrays or list[(lb_k, ub_k)]
    if bounds is not None:
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and np.ndim(bounds[0]) == 1:
            lb = np.asarray(bounds[0], float); ub = np.asarray(bounds[1], float)
        else:
            lb = np.array([b[0] for b in bounds], float)
            ub = np.array([b[1] for b in bounds], float)
        delta = 1e-12
        room_up = ub - theta - delta
        eps = np.minimum(eps, np.maximum(room_up, 1e-16))  # keep positive, may be tiny near bounds

    # approx_fprime expects scalar output; loop over coordinates of m(θ)
    m0 = np.asarray(mvec_fn(theta), float).ravel()
    assert m0.size == K, f"m(theta) must be length {K}, got {m0.size}"
    D = np.empty((K, K), dtype=float)
    for k in range(K):
        def f_k(th):
            return float(np.asarray(mvec_fn(th), float).ravel()[k])
        D[k, :] = approx_fprime(theta, f_k, epsilon=eps)  # row k is ∂m_k/∂θ'
    return D


def compute_gmm_standard_errors(
    theta_hat: np.ndarray,
    moments_fn,                         # callable: theta -> (m_vec (K,), Psi (N,K))
    weighting: np.ndarray | None = None,# W used in estimation (default identity)
    mode: str = "robust",               # {"robust","efficient"}
    chamberlain_builder=None,           # callable: theta -> G (N,J,K)   [efficient only]
    prob_eval=None,                     # callable: theta -> P_full (N,J+1) [efficient only]
    Y_full=None,                        # (N,J+1) one-hot including outside [efficient only]
    bounds=None,                        # bounds for θ (e.g., γ∈[0,1], others free)
    h_abs: float = 1e-5,
    h_rel: float = 1e-6,
    ridge: float = 1e-12,
) -> dict:
    """
    Compute GMM standard errors at θ̂.

    Parameters
    ----------
    theta_hat : (K,) array_like
        Estimated parameter vector.
    moments_fn : callable
        Returns (m_vec, Psi) at θ for the estimator's moment system.
        m_vec ∈ ℝ^K is the sample mean of moments; Psi ∈ ℝ^{N×K} are per-observation moment rows.
    weighting : (K×K) array_like or None
        Weighting matrix W used in estimation. If None, use identity.
    mode : {"robust","efficient"}
        "robust": sandwich with supplied W on the estimator's moment system.
        "efficient": efficient GMM using Chamberlain instruments evaluated at θ̂ (θ₀=θ̂).
        Efficient SEs use the Fisher-information formula V ≈ (D' S_g^{-1} D)^{-1} / N with s_ij(θ̂) fixed.
        This corresponds to efficient GMM with Chamberlain optimal instruments.
    chamberlain_builder : callable
        Required if mode=="efficient". Returns instruments G(θ) with shape (N,J,K):
        s_ij,k(θ) = ∂/∂θ_k log(P_ij(θ)/P_i0(θ)).
    prob_eval : callable
        Required if mode=="efficient". Returns P_full(θ) with shape (N,J+1), outside at col 0.
    Y_full : ndarray
        Required if mode=="efficient". One-hot choices (N,J+1), outside at col 0.
    bounds : None or ((lb,), (ub,)) or list[(lb_k, ub_k)]
        Parameter bounds to keep forward steps feasible in approx_fprime.
    h_abs, h_rel : floats
        Step sizes for numerical derivatives of the mean moments via approx_fprime.
    ridge : float
        Tikhonov ridge added to the Bread matrix to stabilize inverses.

    Returns
    -------
    dict with:
      "vcov": (K,K) variance-covariance of θ̂
      "se":   (K,)  standard errors
      "bread","meat","S","D","W": matrices used
      "mode": str,  "N": int, "K": int
    """
    # Input validation
    if mode not in {"robust", "efficient"}:
        raise ValueError(f"mode must be 'robust' or 'efficient', got '{mode}'")
    
    # Coerce θ̂ and call moments function
    theta_hat = np.asarray(theta_hat, float).ravel()
    m_hat, Psi = moments_fn(theta_hat)
    N, K = Psi.shape
    assert Psi.shape == (N, K), f"Psi shape {Psi.shape} != (N, K) = ({N}, {K})"
    
    if mode == "robust":
        # Set weighting matrix
        if weighting is None:
            W = np.eye(K)
        else:
            W = np.asarray(weighting, float)
        
        # Long-run variance (i.i.d.): S = (Psi.T @ Psi) / N
        S = (Psi.T @ Psi) / N
        S = (S + S.T) / 2  # symmetrize S
        
        # Build mvec_fn(th) := moments_fn(th)[0] and compute Jacobian
        def mvec_fn(th):
            return moments_fn(th)[0]
        
        D = _numeric_jacobian_afp(theta_hat, mvec_fn, rel_step=h_rel, abs_step=h_abs, bounds=bounds)
        assert D.shape == (K, K), f"D shape {D.shape} != (K, K) = ({K}, {K})"
        
        # Bread B = D.T @ W @ D
        B = D.T @ W @ D
        
        # Meat M = D.T @ W @ S @ W @ D
        M = D.T @ W @ S @ W @ D
        
        # Add ridge regularization
        ridge_lambda = ridge * np.trace(B) / max(K, 1)
        if np.isfinite(ridge_lambda):
            B_reg = B + ridge_lambda * np.eye(K)
        else:
            B_reg = B + ridge * np.eye(K)
        
        # Compute inverse and variance
        B_inv = np.linalg.pinv(B_reg)
        V = (B_inv @ M @ B_inv) / N
        
        # Standard errors
        se = np.sqrt(np.diag(V).clip(min=0))
        
        return {
            "vcov": V,
            "se": se,
            "bread": B,
            "meat": M,
            "S": S,
            "D": D,
            "W": W,
            "mode": mode,
            "N": N,
            "K": K
        }
    
    elif mode == "efficient":
        # Preconditions: chamberlain_builder, prob_eval are callables and Y_full is provided
        if chamberlain_builder is None or prob_eval is None or Y_full is None:
            raise ValueError("efficient mode requires: chamberlain_builder, prob_eval, and Y_full")
        
        # Build instruments at θ̂
        G_hat = chamberlain_builder(theta_hat)  # shape (N,J,K)
        assert G_hat.ndim == 3 and G_hat.shape[2] == K, f"G_hat shape {G_hat.shape} != (N, J, K) where K={K}"
        N, J, _ = G_hat.shape
        
        # Compute probabilities and residuals at θ̂
        P_full = prob_eval(theta_hat)          # (N, J+1)
        P_full = np.clip(P_full, 1e-300, 1.0)
        R = Y_full[:, 1:] - P_full[:, 1:]      # (N, J)
        
        # Per-observation K-vector moments via instruments
        Psi_g = np.tensordot(R, G_hat, axes=([1], [1]))   # (N, K)
        m_hat = Psi_g.mean(axis=0)                        # (K,)
        
        # Optimal weighting
        S_g = (Psi_g.T @ Psi_g) / N
        S_g = 0.5 * (S_g + S_g.T)  # symmetrize
        W_star = np.linalg.pinv(S_g)
        
        # Numerical Jacobian of m(θ) with instruments **fixed at θ̂**
        G_fixed = G_hat
        def mvec_fn_eff(th):
            Pf = prob_eval(th)
            Pf = np.clip(Pf, 1e-300, 1.0)
            Rf = Y_full[:, 1:] - Pf[:, 1:]
            Psif = np.tensordot(Rf, G_fixed, axes=([1], [1]))  # (N,K)
            return Psif.mean(axis=0)
        
        D = _numeric_jacobian_afp(theta_hat, mvec_fn_eff, rel_step=h_rel, abs_step=h_abs, bounds=bounds)
        
        # Bread under optimal instruments/weighting:
        B = D.T @ W_star @ D
        # Ridge regularization for numerical stability:
        lam = ridge * (np.trace(B) / max(K, 1) if np.isfinite(np.trace(B)) else ridge)
        B_reg = B + lam * np.eye(K)
        B_inv = np.linalg.pinv(B_reg)

        # Fisher-information variance (no sandwich):
        V = B_inv / N
        se = np.sqrt(np.clip(np.diag(V), 0.0, np.inf))
        
        # Optional dev-only assertion for numerical stability
        assert np.max(np.abs(V - V.T)) < 1e-10, f"Variance matrix not symmetric: max|V-V'| = {np.max(np.abs(V - V.T))}"
        eigenvals = np.linalg.eigvals(V)
        assert np.all(eigenvals >= -1e-12), f"Variance matrix has negative eigenvalues: min = {np.min(eigenvals)}"
        
        return {
            "vcov": V, "se": se,
            "bread": B, "meat": None, "S": S_g, "D": D, "W": W_star,
            "mode": "efficient_fisher", "N": int(N), "K": int(K),
        }
    
    else:
        # This should never happen due to input validation above
        raise RuntimeError(f"Unexpected mode: {mode}")


# =============================================================================
# DATA I/O FUNCTIONS
# =============================================================================

def read_params_long_csv(path: str) -> Dict[str, Any]:
    """
    Read parameters from long/tidy CSV format and convert to dictionary.
    
    Args:
        path: Path to parameters CSV file
        
    Returns:
        Dictionary of parameter values with type casting
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
        
        # Attempt to cast to appropriate type
        try:
            # Try int first
            if param_value == int(param_value):
                params[param_name] = int(param_value)
            else:
                params[param_name] = float(param_value)
        except (ValueError, TypeError):
            # Keep as string if casting fails
            params[param_name] = str(param_value)
    
    return params


# =============================================================================
# GAUSS-HERMITE QUADRATURE FUNCTIONS
# =============================================================================

def gh_nodes_weights_std_normal(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gauss-Hermite nodes and weights for standard normal distribution.
    
    Args:
        n: Number of quadrature nodes
        
    Returns:
        Tuple of (nodes, weights) for N(0,1) distribution
    """
    # Get Gauss-Hermite nodes and weights
    x, w = np.polynomial.hermite.hermgauss(n)
    
    # Transform to standard normal: z = √2 * x, λ = w / √π
    z = np.sqrt(2) * x
    lambda_weights = w / np.sqrt(np.pi)
    
    # Renormalize weights to sum to 1
    lambda_weights = lambda_weights / lambda_weights.sum()
    
    return z, lambda_weights


def map_to_normal(mu: float, sigma: float, z: np.ndarray, lambda_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map standard normal nodes and weights to N(μ, σ²) distribution.
    
    Args:
        mu: Mean of target distribution
        sigma: Standard deviation of target distribution
        z: Standard normal nodes
        lambda_weights: Standard normal weights
        
    Returns:
        Tuple of (nodes, weights) for N(μ, σ²) distribution
    """
    # Transform nodes: x = μ + σ * z
    x_nodes = mu + sigma * z
    
    # Weights remain the same for linear transformation
    x_weights = lambda_weights.copy()
    
    return x_nodes, x_weights


# =============================================================================
# INTEGERIZATION FUNCTIONS
# =============================================================================

def integerize_poisson(count_float: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Convert float counts to integer counts using Poisson sampling.
    
    Args:
        count_float: Float counts array
        rng: Random number generator
        
    Returns:
        Integer counts array
    """
    return rng.poisson(lam=count_float)


def integerize_remainder(count_float: np.ndarray, N_workers: int, rng: np.random.Generator) -> np.ndarray:
    """
    Convert float counts to integer counts using remainder sampling.
    Ensures exact sum preservation: sum(count_int) = N_workers.
    
    Args:
        count_float: Float counts array
        N_workers: Total number of workers (target sum)
        rng: Random number generator
        
    Returns:
        Integer counts array that sums exactly to N_workers
    """
    # Start with floor values
    count_int = np.floor(count_float).astype(int)
    
    # Compute remaining workers to allocate
    remaining = N_workers - count_int.sum()
    
    if remaining > 0:
        # Compute fractional parts
        fractional_parts = count_float - count_int
        
        # Check if any fractional parts exist
        if np.sum(fractional_parts) > 1e-12:
            # Normalize fractional parts to probabilities
            probs = fractional_parts / fractional_parts.sum()
            
            # Sample which grid points get the extra workers
            n_grid = len(count_float)
            if n_grid > 0:
                sampled_indices = rng.multinomial(remaining, probs)
                count_int += sampled_indices
            else:
                # Edge case: no grid points
                count_int = np.array([remaining])
    
    return count_int


# =============================================================================
# SAFE MATHEMATICAL OPERATIONS
# =============================================================================

def safe_logs(x: np.ndarray, eps: float) -> np.ndarray:
    """
    Safe logarithm with floor at eps.
    
    Args:
        x: Input array
        eps: Safety floor
    
    Returns:
        log(max(x, eps))
    """
    return np.log(np.maximum(x, eps))


def load_weight_matrix(path: str | None, J: int) -> np.ndarray:
    """Load a J×J weight matrix from csv/json/npy; default identity."""
    if path is None:
        return np.eye(J, dtype=float)

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
    elif suffix == ".json":
        with open(p, "r") as f:
            arr = np.asarray(json.load(f), dtype=float)
    else:
        arr = pd.read_csv(p, header=None).values.astype(float)

    if arr.shape != (J, J):
        raise ValueError(f"Weight matrix must be {J}x{J}, got {arr.shape}.")

    arr = 0.5 * (arr + arr.T)
    return arr


# =============================================================================
# GMM ESTIMATION FUNCTIONS (moved from estimate_gmm_gamma_V.py)
# =============================================================================

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
                               w: np.ndarray, Y: np.ndarray, A: np.ndarray, c: np.ndarray, 
                               x_skill: np.ndarray, firm_ids: np.ndarray, 
                               beta: float, phi: float, mu_a: float, sigma_a: float,
                               V_nat: np.ndarray | None = None) -> np.ndarray:
    """
    Probability kernel per worker i for a given γ.
    
    Args:
        gamma: Distance decay parameter
        V: Firm attractiveness values (J,) in by-c order (for backward compatibility)
        distances: Worker-firm distances (N, J)
        w: Firm wages (J,)
        Y: Firm output (J,)
        A: Firm TFP (J,)
        c: Firm cutoffs (J,)
        x_skill: Worker skills (N,)
        firm_ids: Firm IDs (J,)
        beta: Production parameter
        phi: Skill scaling parameter
        mu_a: Mean of preference shock
        sigma_a: Std dev of preference shock
        V_nat: Optional firm attractiveness values (J,) in natural order
        
    Returns:
        P (N × J+1), rows workers, cols natural firm IDs.
    """
    N, J = distances.shape
    
    # 1) Compute cutoffs from firms (natural order)
    c_nat = compute_cutoffs(w, Y, A, beta)
    
    # 2) Order by c (ascending)
    order_maps = compute_order_maps(c_nat)
    order_idx = order_maps['order_idx']
    inv_order = order_maps['inv_order']
    c_sorted = order_maps['c_sorted']
    
    # 3) Reindex arrays to by-c order
    if V_nat is not None:
        # Use provided V_nat (natural order) and reindex to by-c order
        if len(V_nat) != J:
            raise ValueError(f"V_nat length {len(V_nat)} does not match number of firms {J}")
        V_sorted = V_nat[order_idx]
    else:
        # Fall back to previous default: use V (by-c order)
        V_sorted = V[order_idx]
    
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
    p_x = special.ndtr(z_hi) - special.ndtr(z_lo)  # (N, J+1)
    
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
    
    # 8) Keep all options including outside option (length J+1)
    P_all_byc = P_byc  # (N, J+1)
    
    # 9) Map back to natural order using firm IDs
    # Natural order: index 0=outside, index firm_id=firm
    P_all_nat = np.zeros((N, J + 1))

    # Front-append 0 to inv_order to include outside option at index 0
    inv_order_with_outside = np.concatenate([[0], inv_order+1])
    P_all_nat = P_all_byc[:, inv_order_with_outside]

    return P_all_nat


def create_probability_evaluator(V: np.ndarray, distances: np.ndarray, w: np.ndarray, 
                               Y: np.ndarray, A: np.ndarray, c: np.ndarray, x_skill: np.ndarray, 
                               firm_ids: np.ndarray, beta: float, phi: float, mu_a: float, sigma_a: float) -> Callable:
    """
    Return an evaluator that accepts either scalar γ or parameter vector θ.
    
    Returns:
        Callable that accepts:
        - (i) scalar gamma (backward compatibility), or
        - (ii) tuple (gamma, V_nat) or 1D array θ with θ[0]=gamma, θ[1:]=V_nat
        
    The evaluator returns P (N × J+1), rows workers, cols natural firm IDs.
    """
    J = len(firm_ids)
    
    def probs_worker_level(theta) -> np.ndarray:
        # Handle different input types
        if isinstance(theta, (int, float)):
            # Backward compatibility: scalar gamma
            gamma = float(theta)
            V_nat = None
        elif isinstance(theta, tuple) and len(theta) == 2:
            # Tuple (gamma, V_nat)
            gamma, V_nat = theta
            gamma = float(gamma)
            V_nat = np.asarray(V_nat)
        elif isinstance(theta, np.ndarray) and theta.ndim == 1:
            # 1D array θ with θ[0]=gamma, θ[1:]=V_nat
            if len(theta) == 1:
                # Single element array - treat as scalar gamma
                gamma = float(theta[0])
                V_nat = None
            elif len(theta) == J + 1:
                # Full parameter vector θ[0]=gamma, θ[1:]=V_nat
                gamma = float(theta[0])
                V_nat = theta[1:]
            else:
                raise ValueError(f"θ length {len(theta)} must be 1 (scalar gamma) or J+1={J+1} (gamma + V_nat for {J} firms)")
        else:
            raise ValueError(f"Invalid theta type: {type(theta)}. Expected scalar, tuple (gamma, V_nat), or 1D array θ[0]=gamma, θ[1:]=V_nat")
        
        # Validate V_nat shape if provided
        if V_nat is not None:
            if len(V_nat) != J:
                raise ValueError(f"V_nat length {len(V_nat)} does not match number of firms {J}")
        
        return compute_choice_probabilities(gamma, V, distances, w, Y, A, c, x_skill, 
                                          firm_ids, beta, phi, mu_a, sigma_a, V_nat)
    
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


def chamberlain_instruments_numeric(
    theta,
    prob_evaluator,
    *,
    rel_step: float = 1e-6,
    abs_step: float = 1e-5,
    bounds=None,
    eps: float = 1e-300,
    method: str = "3-point",
):
    """
    Compute Chamberlain optimal instruments via finite-difference Jacobian of
        F(θ) := vec( log P_{ij}(θ) - log P_{i0}(θ) )_{i=1..N, j=1..J}
    using scipy.optimize.approx_derivative.

    Parameters
    ----------
    theta : array_like (K,)
        Parameter vector (length K). Agnostic to contents (γ, V, ...).
    prob_evaluator : callable
        Returns P_full(θ) with shape (N, J+1): column 0 = outside, columns 1..J = firms.
        Signature: P = prob_evaluator(theta)
    rel_step : float
        Relative step passed to approx_derivative (componentwise).
    abs_step : float
        Absolute step floor passed to approx_derivative.
    bounds : None or (lb, ub) or list[tuple]
        Parameter bounds. If provided, will be coerced to arrays (lb, ub) of length K
        and passed to approx_derivative to pick central vs one-sided safely.
    eps : float
        Numerical floor for probabilities prior to log.
    method : {"3-point","2-point","cs"}
        Finite-difference scheme (default "3-point" central differences).

    Returns
    -------
    G : ndarray, shape (N, J, K)
        Instruments array with entries ∂/∂θ_k log(P_{ij}/P_{i0}) at θ.

    Example
    -------
    Build prob_evaluator(θ) returning P_full (N×(J+1)).
    Call G = chamberlain_instruments_numeric(θ, prob_evaluator, bounds=(lb,ub)).
    """
    # Coerce theta to 1D float64
    theta = np.asarray(theta, dtype=np.float64).ravel()
    K = len(theta)
    
    # Call prob_evaluator and assert shape
    P0 = prob_evaluator(theta)
    P0 = np.asarray(P0, dtype=np.float64)
    assert P0.ndim == 2, f"P0 must be 2D, got shape {P0.shape}"
    N, J_plus_1 = P0.shape
    assert J_plus_1 >= 2, f"P0 must have at least 2 columns (outside + 1 firm), got {J_plus_1}"
    J = J_plus_1 - 1
    
    # Assertions for data quality
    assert np.all(np.isfinite(P0)), "All probabilities must be finite"
    row_sums = np.sum(P0, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10), f"Row sums must be ≈ 1, got range [{np.min(row_sums):.2e}, {np.max(row_sums):.2e}]"
    
    # Define vector-valued function F(θ) returning stacked log-odds against outside
    def F(theta_vec):
        """F(θ) = vec( log P[:,1:] - log P[:,[0]] ) with shape (N*J,)"""
        theta_vec = np.asarray(theta_vec, dtype=np.float64).ravel()
        P = prob_evaluator(theta_vec)
        P = np.asarray(P, dtype=np.float64)
        
        # Extract outside and firm probabilities
        P_outside = P[:, 0:1]  # (N, 1) - keep 2D for broadcasting
        P_firms = P[:, 1:]     # (N, J)
        
        # Compute log probabilities with clipping for stability
        log_P_outside = np.log(np.clip(P_outside, eps, 1.0))
        log_P_firms = np.log(np.clip(P_firms, eps, 1.0))
        
        # Compute log-odds: log(P_firms / P_outside) = log(P_firms) - log(P_outside)
        log_odds = log_P_firms - log_P_outside  # (N, J)
        
        # Stack into vector: vec(log_odds) with shape (N*J,)
        return log_odds.ravel()
    
    # Coerce bounds to approx_derivative format
    if bounds is None:
        bounds_for_approx = None
    elif isinstance(bounds, (list, tuple)):
        if len(bounds) == 2 and isinstance(bounds[0], (list, tuple)) and isinstance(bounds[1], (list, tuple)):
            # bounds is [(lo1, hi1), (lo2, hi2), ...] per component
            lb = np.array([b[0] for b in bounds], dtype=np.float64)
            ub = np.array([b[1] for b in bounds], dtype=np.float64)
            bounds_for_approx = (lb, ub)
        else:
            # bounds is (lb, ub) with arrays
            bounds_for_approx = (np.asarray(bounds[0], dtype=np.float64), 
                               np.asarray(bounds[1], dtype=np.float64))
    else:
        # bounds is already in (lb, ub) format
        bounds_for_approx = bounds
    
    # Call approx_derivative to compute Jacobian
    JF = approx_fprime(theta, F)
    
    # JF has shape (N*J, K), reshape to G with shape (N, J, K)
    G = JF.reshape(N, J, K)
    
    # Final assertion
    assert np.all(np.isfinite(G)), "All derivatives must be finite"
    
    return G


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    N, J, K = 5, 3, 4
    X = rng.normal(size=(N, K))
    b = rng.normal(size=(K,))
    offs = np.linspace(-0.4, 0.4, J)

    def toy_eval(theta):
        theta = np.asarray(theta, float).ravel()
        z = X @ theta
        U = np.column_stack([np.zeros(N), z[:,None] + offs[None,:]])
        U -= U.max(axis=1, keepdims=True)
        P = np.exp(U); P /= P.sum(axis=1, keepdims=True)
        return P

    theta0 = rng.normal(size=(K,))
    G = chamberlain_instruments_numeric(theta0, toy_eval)
    print("G shape:", G.shape, "expected:", (N, J, K))
    assert G.shape == (N, J, K)


# =============================================================================
# NAIVE INITIAL THETA (γ, V, A) VIA MNL + SAMPLE AVERAGES
# =============================================================================

def _estimate_gamma_V_mnl(
    x_skill: np.ndarray,
    ell_x: np.ndarray,
    ell_y: np.ndarray,
    chosen_firm: np.ndarray,
    loc_firms: np.ndarray,
    *,
    gamma0: float = 0.05,
    gamma_bounds: tuple[float, float] = (0.0, 1.0),
    reg_V: float = 1e-6,
    maxiter: int = 500,
    verbose: bool = False,
) -> tuple[float, np.ndarray]:
    """Estimate (γ, V) via MNL using worker choices and distances."""

    # Local import to avoid polluting module namespace when unused
    from scipy.optimize import minimize

    x_skill = np.asarray(x_skill, float).ravel()
    ell_x = np.asarray(ell_x, float).ravel()
    ell_y = np.asarray(ell_y, float).ravel()
    chosen_firm = np.asarray(chosen_firm, int).ravel()
    loc_firms = np.asarray(loc_firms, float)

    N = x_skill.size
    J = loc_firms.shape[0]
    assert loc_firms.shape == (J, 2), f"loc_firms shape {loc_firms.shape} != (J,2)"

    D = compute_worker_firm_distances(ell_x, ell_y, loc_firms)

    counts_all = np.bincount(np.clip(chosen_firm, 0, J), minlength=J + 1)
    s0 = counts_all[0] / max(N, 1)
    s = counts_all[1:] / max(N, 1)
    eps = 1e-12
    V0 = np.log(np.maximum(s, eps)) - np.log(np.maximum(s0, eps))
    theta0 = np.concatenate(([float(gamma0)], V0))

    mask_in = chosen_firm > 0
    j_idx = (chosen_firm[mask_in] - 1).astype(int)
    d_chosen = D[mask_in, j_idx] if np.any(mask_in) else np.array([], dtype=float)
    counts_firms = np.bincount(j_idx, minlength=J) if np.any(mask_in) else np.zeros(J)

    def nll_and_grad(theta_vec: np.ndarray) -> tuple[float, np.ndarray]:
        theta_vec = np.asarray(theta_vec, float).ravel()
        gamma = float(theta_vec[0])
        V = theta_vec[1:]

        U = V[None, :] - gamma * D
        a = np.maximum(0.0, np.max(U, axis=1))
        exp_U_shift = np.exp(U - a[:, None])
        denom_scaled = np.exp(-a) + exp_U_shift.sum(axis=1)
        log_denom = a + np.log(denom_scaled)

        P = np.exp(U - log_denom[:, None])

        sum_log_denom = np.sum(log_denom)
        sum_V = float(V @ counts_firms)
        sum_d = float(np.sum(d_chosen))
        nll = sum_log_denom - (sum_V - gamma * sum_d) + 0.5 * reg_V * float(V @ V)

        g_V = P.sum(axis=0) - counts_firms + reg_V * V
        g_gamma = -float(np.sum(P * D)) + sum_d

        g = np.concatenate(([g_gamma], g_V.astype(float)))
        return nll, g

    bounds = [(gamma_bounds[0], gamma_bounds[1])] + [(-np.inf, np.inf)] * J
    res = minimize(
        fun=lambda th: nll_and_grad(th),
        x0=theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-10, "iprint": 1 if verbose else -1},
    )

    if verbose:
        print(f"[naive_theta_guess] success={res.success}, iters={res.nit}, nfev={res.nfev}, nll={res.fun:.6f}")

    theta_mnl = np.asarray(res.x, float).ravel()
    gamma_hat = float(theta_mnl[0])
    V_hat = theta_mnl[1:]
    return gamma_hat, V_hat


def naive_theta_guess_gamma_V_A(
    x_skill: np.ndarray,
    ell_x: np.ndarray,
    ell_y: np.ndarray,
    chosen_firm: np.ndarray,
    loc_firms: np.ndarray,
    firm_ids: np.ndarray,
    *,
    beta: float,
    gamma0: float = 0.05,
    gamma_bounds: tuple[float, float] = (0.0, 1.0),
    reg_V: float = 1e-6,
    maxiter: int = 500,
    verbose: bool = False,
    firms_csv_path: str = "output/equilibrium_firms.csv",
    params_csv_path: str = "output/parameters_effective.csv",
) -> np.ndarray:
    """
    Build a naive initial guess for θ = (γ, V_1..V_J, A_1..A_J) from worker data.

    - A_j: exp( log Y_j − (1−β)·log( x̄_j · L_j ) ), where
        x̄_j is average x_skill among workers who chose firm j (fallback to global mean),
        and (Y_j, L_j) are read from firms CSV (aligned by firm_id).
    - (γ, V_j): maximum-likelihood estimate from a standard multinomial logit where
      U_{ij} = V_j − γ·d_{ij}, outside option utility normalized to 0.

    Inputs are expected in NATURAL firm order (columns 1..J correspond to firm_ids 1..J).

    Args:
      x_skill: (N,) worker skills.
      ell_x, ell_y: (N,) worker locations.
      chosen_firm: (N,) choices with 0=outside and j∈firm_ids for firm j.
      loc_firms: (J,2) firm locations aligned with firm_ids ascending.
      firm_ids: (J,) array of economic firm identifiers (1..J).
      beta: production parameter β used in the A_j formula.
      gamma0: initial value for γ.
      gamma_bounds: bounds for γ in MNL estimation.
      reg_V: L2 ridge on V to stabilize rare choices (penalty 0.5·reg·||V||²).
      maxiter: optimizer iterations.
      verbose: print optimizer diagnostics.
      firms_csv_path: path to equilibrium firms CSV for (Y, L) lookup.

    Returns:
      theta0: (1+2J,) array [γ, V_1..V_J, A_1..A_J] in natural firm order.
    """
    x_skill = np.asarray(x_skill, float).ravel()
    ell_x = np.asarray(ell_x, float).ravel()
    ell_y = np.asarray(ell_y, float).ravel()
    chosen_firm = np.asarray(chosen_firm, int).ravel()
    loc_firms = np.asarray(loc_firms, float)
    firm_ids = np.asarray(firm_ids, int).ravel()

    N = x_skill.size
    J = firm_ids.size
    assert loc_firms.shape == (J, 2), f"loc_firms shape {loc_firms.shape} != (J,2)"

    gamma_hat, V_hat = _estimate_gamma_V_mnl(
        x_skill=x_skill,
        ell_x=ell_x,
        ell_y=ell_y,
        chosen_firm=chosen_firm,
        loc_firms=loc_firms,
        gamma0=gamma0,
        gamma_bounds=gamma_bounds,
        reg_V=reg_V,
        maxiter=maxiter,
        verbose=verbose,
    )

    # A_j via: log A_j = log Y_j − (1−β)·log( x̄_j · L_j )
    #          x̄_j = mean x_skill among workers choosing firm j (fallback: global mean)
    x_global_mean = float(np.mean(x_skill)) if N > 0 else 0.0
    xbar_by_firm = np.full(J, x_global_mean, dtype=float)
    for j in range(J):
        j_firm_id = j + 1
        mask_j = (chosen_firm == j_firm_id)
        if np.any(mask_j):
            xbar_by_firm[j] = float(np.mean(x_skill[mask_j]))

    # Read Y and L from firms CSV and align to firm_ids
    try:
        df_firms = pd.read_csv(firms_csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Firms CSV not found at '{firms_csv_path}'. Provide a valid path via firms_csv_path.") from e

    required_cols = {"firm_id", "Y", "L"}
    missing = required_cols - set(df_firms.columns)
    if missing:
        raise ValueError(f"Firms CSV missing required columns: {sorted(missing)}")

    # Ensure alignment by firm_id
    # Build a map from firm_id -> row index
    id_to_idx = {int(fid): idx for idx, fid in enumerate(df_firms.sort_values("firm_id")["firm_id"].values)}
    # Reindex Y and L into natural order 1..J as in firm_ids
    Y_vec = np.empty(J, dtype=float)
    L_vec = np.empty(J, dtype=float)
    for j, fid in enumerate(firm_ids):
        row = df_firms[df_firms["firm_id"] == int(fid)]
        if row.empty:
            raise ValueError(f"firm_id {fid} not found in {firms_csv_path}")
        Y_vec[j] = float(row.iloc[0]["Y"])  # production/output
        L_vec[j] = float(row.iloc[0]["L"])  # employment/mass of workers

    # Compute log A guess, then exponentiate to A levels
    eps = 1e-12
    logY = np.log(np.maximum(Y_vec, eps))
    # N from parameters_effective: scale L (a share) by total workers N
    try:
        params_eff = read_parameters(params_csv_path)
        N_param = float(params_eff.get('N_workers', float(N)))
    except Exception:
        # Fallback to sample size if parameters file missing/unreadable
        N_param = float(N)
    L_eff = np.maximum(L_vec, eps) * max(N_param, eps)
    log_xbarL = np.log(np.maximum(xbar_by_firm * L_eff, eps))
    logA_guess = logY - (1.0 - float(beta)) * log_xbarL
    A_guess = np.exp(logA_guess)

    # Stack final θ: [γ, V, A]
    theta_full = np.concatenate(([gamma_hat], V_hat, A_guess))
    return theta_full


def naive_theta_guess_gamma_V_c(
    x_skill: np.ndarray,
    ell_x: np.ndarray,
    ell_y: np.ndarray,
    chosen_firm: np.ndarray,
    loc_firms: np.ndarray,
    firm_ids: np.ndarray,
    *,
    gamma0: float = 0.05,
    gamma_bounds: tuple[float, float] = (0.0, 1.0),
    reg_V: float = 1e-6,
    maxiter: int = 500,
    verbose: bool = False,
    firms_csv_path: str = "output/equilibrium_firms.csv",
) -> np.ndarray:
    """Naive θ₀ = (γ, V, c) using only choices, locations, and firm cutoffs."""

    x_skill = np.asarray(x_skill, float).ravel()
    ell_x = np.asarray(ell_x, float).ravel()
    ell_y = np.asarray(ell_y, float).ravel()
    chosen_firm = np.asarray(chosen_firm, int).ravel()
    loc_firms = np.asarray(loc_firms, float)
    firm_ids = np.asarray(firm_ids, int).ravel()

    gamma_hat, V_hat = _estimate_gamma_V_mnl(
        x_skill=x_skill,
        ell_x=ell_x,
        ell_y=ell_y,
        chosen_firm=chosen_firm,
        loc_firms=loc_firms,
        gamma0=gamma0,
        gamma_bounds=gamma_bounds,
        reg_V=reg_V,
        maxiter=maxiter,
        verbose=verbose,
    )

    try:
        df_firms = pd.read_csv(firms_csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Firms CSV not found at '{firms_csv_path}'. Provide a valid path via firms_csv_path."
        ) from e

    if 'firm_id' not in df_firms.columns or 'c' not in df_firms.columns:
        raise ValueError("Firms CSV must contain 'firm_id' and 'c' columns to build cutoff guesses.")

    df_sorted = df_firms.sort_values('firm_id').reset_index(drop=True)
    id_to_idx = {int(fid): idx for idx, fid in enumerate(df_sorted['firm_id'].values)}

    c_guess = np.empty(firm_ids.size, dtype=float)
    for idx, fid in enumerate(firm_ids):
        if int(fid) not in id_to_idx:
            raise ValueError(f"firm_id {fid} not found in {firms_csv_path}")
        row = df_sorted.iloc[id_to_idx[int(fid)]]
        c_val = float(row['c'])
        if c_val <= 0.0:
            c_val = float(np.nan)
        c_guess[idx] = c_val

    if np.any(~np.isfinite(c_guess)):
        raise ValueError("Encountered non-positive or missing cutoffs when reading firms CSV.")

    theta_full = np.concatenate(([gamma_hat], V_hat, c_guess))
    return theta_full
