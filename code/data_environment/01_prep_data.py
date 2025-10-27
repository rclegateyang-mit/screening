#!/usr/bin/env python3
"""
Worker Screening Data Preparation

This script generates firm data and worker quadrature points for the equilibrium solver.
It handles parameter management, firm fundamentals & locations, and Gaussian-Hermite quadrature.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List, Literal
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

try:
    from .. import get_data_dir
except ImportError:  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from code import get_data_dir  # type: ignore


# =============================================================================
# CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class WorkerDistConfig:
    """Worker location distribution parameters."""
    mu_x: float = 0.0
    mu_y: float = 2.0
    sigma_x: float = 5.0
    sigma_y: float = 4.0
    rho: float = 0.0  # correlation in [-1,1]; if ≠0, use Cholesky for Σ


@dataclass
class QuadratureConfig:
    """Gaussian-Hermite quadrature configuration."""
    kind: Literal["tensor_gh"] = "tensor_gh"
    n_x: int = 1        # nodes per dim (GH)
    n_y: int = 1
    normalize: bool = True  # ensure Σ weights = 1 (recommended)
    write_csv: bool = True  # write output/support_points.csv


@dataclass
class CoreParams:
    """Core economic model parameters."""
    # Skill distribution components
    mu_x_skill: float = 10.009
    sigma_x_skill: float = 2.60
    mu_a_skill: float = 0.0
    sigma_a_skill: float = 1.5
    # Derived skill parameters (computed from components)
    mu_s: float = 10.009  # Will be computed as mu_x_skill + mu_a_skill
    sigma_s: float = 3.0  # Will be computed as sqrt(sigma_x_skill^2 + sigma_a_skill^2)
    # Other model parameters
    alpha: float = 5.0
    beta: float = 0.2
    gamma: float = 0.05
    conduct_mode: int = 1   # 1 = status quo, 0 = elasticity-based
    N_workers: float = 100.0  # Total number of workers in the market
    
    def __post_init__(self):
        """Compute derived skill parameters from components."""
        import math
        self.mu_s = self.mu_x_skill + self.mu_a_skill
        self.sigma_s = math.sqrt(self.sigma_x_skill**2 + self.sigma_a_skill**2)


@dataclass
class FirmConfig:
    """Firm generation parameters."""
    J: int = 10
    # Fundamentals covariance
    sigma_A: float = 0.37
    sigma_xi: float = 0.2
    rho_Axi: float = 0.44
    # Location mixture parameters
    centers: np.ndarray = None  # (3, 2) array
    sds: np.ndarray = None      # (3, 2) array  
    weights: np.ndarray = None  # (3,) array
    
    def __post_init__(self):
        if self.centers is None:
            self.centers = np.array([[0.0, 0.0], [10.0, 4.0], [-8.0, 6.0]])
        if self.sds is None:
            self.sds = np.array([[2.2, 1.0], [3.0, 1.4], [2.6, 1.2]])
        if self.weights is None:
            self.weights = np.array([0.55, 0.25, 0.20])


@dataclass
class ModelConfig:
    """Complete model configuration."""
    core: CoreParams = None
    worker: WorkerDistConfig = None
    quad: QuadratureConfig = None
    firm: FirmConfig = None
    
    def __post_init__(self):
        if self.core is None:
            self.core = CoreParams()
        if self.worker is None:
            self.worker = WorkerDistConfig()
        if self.quad is None:
            self.quad = QuadratureConfig()
        if self.firm is None:
            self.firm = FirmConfig()


def write_parameters_template(path: str) -> None:
    """
    Write a human-friendly template with defaults + descriptions + units.
    
    Args:
        path: Path to write template CSV
    """
    template_data = [
        # Core parameters
        {'parameter': 'mu_s', 'value': 0.0, 'unit': 'miles', 'description': 'Mean of truncated normal skill distribution'},
        {'parameter': 'sigma_s', 'value': 1.0, 'unit': 'miles', 'description': 'Standard deviation of skill distribution'},
        {'parameter': 'alpha', 'value': 1.0, 'unit': 'NA', 'description': 'Wage elasticity parameter'},
        {'parameter': 'beta', 'value': 0.3, 'unit': 'NA', 'description': 'Production function parameter'},
        {'parameter': 'gamma', 'value': 0.5, 'unit': 'NA', 'description': 'Distance decay parameter'},
        {'parameter': 'conduct_mode', 'value': 1, 'unit': 'NA', 'description': '0=monopsonistic (model ε, ε^S), 1=status quo, 2=behavioral ε~N(alpha/sigma_s,sigma_s^2), ε^S~N(1, 0.25)'},
        {'parameter': 'N_workers', 'value': 100000.0, 'unit': 'workers', 'description': 'Total number of workers in the market'},
        
        # Worker distribution parameters
        {'parameter': 'worker_mu_x', 'value': 0.0, 'unit': 'miles', 'description': 'Mean x-coordinate of worker distribution'},
        {'parameter': 'worker_mu_y', 'value': 0.0, 'unit': 'miles', 'description': 'Mean y-coordinate of worker distribution'},
        {'parameter': 'worker_sigma_x', 'value': 2.0, 'unit': 'miles', 'description': 'Standard deviation of worker x-coordinate'},
        {'parameter': 'worker_sigma_y', 'value': 1.414, 'unit': 'miles', 'description': 'Standard deviation of worker y-coordinate'},
        {'parameter': 'worker_rho', 'value': 0.0, 'unit': 'NA', 'description': 'Correlation between worker x and y coordinates'},
        
        # Quadrature parameters
        {'parameter': 'quad_kind', 'value': 'tensor_gh', 'unit': 'NA', 'description': 'Quadrature method (tensor_gh only)'},
        {'parameter': 'quad_n_x', 'value': 15, 'unit': 'NA', 'description': 'Number of Gauss-Hermite nodes in x dimension'},
        {'parameter': 'quad_n_y', 'value': 15, 'unit': 'NA', 'description': 'Number of Gauss-Hermite nodes in y dimension'},
        {'parameter': 'quad_normalize', 'value': True, 'unit': 'NA', 'description': 'Normalize quadrature weights to sum to 1'},
        {'parameter': 'quad_write_csv', 'value': True, 'unit': 'NA', 'description': 'Write quadrature points to CSV'},
        
        # Firm generation parameters
        {'parameter': 'J', 'value': 3000, 'unit': 'NA', 'description': 'Number of firms'},
        {'parameter': 'sigma_A', 'value': 0.37, 'unit': 'NA', 'description': 'Standard deviation of log TFP'},
        {'parameter': 'sigma_xi', 'value': 0.2, 'unit': 'NA', 'description': 'Standard deviation of amenity shock'},
        {'parameter': 'rho_Axi', 'value': 0.44, 'unit': 'NA', 'description': 'Correlation between log TFP and amenity shock'},
    ]
    
    df = pd.DataFrame(template_data)
    df.to_csv(path, index=False)
    print(f"Parameters template written to: {path}")


def load_config(cli_overrides: Dict[str, Any]) -> Tuple[ModelConfig, pd.DataFrame]:
    """
    Load configuration with CLI overrides.
    
    Args:
        cli_overrides: Dictionary of CLI overrides
        
    Returns:
        Tuple of (config, effective_table) where effective_table shows key, value, source
    """
    # Start with defaults
    config = ModelConfig()
    
    # Apply CLI overrides
    # New skill distribution component parameters (take precedence)
    if 'mu_x_skill' in cli_overrides:
        config.core.mu_x_skill = cli_overrides['mu_x_skill']
    if 'sigma_x_skill' in cli_overrides:
        config.core.sigma_x_skill = cli_overrides['sigma_x_skill']
    if 'mu_a_skill' in cli_overrides:
        config.core.mu_a_skill = cli_overrides['mu_a_skill']
    if 'sigma_a_skill' in cli_overrides:
        config.core.sigma_a_skill = cli_overrides['sigma_a_skill']
    
    # Legacy parameters (for backward compatibility)
    if 'mu_s' in cli_overrides:
        config.core.mu_s = cli_overrides['mu_s']
    if 'sigma_s' in cli_overrides:
        config.core.sigma_s = cli_overrides['sigma_s']
    if 'alpha' in cli_overrides:
        config.core.alpha = cli_overrides['alpha']
    if 'beta' in cli_overrides:
        config.core.beta = cli_overrides['beta']
    if 'gamma' in cli_overrides:
        config.core.gamma = cli_overrides['gamma']
    if 'conduct_mode' in cli_overrides:
        config.core.conduct_mode = cli_overrides['conduct_mode']
    if 'N_workers' in cli_overrides:
        config.core.N_workers = cli_overrides['N_workers']
    
    if 'worker_mu_x' in cli_overrides:
        config.worker.mu_x = cli_overrides['worker_mu_x']
    if 'worker_mu_y' in cli_overrides:
        config.worker.mu_y = cli_overrides['worker_mu_y']
    if 'worker_sigma_x' in cli_overrides:
        config.worker.sigma_x = cli_overrides['worker_sigma_x']
    if 'worker_sigma_y' in cli_overrides:
        config.worker.sigma_y = cli_overrides['worker_sigma_y']
    if 'worker_rho' in cli_overrides:
        config.worker.rho = cli_overrides['worker_rho']
    
    if 'quad_n_x' in cli_overrides:
        config.quad.n_x = cli_overrides['quad_n_x']
    if 'quad_n_y' in cli_overrides:
        config.quad.n_y = cli_overrides['quad_n_y']
    if 'normalize' in cli_overrides:
        config.quad.normalize = cli_overrides['normalize']
    
    if 'J' in cli_overrides:
        config.firm.J = cli_overrides['J']
    if 'sigma_A' in cli_overrides:
        config.firm.sigma_A = cli_overrides['sigma_A']
    if 'sigma_xi' in cli_overrides:
        config.firm.sigma_xi = cli_overrides['sigma_xi']
    if 'rho_Axi' in cli_overrides:
        config.firm.rho_Axi = cli_overrides['rho_Axi']
    
    # Recompute derived parameters after applying CLI overrides
    config.core.__post_init__()
    
    # Create effective configuration table
    effective_data = []
    
    # Core parameters - skill distribution components
    effective_data.append({'parameter': 'mu_x_skill', 'value': config.core.mu_x_skill, 'source': 'CLI' if 'mu_x_skill' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'sigma_x_skill', 'value': config.core.sigma_x_skill, 'source': 'CLI' if 'sigma_x_skill' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'mu_a_skill', 'value': config.core.mu_a_skill, 'source': 'CLI' if 'mu_a_skill' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'sigma_a_skill', 'value': config.core.sigma_a_skill, 'source': 'CLI' if 'sigma_a_skill' in cli_overrides else 'DEFAULT'})
    # Core parameters - derived skill parameters (computed from components)
    effective_data.append({'parameter': 'mu_s', 'value': config.core.mu_s, 'source': 'COMPUTED'})
    effective_data.append({'parameter': 'sigma_s', 'value': config.core.sigma_s, 'source': 'COMPUTED'})
    # Legacy parameters for backward compatibility with GMM solver
    effective_data.append({'parameter': 'mu_a', 'value': config.core.mu_a_skill, 'source': 'COMPUTED'})
    effective_data.append({'parameter': 'sigma_a', 'value': config.core.sigma_a_skill, 'source': 'COMPUTED'})
    effective_data.append({'parameter': 'phi', 'value': 1.0, 'source': 'DEFAULT'})
    effective_data.append({'parameter': 'alpha', 'value': config.core.alpha, 'source': 'CLI' if 'alpha' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'beta', 'value': config.core.beta, 'source': 'CLI' if 'beta' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'gamma', 'value': config.core.gamma, 'source': 'CLI' if 'gamma' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'conduct_mode', 'value': config.core.conduct_mode, 'source': 'CLI' if 'conduct_mode' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'N_workers', 'value': config.core.N_workers, 'source': 'CLI' if 'N_workers' in cli_overrides else 'DEFAULT'})
    
    # Worker parameters
    effective_data.append({'parameter': 'worker_mu_x', 'value': config.worker.mu_x, 'source': 'CLI' if 'worker_mu_x' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_mu_y', 'value': config.worker.mu_y, 'source': 'CLI' if 'worker_mu_y' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_sigma_x', 'value': config.worker.sigma_x, 'source': 'CLI' if 'worker_sigma_x' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_sigma_y', 'value': config.worker.sigma_y, 'source': 'CLI' if 'worker_sigma_y' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_rho', 'value': config.worker.rho, 'source': 'CLI' if 'worker_rho' in cli_overrides else 'DEFAULT'})
    
    # Quadrature parameters
    effective_data.append({'parameter': 'quad_kind', 'value': config.quad.kind, 'source': 'DEFAULT'})
    effective_data.append({'parameter': 'quad_n_x', 'value': config.quad.n_x, 'source': 'CLI' if 'quad_n_x' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'quad_n_y', 'value': config.quad.n_y, 'source': 'CLI' if 'quad_n_y' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'quad_normalize', 'value': config.quad.normalize, 'source': 'CLI' if 'normalize' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'quad_write_csv', 'value': config.quad.write_csv, 'source': 'DEFAULT'})
    
    # Firm parameters
    effective_data.append({'parameter': 'J', 'value': config.firm.J, 'source': 'CLI' if 'J' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'sigma_A', 'value': config.firm.sigma_A, 'source': 'CLI' if 'sigma_A' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'sigma_xi', 'value': config.firm.sigma_xi, 'source': 'CLI' if 'sigma_xi' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'rho_Axi', 'value': config.firm.rho_Axi, 'source': 'CLI' if 'rho_Axi' in cli_overrides else 'DEFAULT'})
    
    effective_table = pd.DataFrame(effective_data)
    
    return config, effective_table


def dump_effective_config_csv(table: pd.DataFrame, out_path: str) -> None:
    """
    Write effective configuration to CSV so users see exactly what was used.
    
    Args:
        table: Effective configuration table
        out_path: Output file path
    """
    table.to_csv(out_path, index=False)
    print(f"Effective configuration written to: {out_path}")


# =============================================================================
# FIRM DATA GENERATION
# =============================================================================

def build_covariance(sigma_A: float, sigma_xi: float, rho: float) -> np.ndarray:
    """
    Build covariance matrix for firm fundamentals.
    
    Args:
        sigma_A: Standard deviation of log TFP
        sigma_xi: Standard deviation of amenity shock
        rho: Correlation between log TFP and amenity shock
        
    Returns:
        2x2 covariance matrix
    """
    cov = np.array([
        [sigma_A**2, rho * sigma_A * sigma_xi],
        [rho * sigma_A * sigma_xi, sigma_xi**2]
    ])
    
    # Validate positive definiteness
    eigenvals = np.linalg.eigvals(cov)
    if not np.all(eigenvals > 0):
        raise ValueError(f"Covariance matrix not positive definite. Eigenvalues: {eigenvals}")
    
    return cov


def draw_firm_fundamentals(J: int, cov: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """
    Draw firm fundamentals from bivariate normal distribution.
    
    Args:
        J: Number of firms
        cov: 2x2 covariance matrix
        rng: Random number generator
        
    Returns:
        DataFrame with firm_id, logA, A, xi
    """
    # Draw from multivariate normal
    fundamentals = rng.multivariate_normal(mean=[0, 0], cov=cov, size=J)
    logA, xi = fundamentals[:, 0], fundamentals[:, 1]
    
    # Compute A = exp(logA)
    A = np.exp(logA)
    
    # Create DataFrame
    df = pd.DataFrame({
        'firm_id': np.arange(1, J + 1),
        'logA': logA,
        'A': A,
        'xi': xi
    })
    
    return df


def draw_firm_locations(J: int, centers: np.ndarray, sds: np.ndarray, 
                       weights: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """
    Draw firm locations from Gaussian mixture.
    
    Args:
        J: Number of firms
        centers: Mixture centers (3, 2)
        sds: Component standard deviations (3, 2)
        weights: Mixture weights (3,)
        rng: Random number generator
        
    Returns:
        DataFrame with firm_id, comp, x, y
    """
    # Validate weights sum to 1
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError(f"Mixture weights must sum to 1, got {weights.sum()}")
    
    # Draw component labels
    comp = rng.choice(len(weights), size=J, p=weights)
    
    # Draw locations conditional on component
    locations = np.zeros((J, 2))
    for k in range(len(weights)):
        mask = comp == k
        if mask.sum() > 0:
            locations[mask] = rng.multivariate_normal(
                mean=centers[k], 
                cov=np.diag(sds[k]**2), 
                size=mask.sum()
            )
    
    # Create DataFrame
    df = pd.DataFrame({
        'firm_id': np.arange(1, J + 1),
        'comp': comp + 1,  # 1-indexed components
        'x': locations[:, 0],
        'y': locations[:, 1]
    })
    
    return df


# =============================================================================
# GAUSSIAN-HERMITE QUADRATURE
# =============================================================================

def gh_nodes_weights_std_normal(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D Gauss–Hermite nodes/weights mapped to N(0,1).
    
    Args:
        n: Number of quadrature points
        
    Returns:
        Tuple of (z, lam) with len n, where Σ lam = 1 to ≈ machine precision
    """
    x, w = hermgauss(n)              # weight e^{-x^2}
    z = np.sqrt(2.0) * x
    lam = w / np.sqrt(np.pi)
    lam = lam / lam.sum()            # normalize defensively
    return z, lam


def gh_tensor_2d(mu: np.ndarray, Sigma: np.ndarray, n_x: int, n_y: int, 
                normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 2D quadrature for ℓ ~ N(μ, Σ).
    
    Args:
        mu: Mean vector (2,)
        Sigma: Covariance matrix (2, 2)
        n_x: Number of nodes in x dimension
        n_y: Number of nodes in y dimension
        normalize: Whether to normalize weights to sum to 1
        
    Returns:
        Tuple of (points[S,2], weights[S]) with S = n_x * n_y
    """
    z1, l1 = gh_nodes_weights_std_normal(n_x)
    z2, l2 = gh_nodes_weights_std_normal(n_y)
    
    # Create tensor product
    Z = np.stack(np.meshgrid(z1, z2, indexing="xy"), axis=-1).reshape(-1, 2)    # S×2
    W = np.outer(l1, l2).reshape(-1)                                            # S
    
    # Cholesky decomposition for Σ
    L = np.linalg.cholesky(Sigma)
    P = (mu[None, :] + Z @ L.T)                                                 # points
    
    if normalize:
        W = W / W.sum()
    
    return P, W


def generate_support_points(config: ModelConfig, out_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gaussian-Hermite quadrature points from worker distribution parameters.
    
    Args:
        config: Model configuration
        out_path: Output file path
        
    Returns:
        Tuple of (points, weights)
    """
    # Build covariance matrix from config
    mu = np.array([config.worker.mu_x, config.worker.mu_y])
    Sigma = np.array([
        [config.worker.sigma_x**2, config.worker.rho * config.worker.sigma_x * config.worker.sigma_y],
        [config.worker.rho * config.worker.sigma_x * config.worker.sigma_y, config.worker.sigma_y**2]
    ])
    
    # Generate quadrature points
    points, weights = gh_tensor_2d(mu, Sigma, config.quad.n_x, config.quad.n_y, normalize=config.quad.normalize)
    
    # Adjust weights: scale by N_workers, round to nearest integer, renormalize
    weights_scaled = weights * config.core.N_workers
    counts = np.rint(weights_scaled)
    positive_mask = counts > 0
    counts = counts[positive_mask]
    points = points[positive_mask]
    total_counts = counts.sum()
    if total_counts <= 0:
        raise ValueError("Rounded quadrature weights sum to zero; adjust N_workers or quadrature configuration.")
    weights_adjusted = counts / total_counts

    # Write to CSV
    df = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'weight': weights_adjusted
    })
    df.to_csv(out_path, index=False)
    
    print(f"[QUAD] Generated {len(points)} quadrature points")
    print(f"[QUAD] Weight sum (after rounding): {weights_adjusted.sum():.12f}")
    print(f"[QUAD] Total integer mass: {int(total_counts)}")
    
    # Moment checks
    m = (weights_adjusted[:, None] * points).sum(0)
    C = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            C[i, j] = (weights_adjusted * (points[:, i] - m[i]) * (points[:, j] - m[j])).sum()
    
    print(f"[QUAD] E[ℓ] = {m}")
    print(f"[QUAD] Var[ℓ] = {np.diag(C)}")
    print(f"[QUAD] Cov[ℓ] = {C[0, 1]:.6f}")
    
    return points, weights


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    default_data_dir = get_data_dir(create=True)

    parser = argparse.ArgumentParser(description="Worker Screening Data Preparation")
    
    # Firm generation parameters
    parser.add_argument("--J", type=int, help="Number of firms")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    
    # Worker distribution parameters
    parser.add_argument("--worker_mu_x", type=float, help="Mean x-coordinate of worker distribution")
    parser.add_argument("--worker_mu_y", type=float, help="Mean y-coordinate of worker distribution")
    parser.add_argument("--worker_sigma_x", type=float, help="Standard deviation of worker x-coordinate")
    parser.add_argument("--worker_sigma_y", type=float, help="Standard deviation of worker y-coordinate")
    parser.add_argument("--worker_rho", type=float, help="Correlation between worker x and y coordinates")
    
    # Quadrature parameters
    parser.add_argument("--quad_n_x", type=int, help="Number of Gauss-Hermite nodes in x dimension")
    parser.add_argument("--quad_n_y", type=int, help="Number of Gauss-Hermite nodes in y dimension")
    parser.add_argument("--normalize", action="store_true", help="Normalize quadrature weights to sum to 1")
    
    # Core parameters - skill distribution components
    parser.add_argument("--mu_x_skill", type=float, help="Mean of x_skill component (default: 10.009)")
    parser.add_argument("--sigma_x_skill", type=float, help="Standard deviation of x_skill component (default: 2.60)")
    parser.add_argument("--mu_a_skill", type=float, help="Mean of a_skill component (default: 0.0)")
    parser.add_argument("--sigma_a_skill", type=float, help="Standard deviation of a_skill component (default: 1.5)")
    # Legacy parameters (for backward compatibility, will be overridden by computed values)
    parser.add_argument("--mu_s", type=float, help="DEPRECATED: Use --mu_x_skill and --mu_a_skill instead")
    parser.add_argument("--sigma_s", type=float, help="DEPRECATED: Use --sigma_x_skill and --sigma_a_skill instead")
    parser.add_argument("--alpha", type=float, help="Wage elasticity parameter")
    parser.add_argument("--beta", type=float, help="Production function parameter")
    parser.add_argument("--gamma", type=float, help="Distance decay parameter")
    parser.add_argument("--conduct_mode", type=int, choices=[0, 1, 2], help="0=monopsonistic (model elasticities), 1=status quo, 2=behavioral elasticities ~ N(1, 0.25)")
    parser.add_argument("--N_workers", type=float, help="Total number of workers in the market")
    
    # Firm fundamentals parameters
    parser.add_argument("--sigma_A", type=float, help="Standard deviation of log TFP")
    parser.add_argument("--sigma_xi", type=float, help="Standard deviation of amenity shock")
    parser.add_argument("--rho_Axi", type=float, help="Correlation between log TFP and amenity shock")
    
    # Output parameters
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(default_data_dir),
        help="Output directory for generated data (defaults to project data/ folder)",
    )
    
    args = parser.parse_args()
    
    print("Worker Screening Data Preparation")
    print("=" * 50)
    
    # Build CLI overrides dictionary
    cli_overrides = {}
    for param in ['J', 'mu_x_skill', 'sigma_x_skill', 'mu_a_skill', 'sigma_a_skill', 
                 'mu_s', 'sigma_s', 'alpha', 'beta', 'gamma', 'conduct_mode', 'N_workers',
                 'worker_mu_x', 'worker_mu_y', 'worker_sigma_x', 'worker_sigma_y', 'worker_rho',
                 'quad_n_x', 'quad_n_y', 'sigma_A', 'sigma_xi', 'rho_Axi']:
        value = getattr(args, param, None)
        if value is not None:
            cli_overrides[param] = value
    
    if args.normalize:
        cli_overrides['normalize'] = True
    
    # Load configuration
    config, effective_table = load_config(cli_overrides)
    
    # Set random seed
    rng = np.random.default_rng(args.seed)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration summary
    print("\n=== CONFIGURATION SUMMARY ===")
    print(f"[CONFIG] core:  mu_s={config.core.mu_s}, sigma_s={config.core.sigma_s}, alpha={config.core.alpha}, beta={config.core.beta}, gamma={config.core.gamma}, conduct_mode={config.core.conduct_mode}, N_workers={config.core.N_workers}")
    print(f"[CONFIG] worker: mu=({config.worker.mu_x}, {config.worker.mu_y}), sigma=({config.worker.sigma_x}, {config.worker.sigma_y}), rho={config.worker.rho}")
    print(f"[CONFIG] quad:   kind={config.quad.kind}, n_x={config.quad.n_x}, n_y={config.quad.n_y}, normalize={config.quad.normalize}")
    print(f"[CONFIG] firm:   J={config.firm.J}, sigma_A={config.firm.sigma_A}, sigma_xi={config.firm.sigma_xi}, rho_Axi={config.firm.rho_Axi}")
    
    # Dump effective configuration
    dump_effective_config_csv(effective_table, out_dir / "parameters_effective.csv")
    
    # Generate firm data
    print(f"\nGenerating {config.firm.J} firms...")
    
    # Build covariance matrix
    cov = build_covariance(config.firm.sigma_A, config.firm.sigma_xi, config.firm.rho_Axi)
    
    # Draw fundamentals
    fundamentals_df = draw_firm_fundamentals(config.firm.J, cov, rng)
    
    # Draw locations
    locations_df = draw_firm_locations(config.firm.J, config.firm.centers, config.firm.sds, config.firm.weights, rng)
    
    # Merge fundamentals and locations
    firms_df = pd.merge(fundamentals_df, locations_df, on='firm_id')
    
    # Add behavioral elasticities if conduct_mode=2
    if config.core.conduct_mode == 2:
        # Draw behavioral elasticities: eps_L ~ N(alpha, 1), eps_S ~ N(1, 0.25)
        mean_eps_L = 3
        std_eps_L = 1
        mean_eps_S = 1
        std_eps_S = 2
        
        eps_L_behavioral = rng.normal(mean_eps_L, std_eps_L, config.firm.J)
        eps_S_behavioral = rng.normal(mean_eps_S, std_eps_S, config.firm.J)
        
        # Add to firms DataFrame
        firms_df['eps_L_behavioral'] = eps_L_behavioral
        firms_df['eps_S_behavioral'] = eps_S_behavioral
        
        print(f"  Behavioral elasticities drawn:")
        print(f"    eps_L: mean={eps_L_behavioral.mean():.4f}, std={eps_L_behavioral.std():.4f}")
        print(f"    eps_S: mean={eps_S_behavioral.mean():.4f}, std={eps_S_behavioral.std():.4f}")
    
    # Sort by firm_id and reindex
    firms_df = firms_df.sort_values('firm_id').reset_index(drop=True)
    
    # Write firms CSV
    firms_path = out_dir / "firms.csv"
    firms_df.to_csv(firms_path, index=False)
    print(f"Firms data written to: {firms_path}")
    
    # Print firm summary statistics
    print(f"Firm summary:")
    print(f"  A: mean={firms_df['A'].mean():.4f}, std={firms_df['A'].std():.4f}")
    print(f"  xi: mean={firms_df['xi'].mean():.4f}, std={firms_df['xi'].std():.4f}")
    print(f"  Components: {firms_df['comp'].value_counts().sort_index().to_dict()}")
    print(f"  Spatial bounds: x=[{firms_df['x'].min():.2f}, {firms_df['x'].max():.2f}], y=[{firms_df['y'].min():.2f}, {firms_df['y'].max():.2f}]")
    
    # Generate support points
    print(f"\nGenerating worker quadrature points...")
    support_path = out_dir / "support_points.csv"
    points, weights = generate_support_points(config, support_path)
    print(f"Support points written to: {support_path}")
    
    # Write parameters CSV
    params_path = out_dir / "parameters.csv"
    write_parameters_template(params_path)
    print(f"Parameters template written to: {params_path}")
    
    print(f"\n=== OUTPUT FILES ===")
    print(f"  {firms_path}")
    print(f"  {support_path}")
    print(f"  {params_path}")
    print(f"  {out_dir / 'parameters_effective.csv'}")


if __name__ == "__main__":
    main()
