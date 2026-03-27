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
    from .. import get_data_subdir, DATA_RAW
except ImportError:  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from screening import get_data_subdir, DATA_RAW  # type: ignore


# =============================================================================
# CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class WorkerDistConfig:
    """Worker location distribution parameters."""
    mu_x: float = 0.0
    mu_y: float = 0.0
    sigma_x: float = 5.0
    sigma_y: float = 4.0
    rho: float = 0.0  # correlation in [-1,1]; if ≠0, use Cholesky for Σ
    loc_mode: str = "cartesian"  # cartesian or polar
    r_mu: float = 0.0
    r_sigma: float = 1.0


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
    """Core economic model parameters.

    Log-additive skill specification:
        ln q_i = gamma0 + gamma1 * v_i + e_i
        v_i ~ N(mu_v, sigma_v^2)        observable component
        e_i ~ N(0, sigma_e^2)           unobservable component
    """
    # Skill distribution components (log-additive)
    gamma0: float = 0.76
    gamma1: float = 0.94
    mu_v: float = 12.0
    sigma_v: float = 0.25
    sigma_e: float = 0.135
    rho_x_skill_ell_x: float = 0.0
    rho_x_skill_ell_y: float = 0.0
    rho_x_skill_r: float = 0.0
    # Derived skill parameters (computed from components)
    mu_lnq: float = 0.0   # Will be computed as gamma0 + gamma1 * mu_v
    sigma_lnq: float = 1.0  # Will be computed as sqrt(gamma1^2 * sigma_v^2 + sigma_e^2)
    # Other model parameters
    eta: float = 5.0
    alpha: float = 0.2
    tau: float = 0.05
    conduct_mode: int = 1   # 1 = status quo, 0 = elasticity-based
    N_workers: float = 100.0  # Total number of workers in the market

    def __post_init__(self):
        """Compute derived skill parameters from components."""
        import math
        self.mu_lnq = self.gamma0 + self.gamma1 * self.mu_v
        self.sigma_lnq = math.sqrt(self.gamma1**2 * self.sigma_v**2 + self.sigma_e**2)


@dataclass
class FirmConfig:
    """Firm generation parameters."""
    J: int = 10
    mu_xi: float = 0.0
    # Fundamentals covariance
    sigma_A: float = 0.37
    sigma_xi: float = 0.2
    rho_Axi: float = 0.44
    # Observed instrument standard deviations
    sigma_z1: float = 0.2   # TFP shifter
    sigma_z2: float = 0.1   # amenity shifter
    # Location mixture parameters
    centers: np.ndarray = None  # (3, 2) array
    sds: np.ndarray = None      # (3, 2) array  
    weights: np.ndarray = None  # (3,) array
    
    def __post_init__(self):
        if self.centers is None:
            self.centers = np.array([[0.0, 0.0], [10.0, 4.0], [-8.0, 6.0]])
        if self.sds is None:
            self.sds = np.array([[3.3, 1.5], [4.5, 2.1], [3.9, 1.8]])
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
        {'parameter': 'eta', 'value': 1.0, 'unit': 'NA', 'description': 'Wage elasticity parameter'},
        {'parameter': 'alpha', 'value': 0.3, 'unit': 'NA', 'description': 'Production function parameter'},
        {'parameter': 'tau', 'value': 0.5, 'unit': 'NA', 'description': 'Distance decay / commuting cost parameter'},
        {'parameter': 'conduct_mode', 'value': 1, 'unit': 'NA', 'description': '0=monopsonistic (model ε, ε^S), 1=status quo, 2=behavioral ε~N(eta/sigma_s,sigma_s^2), ε^S~N(1, 0.25)'},
        {'parameter': 'N_workers', 'value': 100000.0, 'unit': 'workers', 'description': 'Total number of workers in the market'},
        {'parameter': 'rho_x_skill_ell_x', 'value': 0.0, 'unit': 'NA', 'description': 'Correlation between x_skill and worker x-location'},
        {'parameter': 'rho_x_skill_ell_y', 'value': 0.0, 'unit': 'NA', 'description': 'Correlation between x_skill and worker y-location'},
        {'parameter': 'rho_x_skill_r', 'value': 0.0, 'unit': 'NA', 'description': 'Correlation between x_skill and distance from worker mean location'},
        
        # Worker distribution parameters
        {'parameter': 'worker_loc_mode', 'value': 'cartesian', 'unit': 'NA', 'description': 'Worker location distribution: cartesian or polar'},
        {'parameter': 'worker_mu_x', 'value': 0.0, 'unit': 'miles', 'description': 'Mean x-coordinate of worker distribution'},
        {'parameter': 'worker_mu_y', 'value': 0.0, 'unit': 'miles', 'description': 'Mean y-coordinate of worker distribution'},
        {'parameter': 'worker_sigma_x', 'value': 2.0, 'unit': 'miles', 'description': 'Standard deviation of worker x-coordinate'},
        {'parameter': 'worker_sigma_y', 'value': 1.414, 'unit': 'miles', 'description': 'Standard deviation of worker y-coordinate'},
        {'parameter': 'worker_rho', 'value': 0.0, 'unit': 'NA', 'description': 'Correlation between worker x and y coordinates'},
        {'parameter': 'worker_r_mu', 'value': 0.0, 'unit': 'miles', 'description': 'Mean radius for polar worker location distribution'},
        {'parameter': 'worker_r_sigma', 'value': 1.0, 'unit': 'miles', 'description': 'Std deviation of radius for polar worker location distribution'},
        
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
        {'parameter': 'mu_xi', 'value': 0.0, 'unit': 'NA', 'description': 'Mean of amenity shock'},
        {'parameter': 'sigma_z1', 'value': 0.2, 'unit': 'NA', 'description': 'Std dev of observed TFP shifter (instrument for A)'},
        {'parameter': 'sigma_z2', 'value': 0.1, 'unit': 'NA', 'description': 'Std dev of observed amenity shifter (instrument for xi)'},
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
    # Skill distribution component parameters (log-additive)
    if 'gamma0' in cli_overrides:
        config.core.gamma0 = cli_overrides['gamma0']
    if 'gamma1' in cli_overrides:
        config.core.gamma1 = cli_overrides['gamma1']
    if 'mu_v' in cli_overrides:
        config.core.mu_v = cli_overrides['mu_v']
    if 'sigma_v' in cli_overrides:
        config.core.sigma_v = cli_overrides['sigma_v']
    if 'sigma_e' in cli_overrides:
        config.core.sigma_e = cli_overrides['sigma_e']
    if 'rho_x_skill_ell_x' in cli_overrides:
        config.core.rho_x_skill_ell_x = cli_overrides['rho_x_skill_ell_x']
    if 'rho_x_skill_ell_y' in cli_overrides:
        config.core.rho_x_skill_ell_y = cli_overrides['rho_x_skill_ell_y']
    if 'rho_x_skill_r' in cli_overrides:
        config.core.rho_x_skill_r = cli_overrides['rho_x_skill_r']
    
    if 'eta' in cli_overrides:
        config.core.eta = cli_overrides['eta']
    if 'alpha' in cli_overrides:
        config.core.alpha = cli_overrides['alpha']
    if 'tau' in cli_overrides:
        config.core.tau = cli_overrides['tau']
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
    if 'worker_loc_mode' in cli_overrides:
        config.worker.loc_mode = cli_overrides['worker_loc_mode']
    if 'worker_r_mu' in cli_overrides:
        config.worker.r_mu = cli_overrides['worker_r_mu']
    if 'worker_r_sigma' in cli_overrides:
        config.worker.r_sigma = cli_overrides['worker_r_sigma']
    
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
    if 'mu_xi' in cli_overrides:
        config.firm.mu_xi = cli_overrides['mu_xi']
    if 'sigma_z1' in cli_overrides:
        config.firm.sigma_z1 = cli_overrides['sigma_z1']
    if 'sigma_z2' in cli_overrides:
        config.firm.sigma_z2 = cli_overrides['sigma_z2']

    # Recompute derived parameters after applying CLI overrides
    config.core.__post_init__()
    
    # Create effective configuration table
    effective_data = []
    
    # Core parameters - skill distribution (log-additive)
    effective_data.append({'parameter': 'gamma0', 'value': config.core.gamma0, 'source': 'CLI' if 'gamma0' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'gamma1', 'value': config.core.gamma1, 'source': 'CLI' if 'gamma1' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'mu_v', 'value': config.core.mu_v, 'source': 'CLI' if 'mu_v' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'sigma_v', 'value': config.core.sigma_v, 'source': 'CLI' if 'sigma_v' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'sigma_e', 'value': config.core.sigma_e, 'source': 'CLI' if 'sigma_e' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'rho_x_skill_ell_x', 'value': config.core.rho_x_skill_ell_x, 'source': 'CLI' if 'rho_x_skill_ell_x' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'rho_x_skill_ell_y', 'value': config.core.rho_x_skill_ell_y, 'source': 'CLI' if 'rho_x_skill_ell_y' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'rho_x_skill_r', 'value': config.core.rho_x_skill_r, 'source': 'CLI' if 'rho_x_skill_r' in cli_overrides else 'DEFAULT'})
    # Derived skill parameters
    effective_data.append({'parameter': 'mu_lnq', 'value': config.core.mu_lnq, 'source': 'COMPUTED'})
    effective_data.append({'parameter': 'sigma_lnq', 'value': config.core.sigma_lnq, 'source': 'COMPUTED'})
    effective_data.append({'parameter': 'eta', 'value': config.core.eta, 'source': 'CLI' if 'eta' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'alpha', 'value': config.core.alpha, 'source': 'CLI' if 'alpha' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'tau', 'value': config.core.tau, 'source': 'CLI' if 'tau' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'conduct_mode', 'value': config.core.conduct_mode, 'source': 'CLI' if 'conduct_mode' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'N_workers', 'value': config.core.N_workers, 'source': 'CLI' if 'N_workers' in cli_overrides else 'DEFAULT'})
    
    # Worker parameters
    effective_data.append({'parameter': 'worker_loc_mode', 'value': config.worker.loc_mode, 'source': 'CLI' if 'worker_loc_mode' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_mu_x', 'value': config.worker.mu_x, 'source': 'CLI' if 'worker_mu_x' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_mu_y', 'value': config.worker.mu_y, 'source': 'CLI' if 'worker_mu_y' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_sigma_x', 'value': config.worker.sigma_x, 'source': 'CLI' if 'worker_sigma_x' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_sigma_y', 'value': config.worker.sigma_y, 'source': 'CLI' if 'worker_sigma_y' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_rho', 'value': config.worker.rho, 'source': 'CLI' if 'worker_rho' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_r_mu', 'value': config.worker.r_mu, 'source': 'CLI' if 'worker_r_mu' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'worker_r_sigma', 'value': config.worker.r_sigma, 'source': 'CLI' if 'worker_r_sigma' in cli_overrides else 'DEFAULT'})
    
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
    effective_data.append({'parameter': 'mu_xi', 'value': config.firm.mu_xi, 'source': 'CLI' if 'mu_xi' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'sigma_z1', 'value': config.firm.sigma_z1, 'source': 'CLI' if 'sigma_z1' in cli_overrides else 'DEFAULT'})
    effective_data.append({'parameter': 'sigma_z2', 'value': config.firm.sigma_z2, 'source': 'CLI' if 'sigma_z2' in cli_overrides else 'DEFAULT'})

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


def draw_firm_fundamentals(J: int, cov: np.ndarray, rng: np.random.Generator,
                           mu_xi: float = 0.0,
                           sigma_z1: float = 0.0, sigma_z2: float = 0.0) -> pd.DataFrame:
    """
    Draw firm fundamentals with observed instrument decomposition.

    logA_j = z1_j + nu_j^A,   xi_j = z2_j + nu_j^xi
    where z1 ~ N(0, sigma_z1^2), z2 ~ N(0, sigma_z2^2) independently,
    and (nu^A, nu^xi) ~ BivariateNormal(0, Sigma_nu) with Sigma_nu
    chosen so the marginal (logA, xi) distribution matches ``cov``.

    Args:
        J: Number of firms
        cov: 2x2 covariance matrix for total (logA, xi)
        rng: Random number generator
        mu_xi: Mean of amenity shock xi
        sigma_z1: Std dev of observed TFP shifter
        sigma_z2: Std dev of observed amenity shifter

    Returns:
        DataFrame with firm_id, logA, A, xi, z1, z2
    """
    sigma_A = np.sqrt(cov[0, 0])
    sigma_xi = np.sqrt(cov[1, 1])
    rho_Axi = cov[0, 1] / (sigma_A * sigma_xi) if sigma_A > 0 and sigma_xi > 0 else 0.0

    if sigma_z1 > 0 or sigma_z2 > 0:
        # Validate instrument variances don't exceed total variances
        if sigma_z1 >= sigma_A:
            raise ValueError(
                f"sigma_z1 ({sigma_z1}) must be < sigma_A ({sigma_A}) "
                f"so that residual variance sigma_nuA^2 > 0")
        if sigma_z2 >= sigma_xi:
            raise ValueError(
                f"sigma_z2 ({sigma_z2}) must be < sigma_xi ({sigma_xi}) "
                f"so that residual variance sigma_nuxi^2 > 0")

        # Residual standard deviations
        sigma_nuA = np.sqrt(sigma_A**2 - sigma_z1**2)
        sigma_nuxi = np.sqrt(sigma_xi**2 - sigma_z2**2)

        # Residual correlation must satisfy |rho_nu| <= 1
        rho_nu = rho_Axi * sigma_A * sigma_xi / (sigma_nuA * sigma_nuxi)
        if abs(rho_nu) > 1.0:
            raise ValueError(
                f"Residual correlation |rho_nu| = {abs(rho_nu):.4f} > 1. "
                f"Reduce sigma_z1/sigma_z2 or increase sigma_A/sigma_xi.")

        # Draw observed instruments independently
        z1 = rng.normal(0.0, sigma_z1, size=J)
        z2 = rng.normal(0.0, sigma_z2, size=J)

        # Draw residuals from bivariate normal
        cov_nu = np.array([
            [sigma_nuA**2, rho_nu * sigma_nuA * sigma_nuxi],
            [rho_nu * sigma_nuA * sigma_nuxi, sigma_nuxi**2]
        ])
        residuals = rng.multivariate_normal(mean=[0.0, mu_xi], cov=cov_nu, size=J)
        nu_A, nu_xi = residuals[:, 0], residuals[:, 1]

        logA = z1 + nu_A
        xi = z2 + nu_xi
    else:
        # No instruments — original path
        fundamentals = rng.multivariate_normal(mean=[0, mu_xi], cov=cov, size=J)
        logA, xi = fundamentals[:, 0], fundamentals[:, 1]
        z1 = np.zeros(J)
        z2 = np.zeros(J)

    A = np.exp(logA)

    df = pd.DataFrame({
        'firm_id': np.arange(1, J + 1),
        'logA': logA,
        'A': A,
        'xi': xi,
        'z1': z1,
        'z2': z2,
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


def gh_polar_2d(
    center: np.ndarray,
    r_mu: float,
    r_sigma: float,
    n_r: int,
    n_theta: int,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build polar quadrature with r ~ N(r_mu, r_sigma^2), theta uniform."""
    if n_r < 1 or n_theta < 1:
        raise ValueError("Polar quadrature requires n_r and n_theta >= 1.")
    if r_sigma <= 0:
        raise ValueError("worker_r_sigma must be positive for polar locations.")

    z_r, w_r = gh_nodes_weights_std_normal(n_r)
    r_vals = r_mu + r_sigma * z_r
    mask = r_vals > 0.0
    if not np.any(mask):
        raise ValueError("Polar radii are non-positive; adjust worker_r_mu or worker_r_sigma.")
    r_vals = r_vals[mask]
    w_r = w_r[mask]
    if normalize:
        w_r = w_r / w_r.sum()

    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    w_theta = np.full(n_theta, 1.0 / n_theta)

    R, Theta = np.meshgrid(r_vals, theta_vals, indexing="xy")
    x = center[0] + R * np.cos(Theta)
    y = center[1] + R * np.sin(Theta)

    points = np.column_stack([x.ravel(), y.ravel()])
    weights = (w_theta[:, None] * w_r[None, :]).ravel()
    if normalize:
        weights = weights / weights.sum()
    return points, weights


def generate_support_points(config: ModelConfig, out_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gaussian-Hermite quadrature points from worker distribution parameters.
    
    Args:
        config: Model configuration
        out_path: Output file path
        
    Returns:
        Tuple of (points, weights)
    """
    mu = np.array([config.worker.mu_x, config.worker.mu_y])
    if config.worker.loc_mode not in ("cartesian", "polar"):
        raise ValueError(f"worker_loc_mode must be cartesian or polar; got {config.worker.loc_mode}.")
    if config.worker.loc_mode == "polar":
        points, weights = gh_polar_2d(
            mu,
            config.worker.r_mu,
            config.worker.r_sigma,
            config.quad.n_x,
            config.quad.n_y,
            normalize=config.quad.normalize,
        )
    else:
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
    default_data_dir = get_data_subdir(DATA_RAW, create=True)

    parser = argparse.ArgumentParser(description="Worker Screening Data Preparation")
    
    # Firm generation parameters
    parser.add_argument("--J", type=int, help="Number of firms")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    
    # Worker distribution parameters
    parser.add_argument(
        "--worker_loc_mode",
        type=str,
        choices=["cartesian", "polar"],
        help="Worker location distribution (cartesian or polar)",
    )
    parser.add_argument("--worker_mu_x", type=float, help="Mean x-coordinate of worker distribution")
    parser.add_argument("--worker_mu_y", type=float, help="Mean y-coordinate of worker distribution")
    parser.add_argument("--worker_sigma_x", type=float, help="Standard deviation of worker x-coordinate")
    parser.add_argument("--worker_sigma_y", type=float, help="Standard deviation of worker y-coordinate")
    parser.add_argument("--worker_rho", type=float, help="Correlation between worker x and y coordinates")
    parser.add_argument("--worker_r_mu", type=float, help="Mean radius for polar worker locations")
    parser.add_argument("--worker_r_sigma", type=float, help="Std deviation of radius for polar worker locations")
    
    # Quadrature parameters
    parser.add_argument("--quad_n_x", type=int, help="Number of Gauss-Hermite nodes in x dimension")
    parser.add_argument("--quad_n_y", type=int, help="Number of Gauss-Hermite nodes in y dimension")
    parser.add_argument("--normalize", action="store_true", help="Normalize quadrature weights to sum to 1")
    
    # Core parameters - skill distribution (log-additive: ln q = gamma0 + gamma1*v + e)
    parser.add_argument("--gamma0", type=float, help="Intercept in ln q = gamma0 + gamma1*v + e")
    parser.add_argument("--gamma1", type=float, help="Loading on observable skill v")
    parser.add_argument("--mu_v", type=float, help="Mean of observable component v")
    parser.add_argument("--sigma_v", type=float, help="Std dev of observable component v")
    parser.add_argument("--sigma_e", type=float, help="Std dev of unobservable component e")
    parser.add_argument("--rho_x_skill_ell_x", type=float, help="Correlation between v and worker x-location")
    parser.add_argument("--rho_x_skill_ell_y", type=float, help="Correlation between v and worker y-location")
    parser.add_argument("--rho_x_skill_r", type=float, help="Correlation between v and distance from worker mean location")
    parser.add_argument("--eta", type=float, help="Wage elasticity parameter")
    parser.add_argument("--alpha", type=float, help="Production function parameter")
    parser.add_argument("--tau", type=float, help="Distance decay / commuting cost parameter")
    parser.add_argument("--conduct_mode", type=int, choices=[0, 1, 2], help="0=monopsonistic (model elasticities), 1=status quo, 2=behavioral elasticities ~ N(1, 0.25)")
    parser.add_argument("--N_workers", type=float, help="Total number of workers in the market")
    
    # Firm fundamentals parameters
    parser.add_argument("--sigma_A", type=float, help="Standard deviation of log TFP")
    parser.add_argument("--sigma_xi", type=float, help="Standard deviation of amenity shock")
    parser.add_argument("--mu_xi", type=float, help="Mean of amenity shock")
    parser.add_argument("--rho_Axi", type=float, help="Correlation between log TFP and amenity shock")
    parser.add_argument("--sigma_z1", type=float, help="Std dev of observed TFP shifter (instrument for A)")
    parser.add_argument("--sigma_z2", type=float, help="Std dev of observed amenity shifter (instrument for xi)")
    parser.add_argument(
        "--firms_input_path",
        type=str,
        help="Optional CSV path to specify firms manually with columns firm_id,A,xi,x,y (optional: comp, logA). Overrides random generation.",
    )
    
    # Multi-market
    parser.add_argument("--M", type=int, default=1,
                        help="Number of independent markets to generate (default: 1)")

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
    for param in ['J', 'gamma0', 'gamma1', 'mu_v', 'sigma_v', 'sigma_e',
                 'rho_x_skill_ell_x', 'rho_x_skill_ell_y', 'rho_x_skill_r',
                 'eta', 'alpha', 'tau', 'conduct_mode', 'N_workers',
                 'worker_loc_mode', 'worker_mu_x', 'worker_mu_y', 'worker_sigma_x', 'worker_sigma_y',
                 'worker_rho', 'worker_r_mu', 'worker_r_sigma',
                 'quad_n_x', 'quad_n_y', 'sigma_A', 'sigma_xi', 'rho_Axi', 'mu_xi',
                 'sigma_z1', 'sigma_z2']:
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
    
    input_firms_df: Optional[pd.DataFrame] = None
    if args.firms_input_path:
        input_path = Path(args.firms_input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"firms_input_path does not exist: {input_path}")
        input_firms_df = pd.read_csv(input_path)
        required_cols = {"firm_id", "A", "xi", "x", "y"}
        missing_cols = required_cols - set(input_firms_df.columns)
        if missing_cols:
            raise ValueError(f"Input firms CSV missing required columns: {missing_cols}")
        if "logA" not in input_firms_df.columns:
            input_firms_df["logA"] = np.log(np.maximum(input_firms_df["A"].to_numpy(dtype=float), 1e-300))
        if "comp" not in input_firms_df.columns:
            input_firms_df["comp"] = 1
        if "z1" not in input_firms_df.columns:
            input_firms_df["z1"] = 0.0
        if "z2" not in input_firms_df.columns:
            input_firms_df["z2"] = 0.0
        input_firms_df = input_firms_df[["firm_id", "logA", "A", "xi", "z1", "z2", "comp", "x", "y"]].copy()
        config.firm.J = int(len(input_firms_df))
        print(f"\nUsing manually specified firms from {input_path} (J={config.firm.J})")

    # Print configuration summary (after applying any manual firm overrides)
    print("\n=== CONFIGURATION SUMMARY ===")
    print(f"[CONFIG] core:  gamma0={config.core.gamma0}, gamma1={config.core.gamma1}, mu_v={config.core.mu_v}, sigma_v={config.core.sigma_v}, sigma_e={config.core.sigma_e}")
    print(f"[CONFIG] core:  mu_lnq={config.core.mu_lnq:.4f}, sigma_lnq={config.core.sigma_lnq:.4f}, eta={config.core.eta}, alpha={config.core.alpha}, tau={config.core.tau}")
    print(f"[CONFIG] skill-location corr: rho_x_skill_ell_x={config.core.rho_x_skill_ell_x}, rho_x_skill_ell_y={config.core.rho_x_skill_ell_y}")
    print(
        "[CONFIG] worker: mode={}, mu=({}, {}), sigma=({}, {}), rho={}, r_mu={}, r_sigma={}".format(
            config.worker.loc_mode,
            config.worker.mu_x,
            config.worker.mu_y,
            config.worker.sigma_x,
            config.worker.sigma_y,
            config.worker.rho,
            config.worker.r_mu,
            config.worker.r_sigma,
        )
    )
    print(f"[CONFIG] quad:   kind={config.quad.kind}, n_x={config.quad.n_x}, n_y={config.quad.n_y}, normalize={config.quad.normalize}")
    print(f"[CONFIG] firm:   J={config.firm.J}, sigma_A={config.firm.sigma_A}, sigma_xi={config.firm.sigma_xi}, rho_Axi={config.firm.rho_Axi}, mu_xi={config.firm.mu_xi}, sigma_z1={config.firm.sigma_z1}, sigma_z2={config.firm.sigma_z2}")

    if input_firms_df is not None:
        mask_J = effective_table["parameter"] == "J"
        effective_table.loc[mask_J, "value"] = config.firm.J
        effective_table.loc[mask_J, "source"] = "MANUAL"

    # Add M (number of markets) to effective table
    effective_table = pd.concat([
        effective_table,
        pd.DataFrame([{'parameter': 'M', 'value': args.M, 'source': 'CLI' if args.M > 1 else 'DEFAULT'}]),
    ], ignore_index=True)

    # Dump effective configuration (after any manual firm overrides)
    dump_effective_config_csv(effective_table, out_dir / "parameters_effective.csv")

    M = args.M

    def _generate_firms_df(rng_market):
        """Generate a single market's firms DataFrame."""
        cov = build_covariance(config.firm.sigma_A, config.firm.sigma_xi, config.firm.rho_Axi)
        fundamentals_df = draw_firm_fundamentals(
            config.firm.J, cov, rng_market, mu_xi=config.firm.mu_xi,
            sigma_z1=config.firm.sigma_z1, sigma_z2=config.firm.sigma_z2)
        locations_df = draw_firm_locations(config.firm.J, config.firm.centers, config.firm.sds, config.firm.weights, rng_market)
        df = pd.merge(fundamentals_df, locations_df, on='firm_id')
        if config.core.conduct_mode == 2:
            df['eps_L_behavioral'] = rng_market.normal(3, 1, config.firm.J)
            df['eps_S_behavioral'] = rng_market.normal(1, 2, config.firm.J)
        return df.sort_values('firm_id').reset_index(drop=True)

    # Generate firm data
    market_dfs = []
    if input_firms_df is not None:
        if M > 1:
            raise ValueError("--M > 1 is not supported with --firms_input_path. "
                             "Remove --firms_input_path to generate multiple markets.")
        market_dfs.append(input_firms_df.copy().sort_values('firm_id').reset_index(drop=True))
    else:
        print(f"\nGenerating {config.firm.J} firms x {M} market(s)...")
        for m in range(M):
            rng_m = np.random.default_rng(args.seed + m)
            market_dfs.append(_generate_firms_df(rng_m))

    # Write per-market files if M > 1
    if M > 1:
        markets_dir = out_dir / "markets"
        markets_dir.mkdir(parents=True, exist_ok=True)
        for m, df_m in enumerate(market_dfs, start=1):
            market_path = markets_dir / f"firms_market_{m}.csv"
            df_m.to_csv(market_path, index=False)
            print(f"  Market {m}: {len(df_m)} firms -> {market_path}")

    # Build combined firms DataFrame (with market_id column when M > 1)
    if M == 1:
        firms_df = market_dfs[0]
    else:
        parts = []
        for m, df_m in enumerate(market_dfs, start=1):
            df_copy = df_m.copy()
            df_copy.insert(0, 'market_id', m)
            parts.append(df_copy)
        firms_df = pd.concat(parts, ignore_index=True)

    # Write combined firms CSV
    firms_path = out_dir / "firms.csv"
    firms_df.to_csv(firms_path, index=False)
    print(f"Firms data written to: {firms_path}")

    # Print firm summary statistics
    print(f"Firm summary ({M} market(s), {len(firms_df)} total firms):")
    print(f"  A: mean={firms_df['A'].mean():.4f}, std={firms_df['A'].std():.4f}")
    print(f"  xi: mean={firms_df['xi'].mean():.4f}, std={firms_df['xi'].std():.4f}")
    print(f"  Spatial bounds: x=[{firms_df['x'].min():.2f}, {firms_df['x'].max():.2f}], y=[{firms_df['y'].min():.2f}, {firms_df['y'].max():.2f}]")

    # Generate support points (shared across markets)
    print(f"\nGenerating worker quadrature points...")
    support_path = out_dir / "support_points.csv"
    points, weights = generate_support_points(config, support_path)
    print(f"Support points written to: {support_path}")

    # Write parameters CSV (shared across markets, includes M)
    params_path = out_dir / "parameters.csv"
    write_parameters_template(params_path)
    print(f"Parameters template written to: {params_path}")

    print(f"\n=== OUTPUT FILES ===")
    print(f"  {firms_path}")
    if M > 1:
        print(f"  {out_dir / 'markets/'} ({M} per-market firm files)")
    print(f"  {support_path}")
    print(f"  {params_path}")
    print(f"  {out_dir / 'parameters_effective.csv'}")


if __name__ == "__main__":
    main()
