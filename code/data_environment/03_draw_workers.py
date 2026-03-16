#!/usr/bin/env python3
"""
Worker Dataset Construction

This script constructs a worker dataset by drawing worker characteristics from continuous distributions.
It reads effective parameters to generate workers with skill, location, and auxiliary skill components.
"""

import argparse
import colorsys
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import norm

if __package__ is None or __package__ == "":
    import sys

    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
    from code import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_WORKERS  # type: ignore
    from code.estimation.helpers import (  # type: ignore
        compute_order_maps,
        read_params_long_csv,
    )
else:  # pragma: no cover - executed when running as package module
    from .. import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_WORKERS
    from ..estimation.helpers import compute_order_maps, read_params_long_csv


DEFAULT_RAW_DIR = get_data_subdir(DATA_RAW)
DEFAULT_CLEAN_DIR = get_data_subdir(DATA_CLEAN)
DEFAULT_BUILD_DIR = get_data_subdir(DATA_BUILD)
DEFAULT_PLOT_DIR = get_output_subdir(OUTPUT_WORKERS)


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def suffix_sum(arr: np.ndarray) -> np.ndarray:
    """Compute suffix sums: result[i] = sum(arr[i:])"""
    return np.cumsum(arr[::-1])[::-1]


def _categorical_palette(n: int) -> list[tuple[float, float, float]]:
    if n <= 10:
        return list(plt.get_cmap("tab10").colors)[:n]
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    return [colorsys.hls_to_rgb(float(h), 0.5, 0.65) for h in hues]




def simulate_worker_choice_probabilities(
    worker_idx: int,
    df: pd.DataFrame,
    firms_data: Dict[str, Any],
    params: Dict[str, Any],
    n_simulations: int = 10000,
    seed: int = 123,
    out_dir: Path | str = DEFAULT_BUILD_DIR,
) -> Dict[str, Any]:
    """
    Simulate choice probabilities for a single worker using Type 1 extreme value shocks.
    Print detailed simulation data for manual verification.
    
    Returns:
        Dictionary with simulation results
    """
    print(f"\n=== WORKER {worker_idx} SIMULATION - DETAILED OUTPUT ===")
    
    # Extract worker characteristics
    x_skill = df['x_skill'].values[worker_idx]
    ell_x = df['ell_x'].values[worker_idx]
    ell_y = df['ell_y'].values[worker_idx]
    
    # Extract firm data
    w = firms_data['w']
    qbar = firms_data['qbar']
    xi = firms_data['xi']
    A = firms_data['A']
    loc_firms = firms_data['loc_firms']
    firm_ids = firms_data['firm_ids']
    J = len(w)
    
    # Extract parameters
    eta = float(params['eta'])
    alpha = float(params['alpha'])
    tau = float(params.get('tau', 0.1))
    sigma_e = float(params.get('sigma_e', 1.5))

    print(f"Worker characteristics:")
    print(f"  x_skill: {x_skill:.6f}")
    print(f"  location: ({ell_x:.6f}, {ell_y:.6f})")
    print(f"Parameters:")
    print(f"  eta: {eta}")
    print(f"  tau: {tau}")
    print(f"  sigma_e: {sigma_e}")
    
    # SIMULATION using Type 1 extreme value shocks
    rng = np.random.default_rng(seed)
    
    # Distances to firms
    distances = np.sqrt((ell_x - loc_firms[:, 0])**2 + (ell_y - loc_firms[:, 1])**2)
    
    # Systematic utilities: U_ij = eta * log(w_j) + xi_j - tau * distance_ij
    U_systematic = eta * np.log(w) + xi - tau * distances  # (J,)
    U_outside_systematic = 0.0  # Outside option normalized to 0
    
    print(f"\nFirm data:")
    for j in range(J):
        print(f"  Firm {firm_ids[j]}: w={w[j]:.6f}, qbar={qbar[j]:.6f}, xi={xi[j]:.6f}, distance={distances[j]:.6f}, U_sys={U_systematic[j]:.6f}")
    
    print(f"\nRunning {n_simulations} simulations, saving to CSV...")
    
    # Prepare lists to store simulation data
    sim_data = []
    choices = np.zeros(n_simulations, dtype=int)  # 0 = outside, 1,...,J = firms
    
    for sim in range(n_simulations):
        # Draw a_skill for this simulation
        a_skill_sim = rng.normal(0, sigma_e)  # a_skill ~ N(0, σ_e²)
        s_skill_sim = x_skill + a_skill_sim   # Total skill for this simulation
        
        # Check which firms are feasible for this skill draw
        feasible_firms_sim = s_skill_sim > qbar  # (J,) boolean array
        
        # Draw Type 1 extreme value errors for all choices
        eps_outside = rng.gumbel(0, 1)  # Outside option error
        eps_firms = rng.gumbel(0, 1, size=J)  # Firm errors
        
        # Total utilities including random shocks
        U_outside_total = U_outside_systematic + eps_outside
        U_firms_total = U_systematic + eps_firms
        
        # Apply feasibility constraints for this skill draw
        U_firms_feasible = np.where(feasible_firms_sim, U_firms_total, -np.inf)
        
        # Find choice with maximum utility
        all_utilities = np.concatenate([[U_outside_total], U_firms_feasible])
        best_choice = np.argmax(all_utilities)
        
        choices[sim] = best_choice  # 0 = outside, 1,...,J = firms
        
        # Store detailed info for this simulation
        row_data = {
            'sim': sim,
            'a_skill': a_skill_sim,
            's_skill': s_skill_sim,
            'eps_outside': eps_outside,
            'U_outside_total': U_outside_total,
            'choice': best_choice
        }
        
        # Add feasible flags for each firm
        for j in range(J):
            row_data[f'feasible_firm_{firm_ids[j]}'] = feasible_firms_sim[j]
        
        # Add epsilon values for each firm
        for j in range(J):
            row_data[f'eps_firm_{firm_ids[j]}'] = eps_firms[j]
        
        # Add total utilities for each firm
        for j in range(J):
            row_data[f'U_firm_{firm_ids[j]}'] = U_firms_feasible[j]
        
        sim_data.append(row_data)
        
        # Print progress every 1000 simulations
        if (sim + 1) % 1000 == 0:
            print(f"  Completed {sim + 1:,} simulations")
    
    # Save detailed simulation data to CSV
    sim_df = pd.DataFrame(sim_data)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"worker_{worker_idx}_detailed_sims.csv"
    sim_df.to_csv(csv_path, index=False)
    print(f"Detailed simulation data saved to: {csv_path}")
    
    # 3. COMPUTE SIMULATED PROBABILITIES
    choice_counts = np.bincount(choices, minlength=J+1)
    P_simulated = choice_counts / n_simulations
    
    P_outside_simulated = P_simulated[0]
    P_firms_simulated = P_simulated[1:]  # Natural firm order
    
    print(f"\n=== SIMULATION RESULTS ===")
    print(f"Choice counts: {choice_counts}")
    print(f"Total simulations: {n_simulations}")
    print(f"\nSimulated probabilities:")
    print(f"  P_outside: {P_outside_simulated:.6f} ({choice_counts[0]} choices)")
    for j in range(J):
        firm_id = firm_ids[j]
        print(f"  P_firm_{firm_id}: {P_firms_simulated[j]:.6f} ({choice_counts[j+1]} choices)")
    print(f"  Total probability: {P_simulated.sum():.6f}")
    
    return {
        'worker_idx': worker_idx,
        'worker_characteristics': {
            'x_skill': float(x_skill),
            'location': [float(ell_x), float(ell_y)],
            'sigma_e': float(sigma_e)
        },
        'simulation_setup': {
            'qbar_firms': qbar.tolist(),
            'systematic_utilities': U_systematic.tolist(),
            'distances': distances.tolist(),
            'n_simulations': int(n_simulations)
        },
        'simulated': {
            'P_outside': float(P_outside_simulated),
            'P_firms': P_firms_simulated.tolist(),
            'choice_counts': choice_counts.tolist(),
            'n_simulations': int(n_simulations)
        },
        'firm_mapping': {
            'firm_ids': firm_ids.tolist()
        }
    }


# =============================================================================
# MAIN WORKER CONSTRUCTION FUNCTIONS
# =============================================================================

def construct_worker_dataset(
    params: Dict[str, Any],
    firms_data: Dict[str, Any] = None,
    seed: int = 123
) -> pd.DataFrame:
    """
    Construct worker dataset by drawing from continuous distributions.
    
    Args:
        params: Parameters dictionary with worker distribution parameters
        firms_data: Dictionary with firm data (w, c, xi, locations) or None
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with worker data drawn from continuous distributions
    """
    # Extract parameters - use new skill distribution parameterization
    N_workers = int(float(params['N_workers']))
    
    # Worker location distribution parameters
    worker_mu_x = float(params['worker_mu_x'])
    worker_mu_y = float(params['worker_mu_y'])
    worker_sigma_x = float(params['worker_sigma_x'])
    worker_sigma_y = float(params['worker_sigma_y'])
    worker_rho = float(params['worker_rho'])
    worker_loc_mode = str(params.get('worker_loc_mode', 'cartesian'))
    worker_r_mu = float(params.get('worker_r_mu', 0.0))
    worker_r_sigma = float(params.get('worker_r_sigma', 1.0))
    
    # New skill distribution component parameters
    mu_x_skill = float(params.get('mu_x_skill', params.get('mu_s', 10.009)))  # Fallback for backward compatibility
    sigma_x_skill = float(params.get('sigma_x_skill', 2.6))
    mu_e = float(params.get('mu_e', 0.0))
    sigma_e = float(params.get('sigma_e', 1.5))
    rho_x_skill_ell_x = float(params.get('rho_x_skill_ell_x', 0.0))
    rho_x_skill_ell_y = float(params.get('rho_x_skill_ell_y', 0.0))
    rho_x_skill_r = float(params.get('rho_x_skill_r', 0.0))

    if worker_loc_mode not in ("cartesian", "polar"):
        raise ValueError(f"worker_loc_mode must be cartesian or polar; got {worker_loc_mode}.")
    
    # Derived parameters (for validation)
    mu_s_computed = mu_x_skill + mu_e
    sigma_s_computed = np.sqrt(sigma_x_skill**2 + sigma_e**2)
    
    # Set up random number generator
    rng = np.random.default_rng(seed)
    
    print(f"Drawing {N_workers} workers from continuous distributions...")
    print(f"  x_skill ~ N({mu_x_skill:.3f}, {sigma_x_skill:.3f}²)")
    print(f"  a_skill ~ N({mu_e:.1f}, {sigma_e:.3f}²)")
    print(f"  s_skill = x_skill + a_skill  =>  s_skill ~ N({mu_s_computed:.3f}, {sigma_s_computed:.3f}²)")
    if worker_loc_mode == "polar":
        print(
            "  locations ~ PolarN(center=({:.1f}, {:.1f}), r~N({:.1f}, {:.1f}^2))".format(
                worker_mu_x, worker_mu_y, worker_r_mu, worker_r_sigma
            )
        )
    else:
        print(
            "  locations ~ BivarN([{:.1f}, {:.1f}], sigma=({:.1f}, {:.1f}), rho={:.1f})".format(
                worker_mu_x, worker_mu_y, worker_sigma_x, worker_sigma_y, worker_rho
            )
        )
    print(
        "  corr(x_skill, ell_x)={:.3f}, corr(x_skill, ell_y)={:.3f}, corr(x_skill, r)={:.3f}".format(
            rho_x_skill_ell_x, rho_x_skill_ell_y, rho_x_skill_r
        )
    )
    
    # Draw continuous samples
    if not (-1.0 <= rho_x_skill_ell_x <= 1.0 and -1.0 <= rho_x_skill_ell_y <= 1.0):
        raise ValueError("rho_x_skill_ell_x and rho_x_skill_ell_y must be in [-1, 1].")
    if not (-1.0 <= rho_x_skill_r <= 1.0):
        raise ValueError("rho_x_skill_r must be in [-1, 1].")
    if abs(rho_x_skill_r) > 0.0 and (abs(rho_x_skill_ell_x) > 0.0 or abs(rho_x_skill_ell_y) > 0.0):
        raise ValueError("Set either rho_x_skill_r or rho_x_skill_ell_x/ell_y, not both.")
    if worker_loc_mode == "polar" and (abs(rho_x_skill_ell_x) > 0.0 or abs(rho_x_skill_ell_y) > 0.0):
        raise ValueError("Polar locations do not support ell_x/ell_y correlations; use rho_x_skill_r instead.")

    def _draw_locations() -> tuple[np.ndarray, np.ndarray]:
        if worker_loc_mode == "polar":
            if worker_r_sigma <= 0:
                raise ValueError("worker_r_sigma must be positive for polar locations.")
            r_vals = rng.normal(worker_r_mu, worker_r_sigma, N_workers)
            for _ in range(10):
                mask = r_vals <= 0
                if not np.any(mask):
                    break
                r_vals[mask] = rng.normal(worker_r_mu, worker_r_sigma, mask.sum())
            r_vals = np.maximum(r_vals, 0.0)
            theta_vals = rng.uniform(0.0, 2.0 * np.pi, N_workers)
            x_vals = worker_mu_x + r_vals * np.cos(theta_vals)
            y_vals = worker_mu_y + r_vals * np.sin(theta_vals)
            return x_vals, y_vals
        cov_matrix = np.array([
            [worker_sigma_x**2, worker_rho * worker_sigma_x * worker_sigma_y],
            [worker_rho * worker_sigma_x * worker_sigma_y, worker_sigma_y**2]
        ])
        mean_vector = np.array([worker_mu_x, worker_mu_y])
        locations = rng.multivariate_normal(mean_vector, cov_matrix, N_workers)
        return locations[:, 0], locations[:, 1]

    if abs(rho_x_skill_r) > 0.0:
        ell_x, ell_y = _draw_locations()
        r_vals = np.sqrt((ell_x - worker_mu_x) ** 2 + (ell_y - worker_mu_y) ** 2)
        r_std = r_vals.std()
        if r_std <= 1e-12:
            raise ValueError("Worker radius has near-zero variance; cannot apply rho_x_skill_r.")
        r_std_vals = (r_vals - r_vals.mean()) / r_std
        z = rng.normal(0.0, 1.0, N_workers)
        x_skill = mu_x_skill + sigma_x_skill * (
            rho_x_skill_r * r_std_vals + np.sqrt(1.0 - rho_x_skill_r**2) * z
        )
    elif abs(rho_x_skill_ell_x) > 0.0 or abs(rho_x_skill_ell_y) > 0.0:
        joint_mean = np.array([mu_x_skill, worker_mu_x, worker_mu_y])
        joint_cov = np.array(
            [
                [sigma_x_skill**2,
                 rho_x_skill_ell_x * sigma_x_skill * worker_sigma_x,
                 rho_x_skill_ell_y * sigma_x_skill * worker_sigma_y],
                [rho_x_skill_ell_x * sigma_x_skill * worker_sigma_x,
                 worker_sigma_x**2,
                 worker_rho * worker_sigma_x * worker_sigma_y],
                [rho_x_skill_ell_y * sigma_x_skill * worker_sigma_y,
                 worker_rho * worker_sigma_x * worker_sigma_y,
                 worker_sigma_y**2],
            ]
        )
        try:
            np.linalg.cholesky(joint_cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Invalid correlation structure for (x_skill, ell_x, ell_y); "
                "check rho_x_skill_ell_x, rho_x_skill_ell_y, and worker_rho."
            ) from exc
        draws = rng.multivariate_normal(joint_mean, joint_cov, N_workers)
        x_skill = draws[:, 0]
        ell_x = draws[:, 1]
        ell_y = draws[:, 2]
    else:
        x_skill = rng.normal(mu_x_skill, sigma_x_skill, N_workers)
        ell_x, ell_y = _draw_locations()
    
    # 3) Draw a_skill from N(mu_e, sigma_e²)
    a_skill = rng.normal(mu_e, sigma_e, N_workers)
    
    # 4) Compute total skill s_i = x_i + a_i
    s_skill = x_skill + a_skill
    
    # Create DataFrame with basic worker characteristics
    df = pd.DataFrame({
        'worker_id': np.arange(N_workers),
        'x_skill': x_skill,
        'ell_x': ell_x,
        'ell_y': ell_y,
        'a_skill': a_skill,
        's_skill': s_skill
    })
    
    # 5) Compute utilities and firm choice if firm data is provided
    if firms_data is not None:
        eta = float(params.get('eta', 5.0))
        tau = float(params.get('tau', 0.05))

        w_firms = firms_data['w']  # wages
        qbar_firms = firms_data['qbar']  # cutoffs
        xi_firms = firms_data['xi']  # firm fixed effects
        loc_firms = firms_data['locations']  # firm locations (J x 2)
        J = len(w_firms)
        
        print(f"Computing utilities and firm choices for {J} firms...")
        
        # Draw Type I extreme value errors for all worker-firm pairs
        # u_i0 = e_i0 (outside option, only error term)
        # u_ij = eta * log(w_j) + xi_j - tau * ||ell_i - ell_j|| + e_ij
        
        # Draw extreme value errors (Gumbel distribution with location=0, scale=1)
        # For each worker: one error for outside option + J errors for firms
        errors = rng.gumbel(0, 1, size=(N_workers, J + 1))  # Shape: (N_workers, J+1)
        
        # Compute utilities
        utilities = np.zeros((N_workers, J + 1))  # Column 0 = outside, 1..J = firms
        
        # Outside option utility: u_i0 = e_i0
        utilities[:, 0] = errors[:, 0]
        
        # Firm utilities: u_ij = eta * log(w_j) + xi_j - tau * ||ell_i - ell_j|| + e_ij
        for j in range(J):
            # Distance between worker i and firm j
            worker_locations = np.column_stack([ell_x, ell_y])  # (N_workers, 2)
            firm_location = loc_firms[j]  # (2,)
            distances = np.linalg.norm(worker_locations - firm_location, axis=1)  # (N_workers,)
            
            # Utility components
            log_wage = np.log(w_firms[j])
            xi_j = xi_firms[j]
            
            # Total utility for firm j
            utilities[:, j + 1] = eta * log_wage + xi_j - tau * distances + errors[:, j + 1]
        
        # Add utility columns to dataframe
        utility_cols = {}
        utility_cols['u_i0'] = utilities[:, 0]  # Outside option
        for j in range(J):
            utility_cols[f'u_i{j+1}'] = utilities[:, j + 1]  # Firm j+1
        
        df = pd.concat([df, pd.DataFrame(utility_cols)], axis=1)
        
        # 6) Determine chosen firm based on skill constraints and utility maximization
        chosen_firm = np.zeros(N_workers, dtype=int)  # Default to outside option (0)
        
        for i in range(N_workers):
            s_i = s_skill[i]
            
            # Find feasible firms where s_i > qbar_j
            feasible_firms = []
            feasible_utilities = []
            3
            # Outside option is always feasible
            feasible_firms.append(0)
            feasible_utilities.append(utilities[i, 0])
            
            # Check firms where skill constraint is satisfied
            for j in range(J):
                if s_i > qbar_firms[j]:
                    feasible_firms.append(j + 1)  # Firm indices 1, 2, ..., J
                    feasible_utilities.append(utilities[i, j + 1])
            
            # Choose firm with highest utility among feasible options
            if feasible_utilities:
                best_idx = np.argmax(feasible_utilities)
                chosen_firm[i] = feasible_firms[best_idx]
        
        df['chosen_firm'] = chosen_firm
        
        print(f"  Firm choice distribution: {np.bincount(chosen_firm, minlength=J+1)}")
    else:
        print("No firm data provided; skipping utility computation and firm choice.")
    
    # Validation checks
    print(f"\n=== VALIDATION CHECKS ===")
    print(f"N_workers: {N_workers}")
    print(f"μ_s: {mu_s_computed:.3f}, σ_s: {sigma_s_computed:.3f}")
    print(f"σ_x: {sigma_x_skill:.3f}, σ_e: {sigma_e:.3f}")
    print(f"μ_e: {mu_e:.3f}")
    print(
        "Location params: mode={}, mu=({:.1f}, {:.1f}), sigma=({:.1f}, {:.1f}), rho={:.1f}, r_mu={:.1f}, r_sigma={:.1f}".format(
            worker_loc_mode,
            worker_mu_x,
            worker_mu_y,
            worker_sigma_x,
            worker_sigma_y,
            worker_rho,
            worker_r_mu,
            worker_r_sigma,
        )
    )
    
    # Basic statistics
    print(f"x_skill: mean={x_skill.mean():.3f}, std={x_skill.std():.3f}")
    print(f"ell_x: mean={ell_x.mean():.3f}, std={ell_x.std():.3f}")
    print(f"ell_y: mean={ell_y.mean():.3f}, std={ell_y.std():.3f}")
    print(f"a_skill: mean={a_skill.mean():.3f}, std={a_skill.std():.3f}")
    print(f"s_skill: mean={s_skill.mean():.3f}, std={s_skill.std():.3f}")
    
    # Compute empirical correlation for validation
    empirical_corr = np.corrcoef(ell_x, ell_y)[0, 1]
    corr_x_skill_ell_x = np.corrcoef(x_skill, ell_x)[0, 1]
    corr_x_skill_ell_y = np.corrcoef(x_skill, ell_y)[0, 1]
    r_vals = np.sqrt((ell_x - worker_mu_x) ** 2 + (ell_y - worker_mu_y) ** 2)
    if r_vals.std() > 0:
        corr_x_skill_r = np.corrcoef(x_skill, r_vals)[0, 1]
    else:
        corr_x_skill_r = np.nan
    print(f"Location correlation: empirical={empirical_corr:.3f}, expected={worker_rho:.1f}")
    print(
        "x_skill-location correlation: empirical=({:.3f}, {:.3f}), target=({:.3f}, {:.3f})".format(
            corr_x_skill_ell_x,
            corr_x_skill_ell_y,
            rho_x_skill_ell_x,
            rho_x_skill_ell_y,
        )
    )
    print(
        "x_skill-radius correlation: empirical={:.3f}, target={:.3f}".format(
            corr_x_skill_r,
            rho_x_skill_r,
        )
    )
    
    # Preview first 5 rows
    print(f"\n=== WORKER PREVIEW (first 5 rows) ===")
    preview_cols = ['worker_id', 'x_skill', 'ell_x', 'ell_y', 'a_skill', 's_skill']
    if 'chosen_firm' in df.columns:
        preview_cols.append('chosen_firm')
    print(df[preview_cols].head().to_string(index=False))
    
    # Store summary statistics in DataFrame attributes for later use
    df.attrs['summary'] = {
        'N_workers': N_workers,
        'mu_s': mu_s_computed,
        'sigma_s': sigma_s_computed,
        'mu_x_skill': mu_x_skill,
        'sigma_x_skill': sigma_x_skill,
        'mu_e': mu_e,
        'sigma_e': sigma_e,
        'worker_mu_x': worker_mu_x,
        'worker_mu_y': worker_mu_y,
        'worker_sigma_x': worker_sigma_x,
        'worker_sigma_y': worker_sigma_y,
        'worker_rho': worker_rho,
        'worker_loc_mode': worker_loc_mode,
        'worker_r_mu': worker_r_mu,
        'worker_r_sigma': worker_r_sigma,
        'rho_x_skill_ell_x': rho_x_skill_ell_x,
        'rho_x_skill_ell_y': rho_x_skill_ell_y,
        'rho_x_skill_r': rho_x_skill_r,
        'x_skill_mean': x_skill.mean(),
        'x_skill_std': x_skill.std(),
        'ell_x_mean': ell_x.mean(),
        'ell_x_std': ell_x.std(),
        'ell_y_mean': ell_y.mean(),
        'ell_y_std': ell_y.std(),
        'location_corr_empirical': empirical_corr,
        'x_skill_ell_x_corr_empirical': corr_x_skill_ell_x,
        'x_skill_ell_y_corr_empirical': corr_x_skill_ell_y,
        'x_skill_r_corr_empirical': corr_x_skill_r,
        'a_skill_mean': a_skill.mean(),
        'a_skill_std': a_skill.std(),
        's_skill_mean': s_skill.mean(),
        's_skill_std': s_skill.std()
    }
    
    return df


def compute_choices_no_cutoff(
    worker_df: pd.DataFrame, firms_df: pd.DataFrame, params: Dict[str, Any], seed: int
) -> np.ndarray:
    """
    Compute worker firm choices when all firms accept all workers (no cutoffs).
    Returns array of chosen firm_ids (natural IDs); 0 denotes outside option.
    """
    eta = float(params.get("eta", 5.0))
    tau = float(params.get("tau", 0.05))
    wages_col = "w_noscreening" if "w_noscreening" in firms_df.columns else "w"

    wages = np.asarray(firms_df[wages_col].to_numpy(), dtype=float)
    xi_vals = np.asarray(firms_df["xi"].to_numpy(), dtype=float)
    loc_firms = firms_df[["x", "y"]].to_numpy(dtype=float)
    firm_ids = np.asarray(firms_df["firm_id"].to_numpy(), dtype=int)
    J = wages.size

    worker_locations = worker_df[["ell_x", "ell_y"]].to_numpy(dtype=float)
    N = worker_locations.shape[0]

    rng = np.random.default_rng(seed)
    errors = rng.gumbel(0, 1, size=(N, J + 1))  # outside + J firms

    utilities = np.zeros((N, J + 1))
    utilities[:, 0] = errors[:, 0]  # outside option

    for j in range(J):
        distances = np.linalg.norm(worker_locations - loc_firms[j], axis=1)
        utilities[:, j + 1] = (
            eta * np.log(np.maximum(wages[j], 1e-300))
            + xi_vals[j]
            - tau * distances
            + errors[:, j + 1]
        )

    chosen_idx = np.argmax(utilities, axis=1)
    chosen_firm_ids = np.where(chosen_idx == 0, 0, firm_ids[chosen_idx - 1])
    return chosen_firm_ids


# =============================================================================
# Multi-market worker draws
# =============================================================================


def _read_firms_data_from_csv(firms_path: Path):
    """Read firms CSV and return (firms_data dict, firms_df) or (None, None)."""
    if not firms_path.exists():
        return None, None
    try:
        df = pd.read_csv(firms_path)
        J = len(df)
        firm_ids = df['firm_id'].values if 'firm_id' in df.columns else np.arange(1, J + 1)
        data = {
            'w': df['w'].values,
            'qbar': df['qbar'].values,
            'xi': df['xi'].values,
            'locations': df[['x', 'y']].values,
            'firm_ids': firm_ids,
        }
        return data, df
    except Exception as e:
        print(f"Warning: Could not read firms data from {firms_path}: {e}")
        return None, None


def _drop_and_reassign_firms(worker_df: pd.DataFrame, firms_df: pd.DataFrame,
                             drop_below_n: int) -> tuple:
    """Iteratively drop firms with fewer than drop_below_n workers and reassign.

    Workers whose chosen firm is dropped are reassigned to their next best
    feasible choice (highest utility among remaining feasible firms).  The
    process repeats until no more firms fall below the threshold.

    Args:
        worker_df: Worker dataset with columns u_i0..u_iJ, s_skill, chosen_firm.
        firms_df: Firms DataFrame with 'qbar' column (J rows).
        drop_below_n: Minimum number of matched workers a firm must have to survive.

    Returns:
        (worker_df, firms_df) -- modified copies with reindexed firm IDs.
        firms_df is filtered to surviving firms with consecutive firm_id 1..J'.
        worker_df has chosen_firm updated and a chosen_firm_original_id column.
    """
    J = len(firms_df)
    qbar_all = firms_df['qbar'].values
    s_skill = worker_df['s_skill'].values

    # Build utility matrix from stored columns (N x J+1)
    util_cols = [worker_df['u_i0'].values]
    for j in range(J):
        col_name = f'u_i{j + 1}'
        util_cols.append(worker_df[col_name].values)
    util_matrix = np.column_stack(util_cols)

    current_chosen = worker_df['chosen_firm'].values.copy()
    dropped_firms: set = set()

    iteration = 0
    while True:
        iteration += 1
        # Count workers per firm (excluding outside option and dropped firms)
        firm_counts = np.bincount(
            current_chosen[current_chosen > 0], minlength=J + 1
        )[1:]  # index 0..J-1 correspond to firm_id 1..J

        # Find firms below threshold that have not already been dropped
        newly_dropped: set = set()
        for j in range(J):
            firm_id = j + 1
            if firm_id not in dropped_firms and firm_counts[j] < drop_below_n:
                newly_dropped.add(firm_id)

        if not newly_dropped:
            break  # stable

        dropped_firms.update(newly_dropped)
        print(f"  Drop iteration {iteration}: dropping {len(newly_dropped)} firms "
              f"(total dropped: {len(dropped_firms)})")

        # Reassign workers from newly dropped firms
        affected_idx = np.where(np.isin(current_chosen, list(newly_dropped)))[0]
        for i in affected_idx:
            s_i = s_skill[i]
            best_firm = 0  # default to outside option
            best_util = util_matrix[i, 0]
            for j in range(J):
                firm_id = j + 1
                if firm_id in dropped_firms:
                    continue
                if s_i > qbar_all[j] and util_matrix[i, j + 1] > best_util:
                    best_firm = firm_id
                    best_util = util_matrix[i, j + 1]
            current_chosen[i] = best_firm

    worker_df = worker_df.copy()
    worker_df['chosen_firm'] = current_chosen
    print(f"  Final: {len(dropped_firms)} firms dropped, "
          f"{J - len(dropped_firms)} firms remaining")

    # --- Reindex surviving firms to consecutive IDs ---
    chosen_positions = np.sort(
        worker_df.loc[worker_df['chosen_firm'] > 0, 'chosen_firm']
        .astype(int)
        .unique()
    )
    if chosen_positions.size > 0:
        chosen_idx = chosen_positions - 1  # 1-based -> 0-based
        firm_subset = firms_df.iloc[chosen_idx].copy()

        if 'firm_id' in firm_subset.columns:
            firm_subset['original_firm_id'] = firm_subset['firm_id']
        else:
            firm_subset['original_firm_id'] = chosen_positions

        new_ids = np.arange(1, chosen_positions.size + 1, dtype=int)
        mapping = {int(old): int(new) for old, new in zip(chosen_positions, new_ids)}

        worker_df['chosen_firm_original_id'] = worker_df['chosen_firm']
        worker_df['chosen_firm'] = (
            worker_df['chosen_firm']
            .apply(lambda x: mapping.get(int(x), 0))
            .astype(int)
        )

        firm_subset['firm_id'] = new_ids
        firm_subset.reset_index(drop=True, inplace=True)
        firms_df = firm_subset
    else:
        worker_df['chosen_firm_original_id'] = worker_df['chosen_firm']
        firms_df = firms_df.iloc[0:0].copy()  # empty

    return worker_df, firms_df


def _draw_workers_multi_market(args, M, params, raw_dir, clean_dir, build_dir, out_dir, plot_out) -> int:
    """Draw workers for each of M markets and combine."""
    markets_clean = clean_dir / "markets"
    markets_build = out_dir / "markets"
    markets_build.mkdir(parents=True, exist_ok=True)

    print(f"\nMulti-market worker draws: M={M}")
    drop_below_n = getattr(args, 'drop_below_n', None)
    use_drop = drop_below_n is not None and drop_below_n > 0

    # --- Pass 1: draw workers and drop firms per market ---
    market_results = []  # list of (m, worker_df, firms_df)

    for m in range(1, M + 1):
        eq_path = markets_clean / f"equilibrium_firms_market_{m}.csv"
        firms_data, firms_df = _read_firms_data_from_csv(eq_path)
        if firms_data is None:
            print(f"  Market {m}: equilibrium file not found at {eq_path}, skipping.")
            continue

        print(f"  Market {m}: {len(firms_df)} firms, drawing workers (seed={args.seed + m - 1})...")
        worker_df = construct_worker_dataset(params, firms_data=firms_data, seed=args.seed + m - 1)

        if use_drop and 'chosen_firm' in worker_df.columns:
            print(f"    Applying --drop_below_n={drop_below_n} for market {m}...")
            worker_df, firms_df = _drop_and_reassign_firms(worker_df, firms_df, drop_below_n)

        market_results.append((m, worker_df, firms_df))

    if not market_results:
        print("ERROR: No markets produced worker data.")
        return 1

    # --- Pass 2: save per-market files and build combined dataset ---
    all_worker_dfs = []
    all_firm_dfs = []

    for m, worker_df, firms_df in market_results:
        market_path = markets_build / f"workers_dataset_market_{m}.csv"
        worker_df.to_csv(market_path, index=False)

        if use_drop:
            firm_market_path = markets_build / f"firm_data_market_{m}.csv"
            firms_df.to_csv(firm_market_path, index=False)

        worker_df_tagged = worker_df.copy()
        worker_df_tagged.insert(0, 'market_id', m)
        all_worker_dfs.append(worker_df_tagged)

        firms_tagged = firms_df.copy()
        firms_tagged.insert(0, 'market_id', m)
        all_firm_dfs.append(firms_tagged)

        # No-screening variant
        noscreen_path = markets_clean / f"equilibrium_firms_noscreening_market_{m}.csv"
        if noscreen_path.exists():
            try:
                noscreen_df = pd.read_csv(noscreen_path)
                if "firm_id" not in noscreen_df.columns:
                    noscreen_df["firm_id"] = np.arange(1, len(noscreen_df) + 1, dtype=int)
                chosen_ids = compute_choices_no_cutoff(worker_df, noscreen_df, params, seed=args.seed + m)
                worker_noscreen = worker_df_tagged.copy()
                worker_noscreen["chosen_firm"] = chosen_ids
                noscreen_market_path = markets_build / f"workers_dataset_noscreening_market_{m}.csv"
                worker_noscreen.to_csv(noscreen_market_path, index=False)
            except Exception as exc:
                print(f"    Warning: no-screening worker draw failed for market {m}: {exc}")

        print(f"    -> {market_path} ({len(worker_df)} workers, {len(firms_df)} firms)")

    # Combined worker dataset
    combined = pd.concat(all_worker_dfs, ignore_index=True)
    combined_path = out_dir / "workers_dataset.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined worker dataset ({len(combined)} workers) written to: {combined_path}")

    # Combined firm dataset (for estimation code that reads a single firms CSV)
    if all_firm_dfs:
        combined_firms = pd.concat(all_firm_dfs, ignore_index=True)
        firms_combined_path = clean_dir / "equilibrium_firms.csv"
        combined_firms.to_csv(firms_combined_path, index=False)
        j_counts = [len(fdf) for fdf in all_firm_dfs]
        j_min, j_max = min(j_counts), max(j_counts)
        j_str = str(j_min) if j_min == j_max else f"{j_min}-{j_max}"
        print(f"Combined firm dataset ({len(combined_firms)} firms, J_per={j_str}) "
              f"written to: {firms_combined_path}")

    print(f"\n=== OUTPUT FILES ===")
    print(f"  {combined_path}")
    print(f"  {markets_build}/ ({M} per-market worker files)")
    return 0


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(description="Construct worker dataset with continuous sampling")
    raw_dir = get_data_subdir(DATA_RAW, create=True)
    clean_dir = get_data_subdir(DATA_CLEAN, create=True)
    build_dir = get_data_subdir(DATA_BUILD, create=True)
    plot_dir = get_output_subdir(OUTPUT_WORKERS, create=True)

    # File paths
    parser.add_argument(
        "--params_path",
        type=str,
        default=str(raw_dir / "parameters_effective.csv"),
        help="Path to parameters CSV file",
    )
    parser.add_argument(
        "--firms_path",
        type=str,
        default=str(clean_dir / "equilibrium_firms.csv"),
        help="Path to equilibrium firms CSV file (for reference only)",
    )
    parser.add_argument(
        "--firms_noscreening_path",
        type=str,
        default=str(clean_dir / "equilibrium_firms_noscreening.csv"),
        help="Optional path to no-screening equilibrium firms CSV (for worker_noscreening dataset).",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(build_dir),
        help="Output directory for worker data (defaults to data/build/)",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=str(plot_dir),
        help="Output directory for plots (defaults to output/workers/)",
    )
    
    # Sampling parameters
    parser.add_argument("--seed", type=int, default=123,
                       help="Random seed for reproducibility")
    parser.add_argument("--validate_probabilities", action='store_true', default=False,
                       help="Run probability computation validation (default: False)")
    parser.add_argument("--debug_worker_idx", type=int, default=None,
                       help="Worker index for detailed probability debugging (default: None)")

    # Multi-market
    parser.add_argument("--M", type=int, default=1,
                       help="Number of markets. When > 1, draws workers per market from "
                            "data/clean/markets/equilibrium_firms_market_{m}.csv.")
    parser.add_argument("--drop_below_n", type=int, default=None,
                       help="Drop firms with fewer than this many matched workers. "
                            "Workers assigned to dropped firms are reassigned to their next best feasible choice.")

    args = parser.parse_args()
    M = args.M

    print("Worker Dataset Construction")
    print("=" * 50)

    params_path = Path(args.params_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_out = Path(args.plot_dir)
    plot_out.mkdir(parents=True, exist_ok=True)

    # Check file existence
    if not params_path.exists():
        print(f"Warning: {params_path} not found, trying data/raw/parameters.csv")
        fallback_path = raw_dir / "parameters.csv"
        if not fallback_path.exists():
            print(f"Error: Neither {params_path} nor {fallback_path} found")
            return 1
        params_path = fallback_path

    # Read parameters (shared across markets)
    params = read_params_long_csv(str(params_path))
    print(f"Loaded parameters: mu_s={params.get('mu_s', 'N/A')}, N_workers={params.get('N_workers', 'N/A')}")

    # --- Multi-market mode ---
    if M > 1:
        return _draw_workers_multi_market(args, M, params, raw_dir, clean_dir, build_dir, out_dir, plot_out)

    # --- Single-market mode ---
    firms_path = Path(args.firms_path)
    firms_noscreen_path = Path(args.firms_noscreening_path)

    if not firms_path.exists():
        print(f"Warning: {firms_path} not found")
    if not firms_noscreen_path.exists():
        print(f"Note: no-screening firms file {firms_noscreen_path} not found; skipping worker noscreening dataset.")

    # Read data
    print("Reading data...")

    # Read equilibrium firms (for utility computation)
    firms_data = None
    firms_df: pd.DataFrame | None = None
    if firms_path.exists():
        try:
            firms_df = pd.read_csv(firms_path)
            J = len(firms_df)
            if 'firm_id' in firms_df.columns:
                firm_ids = firms_df['firm_id'].values
            else:
                firm_ids = np.arange(1, J + 1)
            firms_data = {
                'w': firms_df['w'].values,
                'qbar': firms_df['qbar'].values,
                'xi': firms_df['xi'].values,
                'locations': firms_df[['x', 'y']].values,
                'firm_ids': firm_ids,
            }
            print(f"Loaded {J} firms from equilibrium data")
        except Exception as e:
            print(f"Warning: Could not read firms data: {e}")
            print("Proceeding without firm data - utilities will not be computed")
    
    # Construct worker dataset
    print(f"\nConstructing worker dataset...")
    print(f"Configuration:")
    print(f"  seed: {args.seed}")
    
    worker_dataset = construct_worker_dataset(
        params,
        firms_data=firms_data,
        seed=args.seed
    )

    firm_output_path: Path | None = None
    if (
        firms_data is not None
        and firms_df is not None
        and 'chosen_firm' in worker_dataset.columns
    ):
        drop_below_n = getattr(args, 'drop_below_n', None)

        # Iterative firm dropping with worker reassignment
        if drop_below_n is not None and drop_below_n > 0:
            print(f"Applying --drop_below_n={drop_below_n} ...")
            worker_dataset, firm_subset = _drop_and_reassign_firms(
                worker_dataset, firms_df, drop_below_n
            )

            if len(firm_subset) > 0:
                firm_output_path = out_dir / "firm_data.csv"
                firm_subset.to_csv(firm_output_path, index=False)
                print(f"Filtered firm data written to: {firm_output_path}")

                new_ids = firm_subset['firm_id'].values
                firms_data = {
                    'w': firm_subset['w'].to_numpy(),
                    'qbar': firm_subset['qbar'].to_numpy(),
                    'xi': firm_subset['xi'].to_numpy(),
                    'locations': firm_subset[['x', 'y']].to_numpy(),
                    'firm_ids': new_ids,
                }
            else:
                print("No firms survived the drop threshold; skipping firm_data.csv creation.")
        else:
            # Original behavior: drop firms with zero workers, reindex
            chosen_positions = np.sort(
                worker_dataset.loc[worker_dataset['chosen_firm'] > 0, 'chosen_firm']
                .astype(int)
                .unique()
            )
            if chosen_positions.size > 0:
                chosen_idx = chosen_positions - 1  # convert 1-based to 0-based index
                firm_subset = firms_df.iloc[chosen_idx].copy()

                # Preserve original identifiers before reindexing
                if 'firm_id' in firm_subset.columns:
                    firm_subset['original_firm_id'] = firm_subset['firm_id']
                else:
                    firm_subset['original_firm_id'] = chosen_positions

                new_ids = np.arange(1, chosen_positions.size + 1, dtype=int)
                mapping = {old: new for old, new in zip(chosen_positions, new_ids)}

                # Reindex chosen firm identifiers in worker data (keep outside option = 0)
                worker_dataset['chosen_firm_original_id'] = worker_dataset['chosen_firm']
                worker_dataset['chosen_firm'] = (
                    worker_dataset['chosen_firm']
                    .apply(lambda x: mapping.get(int(x), 0))
                    .astype(int)
                )

                # Update firm identifiers and write filtered firm data
                firm_subset['firm_id'] = new_ids
                firm_subset.reset_index(drop=True, inplace=True)
                firm_output_path = out_dir / "firm_data.csv"
                firm_subset.to_csv(firm_output_path, index=False)
                print(f"Filtered firm data written to: {firm_output_path}")

                # Refresh firms_data dictionary to reflect filtered/reindexed firms
                firms_data = {
                    'w': firm_subset['w'].to_numpy(),
                    'qbar': firm_subset['qbar'].to_numpy(),
                    'xi': firm_subset['xi'].to_numpy(),
                    'locations': firm_subset[['x', 'y']].to_numpy(),
                    'firm_ids': new_ids,
                }
            else:
                print("No firms were chosen by workers; skipping firm_data.csv creation.")

    if firms_data is not None:
        worker_locations = worker_dataset[['ell_x', 'ell_y']].to_numpy()
        firm_locations = firms_data['locations']

        fig, ax = plt.subplots(figsize=(8, 6))
        worker_skill = worker_dataset["s_skill"].to_numpy(dtype=float)
        worker_scatter = ax.scatter(
            worker_locations[:, 0],
            worker_locations[:, 1],
            s=10,
            c=worker_skill,
            cmap="cividis",
            alpha=0.35,
            label='Workers (skill-shaded)',
            edgecolors='none',
        )
        ax.scatter(
            firm_locations[:, 0],
            firm_locations[:, 1],
            s=80,
            marker='x',
            linewidths=2,
            color='red',
            label='Firms',
        )
        firm_ids = firms_data.get('firm_ids', np.arange(1, firm_locations.shape[0] + 1))
        for idx, (fx, fy) in enumerate(firm_locations):
            label = firm_ids[idx] if idx < len(firm_ids) else idx + 1
            ax.text(
                fx + 0.15,
                fy + 0.15,
                str(label),
                fontsize=10,
                fontweight="bold",
                color="red",
                bbox=dict(facecolor="white", edgecolor="red", alpha=0.8, boxstyle="round,pad=0.2"),
            )
        ax.set_xlabel('Worker x-location')
        ax.set_ylabel('Worker y-location')
        ax.set_title('Worker and Firm Locations')
        ax.legend(loc='best')
        ax.set_aspect('equal', adjustable='box')
        cbar = fig.colorbar(worker_scatter, ax=ax, pad=0.02)
        cbar.set_label('Worker skill')

        plot_path = plot_out / 'workers_firms_locations.png'
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Location scatter saved to: {plot_path}")

        if "chosen_firm" in worker_dataset.columns:
            chosen_firm = worker_dataset["chosen_firm"].to_numpy(dtype=int)
            outside_mask = chosen_firm <= 0
            chosen_mask = ~outside_mask

            fig, ax = plt.subplots(figsize=(8, 6))
            if np.any(outside_mask):
                ax.scatter(
                    worker_locations[outside_mask, 0],
                    worker_locations[outside_mask, 1],
                    s=10,
                    alpha=0.2,
                    color="0.75",
                    label="Outside option",
                    edgecolors="none",
                )
            if np.any(chosen_mask):
                firm_ids_unique = np.unique(chosen_firm[chosen_mask])
                palette = _categorical_palette(firm_ids_unique.size)
                color_map = {
                    int(fid): palette[idx % len(palette)]
                    for idx, fid in enumerate(firm_ids_unique.tolist())
                }
                chosen_colors = [color_map[int(fid)] for fid in chosen_firm[chosen_mask]]
                ax.scatter(
                    worker_locations[chosen_mask, 0],
                    worker_locations[chosen_mask, 1],
                    s=10,
                    alpha=0.6,
                    c=chosen_colors,
                    edgecolors="none",
                )

            ax.scatter(
                firm_locations[:, 0],
                firm_locations[:, 1],
                s=80,
                marker='x',
                linewidths=2,
                color='black',
                label='Firms',
            )
            firm_ids = firms_data.get('firm_ids', np.arange(1, firm_locations.shape[0] + 1))
            for idx, (fx, fy) in enumerate(firm_locations):
                label = firm_ids[idx] if idx < len(firm_ids) else idx + 1
                ax.text(
                    fx + 0.15,
                    fy + 0.15,
                    str(label),
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="black", alpha=0.8, boxstyle="round,pad=0.2"),
                )
            ax.set_xlabel('Worker x-location')
            ax.set_ylabel('Worker y-location')
            ax.set_title('Worker Locations Colored by Chosen Firm')
            if "firm_ids_unique" in locals() and firm_ids_unique.size <= 12:
                legend_handles = [
                    Line2D([0], [0], marker='o', color='none', markerfacecolor=color_map[int(fid)],
                           markersize=6, label=f"Firm {int(fid)}")
                    for fid in firm_ids_unique
                ]
                if np.any(outside_mask):
                    legend_handles.append(
                        Line2D([0], [0], marker='o', color='none', markerfacecolor="0.75",
                               markersize=6, label="Outside option")
                    )
                legend_handles.append(
                    Line2D([0], [0], marker='x', color='black', markersize=7, label="Firms")
                )
                ax.legend(handles=legend_handles, loc='best', ncol=2)
            else:
                ax.legend(loc='best')
            ax.set_aspect('equal', adjustable='box')

            plot_path = plot_out / 'workers_firms_locations_by_choice.png'
            fig.tight_layout()
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            print(f"Choice-colored location scatter saved to: {plot_path}")

        tau_val = float(params.get('tau', 0.05))
        eta_val = float(params.get('eta', 5.0))
        wages = np.asarray(firms_data['w'], dtype=float)
        xi_vals = np.asarray(firms_data['xi'], dtype=float)
        firm_ids = firms_data.get('firm_ids', np.arange(1, wages.size + 1))

        inclusive_vals = eta_val * np.log(np.maximum(wages, 1e-300)) + xi_vals

        deltas = worker_locations[:, None, :] - firm_locations[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        avg_distance = distances.mean(axis=0)
        sd_distance = distances.std(axis=0)

        distance_table = pd.DataFrame({
            'firm_id': firm_ids,
            'V_inclusive_value': inclusive_vals,
            'avg_distance': avg_distance,
            'sd_distance': sd_distance,
            'tau': tau_val,
        })

        table_path = out_dir / 'firm_distance_summary.csv'
        distance_table.to_csv(table_path, index=False)
        print(f"Firm distance summary written to: {table_path}")
        print(f"  tau = {tau_val:.4f}")
    
    # Run simulation-based probability validation if requested
    if args.validate_probabilities and firms_data is not None and args.debug_worker_idx is not None:
        # Need to prepare firms_data in the format expected by validation
        firms_df = pd.read_csv(firms_path)
        firms_validation_data = {
            'w': firms_df['w'].values,
            'qbar': firms_df['qbar'].values,
            'xi': firms_df['xi'].values,
            'A': firms_df['A'].values if 'A' in firms_df.columns else np.ones(len(firms_df)),
            'loc_firms': firms_df[['x', 'y']].values,
            'firm_ids': np.arange(1, len(firms_df) + 1)  # firm IDs 1, 2, ..., J
        }
        
        simulation_results = simulate_worker_choice_probabilities(
            args.debug_worker_idx,
            worker_dataset,
            firms_validation_data,
            params,
            out_dir=out_dir,
        )
        
        # Save simulation results
        simulation_path = out_dir / f"worker_{args.debug_worker_idx}_simulation.json"
        with open(simulation_path, 'w') as f:
            json.dump(simulation_results, f, indent=2)
        print(f"\nSimulation results saved to: {simulation_path}")
        
    elif args.validate_probabilities and args.debug_worker_idx is None:
        print("\nWarning: --debug_worker_idx must be specified for simulation validation")
        print("Usage: --validate_probabilities --debug_worker_idx=0")
        
    elif args.validate_probabilities and firms_data is None:
        print("\nWarning: Cannot run probability validation without firms data")
        print("Make sure --firms_path points to a valid equilibrium_firms.csv file")
    
    # Write output
    output_path = out_dir / "workers_dataset.csv"
    worker_dataset.to_csv(output_path, index=False)
    print(f"\nWorker dataset written to: {output_path}")

    # Write no-screening worker dataset if available (recompute choices with no cutoffs)
    if firms_noscreen_path.exists():
        try:
            noscreen_df = pd.read_csv(firms_noscreen_path)
            if "firm_id" not in noscreen_df.columns:
                noscreen_df["firm_id"] = np.arange(1, len(noscreen_df) + 1, dtype=int)

            chosen_ids = compute_choices_no_cutoff(worker_dataset, noscreen_df, params, seed=args.seed + 1)
            worker_dataset_noscreen = worker_dataset.copy()
            worker_dataset_noscreen["chosen_firm_original_id"] = chosen_ids

            # Reindex to consecutive firm IDs for the chosen set
            chosen_positions = np.sort(np.unique(chosen_ids[chosen_ids > 0]).astype(int))
            if chosen_positions.size > 0:
                new_ids = np.arange(1, chosen_positions.size + 1, dtype=int)
                mapping = {old: new for old, new in zip(chosen_positions, new_ids)}
                worker_dataset_noscreen["chosen_firm"] = worker_dataset_noscreen["chosen_firm_original_id"].apply(
                    lambda x: mapping.get(int(x), 0)
                )
                firm_subset_noscreen = noscreen_df.loc[
                    noscreen_df["firm_id"].isin(chosen_positions)
                ].copy()
                firm_subset_noscreen["original_firm_id"] = firm_subset_noscreen["firm_id"]
                firm_subset_noscreen["firm_id"] = firm_subset_noscreen["firm_id"].map(mapping)
                # Map wages for convenience
                wage_col = "w_noscreening" if "w_noscreening" in noscreen_df.columns else "w"
                wage_map = pd.Series(noscreen_df[wage_col].values, index=noscreen_df["firm_id"]).to_dict()
                worker_dataset_noscreen["wage_worker_noscreening"] = worker_dataset_noscreen[
                    "chosen_firm_original_id"
                ].map(wage_map)
            else:
                worker_dataset_noscreen["chosen_firm"] = 0
                worker_dataset_noscreen["wage_worker_noscreening"] = np.nan

            noscreen_out = out_dir / "workers_dataset_noscreening.csv"
            worker_dataset_noscreen.to_csv(noscreen_out, index=False)
            print(f"Worker no-screening dataset written to: {noscreen_out}")
        except Exception as exc:
            print(f"Warning: Could not create worker no-screening dataset: {exc}")
    
    # Write summary
    summary = worker_dataset.attrs['summary']
    summary_data = {
        'parameter': [
            'N_workers', 'mu_x_skill', 'sigma_x_skill', 'mu_e', 'sigma_e',
            'mu_s', 'sigma_s', 'worker_loc_mode', 'worker_mu_x', 'worker_mu_y', 'worker_sigma_x',
            'worker_sigma_y', 'worker_rho', 'worker_r_mu', 'worker_r_sigma',
            'rho_x_skill_ell_x', 'rho_x_skill_ell_y', 'rho_x_skill_r',
            'x_skill_mean', 'x_skill_std', 'ell_x_mean', 'ell_x_std', 'ell_y_mean', 'ell_y_std',
            'location_corr_empirical', 'x_skill_ell_x_corr_empirical', 'x_skill_ell_y_corr_empirical',
            'x_skill_r_corr_empirical',
            'a_skill_mean', 'a_skill_std', 's_skill_mean', 's_skill_std'
        ],
        'value': [
            summary['N_workers'], summary['mu_x_skill'], summary['sigma_x_skill'],
            summary['mu_e'], summary['sigma_e'], summary['mu_s'], summary['sigma_s'],
            summary['worker_loc_mode'], summary['worker_mu_x'], summary['worker_mu_y'],
            summary['worker_sigma_x'], summary['worker_sigma_y'], summary['worker_rho'],
            summary['worker_r_mu'], summary['worker_r_sigma'],
            summary['rho_x_skill_ell_x'], summary['rho_x_skill_ell_y'], summary['rho_x_skill_r'],
            summary['x_skill_mean'], summary['x_skill_std'],
            summary['ell_x_mean'], summary['ell_x_std'], summary['ell_y_mean'],
            summary['ell_y_std'], summary['location_corr_empirical'],
            summary['x_skill_ell_x_corr_empirical'], summary['x_skill_ell_y_corr_empirical'],
            summary['x_skill_r_corr_empirical'],
            summary['a_skill_mean'], summary['a_skill_std'], summary['s_skill_mean'], summary['s_skill_std']
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = out_dir / "workers_dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary written to: {summary_path}")
    
    print(f"\n=== OUTPUT FILES ===")
    print(f"  {output_path}")
    print(f"  {summary_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
