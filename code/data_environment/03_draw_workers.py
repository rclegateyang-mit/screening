#!/usr/bin/env python3
"""
Worker Dataset Construction

This script constructs a worker dataset by drawing worker characteristics from continuous distributions.
It reads effective parameters to generate workers with skill, location, and auxiliary skill components.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

if __package__ is None or __package__ == "":
    import sys

    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
    from code import get_data_dir  # type: ignore
    from code.estimation.helpers import (  # type: ignore
        compute_order_maps,
        read_params_long_csv,
    )
else:  # pragma: no cover - executed when running as package module
    from .. import get_data_dir
    from ..estimation.helpers import compute_order_maps, read_params_long_csv


DEFAULT_DATA_DIR = get_data_dir()


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def suffix_sum(arr: np.ndarray) -> np.ndarray:
    """Compute suffix sums: result[i] = sum(arr[i:])"""
    return np.cumsum(arr[::-1])[::-1]





def simulate_worker_choice_probabilities(
    worker_idx: int,
    df: pd.DataFrame,
    firms_data: Dict[str, Any],
    params: Dict[str, Any],
    n_simulations: int = 10000,
    seed: int = 123,
    out_dir: Path | str = DEFAULT_DATA_DIR,
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
    c = firms_data['c']
    xi = firms_data['xi']
    A = firms_data['A']
    loc_firms = firms_data['loc_firms']
    firm_ids = firms_data['firm_ids']
    J = len(w)
    
    # Extract parameters
    alpha = float(params['alpha'])
    beta = float(params['beta'])
    gamma = float(params.get('gamma', 0.1))
    sigma_a = float(params.get('sigma_a', 1.5))
    
    print(f"Worker characteristics:")
    print(f"  x_skill: {x_skill:.6f}")
    print(f"  location: ({ell_x:.6f}, {ell_y:.6f})")
    print(f"Parameters:")
    print(f"  alpha: {alpha}")
    print(f"  gamma: {gamma}")
    print(f"  sigma_a: {sigma_a}")
    
    # SIMULATION using Type 1 extreme value shocks
    rng = np.random.default_rng(seed)
    
    # Distances to firms
    distances = np.sqrt((ell_x - loc_firms[:, 0])**2 + (ell_y - loc_firms[:, 1])**2)
    
    # Systematic utilities: U_ij = alpha * log(w_j) + xi_j - gamma * distance_ij
    U_systematic = alpha * np.log(w) + xi - gamma * distances  # (J,)
    U_outside_systematic = 0.0  # Outside option normalized to 0
    
    print(f"\nFirm data:")
    for j in range(J):
        print(f"  Firm {firm_ids[j]}: w={w[j]:.6f}, c={c[j]:.6f}, xi={xi[j]:.6f}, distance={distances[j]:.6f}, U_sys={U_systematic[j]:.6f}")
    
    print(f"\nRunning {n_simulations} simulations, saving to CSV...")
    
    # Prepare lists to store simulation data
    sim_data = []
    choices = np.zeros(n_simulations, dtype=int)  # 0 = outside, 1,...,J = firms
    
    for sim in range(n_simulations):
        # Draw a_skill for this simulation
        a_skill_sim = rng.normal(0, sigma_a)  # a_skill ~ N(0, σ_a²)
        s_skill_sim = x_skill + a_skill_sim   # Total skill for this simulation
        
        # Check which firms are feasible for this skill draw
        feasible_firms_sim = s_skill_sim > c  # (J,) boolean array
        
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
            'sigma_a': float(sigma_a)
        },
        'simulation_setup': {
            'c_firms': c.tolist(),
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
    
    # New skill distribution component parameters
    mu_x_skill = float(params.get('mu_x_skill', params.get('mu_s', 10.009)))  # Fallback for backward compatibility
    sigma_x_skill = float(params.get('sigma_x_skill', 2.6))
    mu_a_skill = float(params.get('mu_a_skill', 0.0))
    sigma_a_skill = float(params.get('sigma_a_skill', 1.5))
    
    # Derived parameters (for validation)
    mu_s_computed = mu_x_skill + mu_a_skill
    sigma_s_computed = np.sqrt(sigma_x_skill**2 + sigma_a_skill**2)
    
    # Set up random number generator
    rng = np.random.default_rng(seed)
    
    print(f"Drawing {N_workers} workers from continuous distributions...")
    print(f"  x_skill ~ N({mu_x_skill:.3f}, {sigma_x_skill:.3f}²)")
    print(f"  a_skill ~ N({mu_a_skill:.1f}, {sigma_a_skill:.3f}²)")
    print(f"  s_skill = x_skill + a_skill  =>  s_skill ~ N({mu_s_computed:.3f}, {sigma_s_computed:.3f}²)")
    print(f"  locations ~ BivarN([{worker_mu_x:.1f}, {worker_mu_y:.1f}], σ=({worker_sigma_x:.1f}, {worker_sigma_y:.1f}), ρ={worker_rho:.1f})")
    
    # Draw continuous samples
    # 1) Draw x_skill from N(mu_x_skill, sigma_x_skill²)
    x_skill = rng.normal(mu_x_skill, sigma_x_skill, N_workers)
    
    # 2) Draw locations from bivariate normal distribution
    # Create covariance matrix for bivariate normal
    cov_matrix = np.array([
        [worker_sigma_x**2, worker_rho * worker_sigma_x * worker_sigma_y],
        [worker_rho * worker_sigma_x * worker_sigma_y, worker_sigma_y**2]
    ])
    mean_vector = np.array([worker_mu_x, worker_mu_y])
    
    # Draw from bivariate normal
    locations = rng.multivariate_normal(mean_vector, cov_matrix, N_workers)
    ell_x = locations[:, 0]
    ell_y = locations[:, 1]
    
    # 3) Draw a_skill from N(mu_a_skill, sigma_a_skill²)
    a_skill = rng.normal(mu_a_skill, sigma_a_skill, N_workers)
    
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
        alpha = float(params.get('alpha', 5.0))
        gamma = float(params.get('gamma', 0.05))
        
        w_firms = firms_data['w']  # wages
        c_firms = firms_data['c']  # cutoffs  
        xi_firms = firms_data['xi']  # firm fixed effects
        loc_firms = firms_data['locations']  # firm locations (J x 2)
        J = len(w_firms)
        
        print(f"Computing utilities and firm choices for {J} firms...")
        
        # Draw Type I extreme value errors for all worker-firm pairs
        # u_i0 = e_i0 (outside option, only error term)
        # u_ij = alpha * log(w_j) + xi_j - gamma * ||ell_i - ell_j|| + e_ij
        
        # Draw extreme value errors (Gumbel distribution with location=0, scale=1)
        # For each worker: one error for outside option + J errors for firms
        errors = rng.gumbel(0, 1, size=(N_workers, J + 1))  # Shape: (N_workers, J+1)
        
        # Compute utilities
        utilities = np.zeros((N_workers, J + 1))  # Column 0 = outside, 1..J = firms
        
        # Outside option utility: u_i0 = e_i0
        utilities[:, 0] = errors[:, 0]
        
        # Firm utilities: u_ij = alpha * log(w_j) + xi_j - gamma * ||ell_i - ell_j|| + e_ij
        for j in range(J):
            # Distance between worker i and firm j
            worker_locations = np.column_stack([ell_x, ell_y])  # (N_workers, 2)
            firm_location = loc_firms[j]  # (2,)
            distances = np.linalg.norm(worker_locations - firm_location, axis=1)  # (N_workers,)
            
            # Utility components
            log_wage = np.log(w_firms[j])
            xi_j = xi_firms[j]
            
            # Total utility for firm j
            utilities[:, j + 1] = alpha * log_wage + xi_j - gamma * distances + errors[:, j + 1]
        
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
            
            # Find feasible firms where s_i > c_j
            feasible_firms = []
            feasible_utilities = []
            3
            # Outside option is always feasible
            feasible_firms.append(0)
            feasible_utilities.append(utilities[i, 0])
            
            # Check firms where skill constraint is satisfied
            for j in range(J):
                if s_i > c_firms[j]:
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
    print(f"σ_x: {sigma_x_skill:.3f}, σ_a_skill: {sigma_a_skill:.3f}")
    print(f"μ_a_skill: {mu_a_skill:.3f}")
    print(f"Location params: μ=({worker_mu_x:.1f}, {worker_mu_y:.1f}), σ=({worker_sigma_x:.1f}, {worker_sigma_y:.1f}), ρ={worker_rho:.1f}")
    
    # Basic statistics
    print(f"x_skill: mean={x_skill.mean():.3f}, std={x_skill.std():.3f}")
    print(f"ell_x: mean={ell_x.mean():.3f}, std={ell_x.std():.3f}")
    print(f"ell_y: mean={ell_y.mean():.3f}, std={ell_y.std():.3f}")
    print(f"a_skill: mean={a_skill.mean():.3f}, std={a_skill.std():.3f}")
    print(f"s_skill: mean={s_skill.mean():.3f}, std={s_skill.std():.3f}")
    
    # Compute empirical correlation for validation
    empirical_corr = np.corrcoef(ell_x, ell_y)[0, 1]
    print(f"Location correlation: empirical={empirical_corr:.3f}, expected={worker_rho:.1f}")
    
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
        'mu_a_skill': mu_a_skill,
        'sigma_a_skill': sigma_a_skill,
        'worker_mu_x': worker_mu_x,
        'worker_mu_y': worker_mu_y,
        'worker_sigma_x': worker_sigma_x,
        'worker_sigma_y': worker_sigma_y,
        'worker_rho': worker_rho,
        'x_skill_mean': x_skill.mean(),
        'x_skill_std': x_skill.std(),
        'ell_x_mean': ell_x.mean(),
        'ell_x_std': ell_x.std(),
        'ell_y_mean': ell_y.mean(),
        'ell_y_std': ell_y.std(),
        'location_corr_empirical': empirical_corr,
        'a_skill_mean': a_skill.mean(),
        'a_skill_std': a_skill.std(),
        's_skill_mean': s_skill.mean(),
        's_skill_std': s_skill.std()
    }
    
    return df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(description="Construct worker dataset with continuous sampling")
    data_dir = get_data_dir(create=True)
    
    # File paths
    parser.add_argument(
        "--params_path",
        type=str,
        default=str(data_dir / "parameters_effective.csv"),
        help="Path to parameters CSV file",
    )
    parser.add_argument(
        "--firms_path",
        type=str,
        default=str(data_dir / "equilibrium_firms.csv"),
        help="Path to equilibrium firms CSV file (for reference only)",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(data_dir),
        help="Output directory for worker data (defaults to project data/ folder)",
    )
    
    # Sampling parameters
    parser.add_argument("--seed", type=int, default=123,
                       help="Random seed for reproducibility")
    parser.add_argument("--validate_probabilities", action='store_true', default=False,
                       help="Run probability computation validation (default: False)")
    parser.add_argument("--debug_worker_idx", type=int, default=None,
                       help="Worker index for detailed probability debugging (default: None)")
    

    
    args = parser.parse_args()
    
    print("Worker Dataset Construction")
    print("=" * 50)
    
    params_path = Path(args.params_path)
    firms_path = Path(args.firms_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check file existence
    if not params_path.exists():
        print(f"Warning: {params_path} not found, trying data/parameters.csv")
        fallback_path = data_dir / "parameters.csv"
        if not fallback_path.exists():
            print(f"Error: Neither {params_path} nor {fallback_path} found")
            return 1
        params_path = fallback_path
    
    if not firms_path.exists():
        print(f"Warning: {firms_path} not found")
    
    # Read data
    print("Reading data...")
    
    # Read parameters
    params = read_params_long_csv(str(params_path))
    print(f"Loaded parameters: mu_s={params.get('mu_s', 'N/A')}, N_workers={params.get('N_workers', 'N/A')}")
    
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
                'c': firms_df['c'].values,
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
                'c': firm_subset['c'].to_numpy(),
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
        ax.scatter(
            worker_locations[:, 0],
            worker_locations[:, 1],
            s=10,
            alpha=0.25,
            label='Workers',
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
        ax.set_xlabel('Worker x-location')
        ax.set_ylabel('Worker y-location')
        ax.set_title('Worker and Firm Locations')
        ax.legend(loc='best')
        ax.set_aspect('equal', adjustable='box')

        plot_path = out_dir / 'workers_firms_locations.png'
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Location scatter saved to: {plot_path}")

        gamma_val = float(params.get('gamma', 0.05))
        alpha_val = float(params.get('alpha', 5.0))
        wages = np.asarray(firms_data['w'], dtype=float)
        xi_vals = np.asarray(firms_data['xi'], dtype=float)
        firm_ids = firms_data.get('firm_ids', np.arange(1, wages.size + 1))

        inclusive_vals = alpha_val * np.log(np.maximum(wages, 1e-300)) + xi_vals

        deltas = worker_locations[:, None, :] - firm_locations[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        avg_distance = distances.mean(axis=0)
        sd_distance = distances.std(axis=0)

        distance_table = pd.DataFrame({
            'firm_id': firm_ids,
            'V_inclusive_value': inclusive_vals,
            'avg_distance': avg_distance,
            'sd_distance': sd_distance,
            'gamma': gamma_val,
        })

        table_path = out_dir / 'firm_distance_summary.csv'
        distance_table.to_csv(table_path, index=False)
        print(f"Firm distance summary written to: {table_path}")
        print(f"  gamma = {gamma_val:.4f}")
    
    # Run simulation-based probability validation if requested
    if args.validate_probabilities and firms_data is not None and args.debug_worker_idx is not None:
        # Need to prepare firms_data in the format expected by validation
        firms_df = pd.read_csv(firms_path)
        firms_validation_data = {
            'w': firms_df['w'].values,
            'c': firms_df['c'].values,
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
    
    # Write summary
    summary = worker_dataset.attrs['summary']
    summary_data = {
        'parameter': [
            'N_workers', 'mu_x_skill', 'sigma_x_skill', 'mu_a_skill', 'sigma_a_skill', 
            'mu_s', 'sigma_s', 'worker_mu_x', 'worker_mu_y', 'worker_sigma_x', 'worker_sigma_y', 'worker_rho',
            'x_skill_mean', 'x_skill_std', 'ell_x_mean', 'ell_x_std', 'ell_y_mean', 'ell_y_std',
            'location_corr_empirical', 'a_skill_mean', 'a_skill_std', 's_skill_mean', 's_skill_std'
        ],
        'value': [
            summary['N_workers'], summary['mu_x_skill'], summary['sigma_x_skill'],
            summary['mu_a_skill'], summary['sigma_a_skill'], summary['mu_s'], summary['sigma_s'],
            summary['worker_mu_x'], summary['worker_mu_y'], summary['worker_sigma_x'], 
            summary['worker_sigma_y'], summary['worker_rho'], summary['x_skill_mean'], summary['x_skill_std'],
            summary['ell_x_mean'], summary['ell_x_std'], summary['ell_y_mean'],
            summary['ell_y_std'], summary['location_corr_empirical'],
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
