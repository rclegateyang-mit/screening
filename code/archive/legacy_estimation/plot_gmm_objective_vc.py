#!/usr/bin/env python3
"""Plot the negative GMM objective over (V, c) for J=1 with γ fixed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .jax_model import enable_x64, compute_choice_probabilities_jax
    from .g_features import chamberlain_instruments_jax
    from .moments import criterion as criterion_jax
    from .helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from jax_model import enable_x64, compute_choice_probabilities_jax
    from g_features import chamberlain_instruments_jax
    from moments import criterion as criterion_jax
    from helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent.parent
    output_dir = root / "output"

    parser = argparse.ArgumentParser(
        description="Plot negative GMM objective over (V, c) for J=1 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--workers_path', type=str, default=str(output_dir / 'workers_dataset.csv'))
    parser.add_argument('--firms_path', type=str, default=str(output_dir / 'equilibrium_firms.csv'))
    parser.add_argument('--params_path', type=str, default=str(output_dir / 'parameters_effective.csv'))
    parser.add_argument('--truth_path', type=str, default=None,
                        help='Optional JSON with true theta (fields gamma, V, c).')
    parser.add_argument('--out_dir', type=str, default=str(output_dir))
    parser.add_argument('--plot_filename', type=str, default='gmm_objective_v_c.png')
    parser.add_argument('--save_grid', action='store_true', help='Persist the evaluated grid to NPZ.')

    parser.add_argument('--v_min', type=float, default=None)
    parser.add_argument('--v_max', type=float, default=None)
    parser.add_argument('--v_steps', type=int, default=75)
    parser.add_argument('--c_min', type=float, default=None)
    parser.add_argument('--c_max', type=float, default=None)
    parser.add_argument('--c_steps', type=int, default=75)
    parser.add_argument('--gamma', type=float, default=None,
                        help='Override gamma; defaults to params or truth if provided.')
    parser.add_argument('--gamma_min', type=float, default=0.0)
    parser.add_argument('--gamma_max', type=float, default=0.15)
    parser.add_argument('--gamma_steps', type=int, default=75)
    parser.add_argument('--threads', type=int, default=None,
                        help='Optional CPU thread pool size (XLA).')
    parser.add_argument(
        '--step1_instruments_path',
        type=str,
        default=str(output_dir / 'gmm_gamma_Vc_estimates_jax.json'),
        help='Path to gmm_gamma_Vc_estimates_jax.json; use step 1 θ̂ for Chamberlain instruments (set empty to disable).',
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.threads is not None:
        print("Set CPU threads by exporting before invocation:")
        print(f'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_cpu_thread_pool_size={int(args.threads)}"')

    enable_x64()

    params = read_parameters(args.params_path)
    firm_ids, w, _, _, xi, loc_firms, c_data = read_firms_data(args.firms_path)
    x_skill, ell_x, ell_y, chosen_firm = read_workers_data(args.workers_path)

    J = firm_ids.size
    if J != 1:
        raise ValueError(f"This visualisation only supports J=1; got J={J}.")

    w = np.asarray(w, dtype=float)
    xi = np.asarray(xi, dtype=float)
    c_data = np.asarray(c_data, dtype=float)

    alpha = float(params.get('alpha', 1.0))
    phi = float(params.get('varphi', params.get('phi', 1.0)))
    mu_a = float(params.get('mu_a', 0.0))
    sigma_a = float(params.get('sigma_a', 0.12))

    gamma_fixed = None
    if args.truth_path is not None:
        with open(args.truth_path, 'r') as f:
            truth = json.load(f)
        gamma_fixed = truth.get('gamma', gamma_fixed)

    if args.gamma is not None:
        gamma_fixed = args.gamma

    if gamma_fixed is None:
        gamma_fixed = float(params.get('gamma', 0.05))

    gamma_fixed = float(gamma_fixed)

    V0_nat = alpha * np.log(np.maximum(w, 1e-300)) + xi
    if not c_data.size or c_data[0] <= 0:
        raise ValueError("Firms CSV must provide positive cutoff 'c' for J=1 case.")
    c0 = float(c_data[0])

    V_center = float(V0_nat[0])
    c_center = c0
    theta_baseline: Optional[jnp.ndarray] = None

    step1_path_raw = (args.step1_instruments_path or '').strip()
    if step1_path_raw:
        estimates_path = Path(step1_path_raw)
        if estimates_path.is_dir():
            estimates_path = estimates_path / 'gmm_gamma_Vc_estimates_jax.json'
        if estimates_path.exists():
            try:
                with open(estimates_path, 'r') as f:
                    estimates = json.load(f)
                steps = estimates.get('steps', [])
                step1_entry = None
                for entry in steps:
                    if entry.get('step') == 1:
                        step1_entry = entry
                        break
                if step1_entry is None and steps:
                    step1_entry = steps[0]
                if step1_entry and 'theta_hat' in step1_entry:
                    theta_step1 = np.asarray(step1_entry['theta_hat'], dtype=float)
                    expected = 1 + 2 * J
                    if theta_step1.size >= expected:
                        if theta_step1.size > expected:
                            print(
                                f"Warning: step 1 theta_hat length {theta_step1.size} exceeds expected {expected}; truncating to first {expected} entries."
                            )
                        theta_trimmed = theta_step1[:expected]
                        theta_baseline = jnp.asarray(theta_trimmed, dtype=jnp.float64)
                        if args.gamma is None:
                            gamma_fixed = float(theta_trimmed[0])
                        V_center = float(theta_trimmed[1])
                        c_center = float(theta_trimmed[1 + J])
                        print(f"Using step 1 θ̂ from {estimates_path} for instrument construction.")
                    else:
                        print(
                            f"Warning: step 1 theta_hat length {theta_step1.size} < expected {expected}; using baseline from data."
                        )
                else:
                    print(f"Warning: No usable step 1 entry in {estimates_path}; using baseline from data.")
            except (OSError, json.JSONDecodeError) as err:
                print(f"Warning: Failed to read {estimates_path}: {err}; using baseline from data.")
        else:
            print(f"Warning: step1_instruments_path {estimates_path} not found; using baseline from data.")

    D_nat = compute_worker_firm_distances(ell_x, ell_y, loc_firms)

    aux: Dict = {
        'D_nat': jnp.asarray(D_nat, dtype=jnp.float64),
        'phi': phi,
        'mu_a': mu_a,
        'sigma_a': sigma_a,
        'firm_ids': jnp.asarray(firm_ids, dtype=jnp.int32),
    }

    X = jnp.asarray(x_skill, dtype=jnp.float64)

    chosen = chosen_firm.astype(int)
    Y = np.zeros((chosen.size, J + 1))
    Y[np.arange(chosen.size), chosen] = 1.0
    Y_jax = jnp.asarray(Y, dtype=jnp.float64)

    def build_instruments(theta: jnp.ndarray) -> jnp.ndarray:
        def prob_eval_theta(theta_arr: jnp.ndarray) -> jnp.ndarray:
            return compute_choice_probabilities_jax(theta_arr, X, aux)

        return chamberlain_instruments_jax(theta, prob_eval_theta, check_rowsum=True)

    if theta_baseline is None:
        theta_baseline = jnp.asarray(np.concatenate(([gamma_fixed], V0_nat, np.array([c0]))), dtype=jnp.float64)

    G_feat = build_instruments(theta_baseline)

    def gmm_objective(theta: jnp.ndarray) -> jnp.ndarray:
        return criterion_jax(theta, X, Y_jax, G_feat, aux)

    gmm_objective = jax.jit(gmm_objective)

    v_min = args.v_min if args.v_min is not None else float(V_center - 2.0)
    v_max = args.v_max if args.v_max is not None else float(V_center + 2.0)
    if v_min >= v_max:
        raise ValueError("Require v_min < v_max")

    if args.c_min is not None:
        c_min = max(args.c_min, 1e-6)
    else:
        c_min = c_center * 0.2

    if args.c_max is not None:
        c_max = max(args.c_max, c_min * 1.01)
    else:
        c_max = c_center * 5.0

    if c_min <= 0 or c_min >= c_max:
        raise ValueError("Require 0 < c_min < c_max")

    v_grid = np.linspace(v_min, v_max, int(args.v_steps))
    c_grid = np.geomspace(c_min, c_max, int(args.c_steps))

    gamma_steps = int(args.gamma_steps)
    gamma_grid = np.linspace(float(args.gamma_min), float(args.gamma_max), gamma_steps)
    if gamma_grid.size == 0:
        raise ValueError("Gamma grid is empty; adjust gamma_min/gamma_max/gamma_steps.")

    neg_obj_vc = np.full((c_grid.size, v_grid.size), np.nan, dtype=float)
    neg_obj_gamma_v = np.full((v_grid.size, gamma_grid.size), np.nan, dtype=float)
    neg_obj_gamma_c = np.full((c_grid.size, gamma_grid.size), np.nan, dtype=float)

    profile_neg_vc = np.full_like(neg_obj_vc, -np.inf)
    profile_neg_gamma_v = np.full_like(neg_obj_gamma_v, -np.inf)
    profile_neg_gamma_c = np.full_like(neg_obj_gamma_c, -np.inf)

    for i_c, c_val in enumerate(c_grid):
        for j_v, V_val in enumerate(v_grid):
            theta_vec = jnp.asarray([gamma_fixed, V_val, c_val], dtype=jnp.float64)
            try:
                obj = float(gmm_objective(theta_vec))
            except FloatingPointError:
                obj = float('inf')
            neg_obj_vc[i_c, j_v] = -obj if np.isfinite(obj) else np.nan

            best_val = -np.inf
            for gamma_val in gamma_grid:
                theta_prof = jnp.asarray([gamma_val, V_val, c_val], dtype=jnp.float64)
                try:
                    obj_prof = float(gmm_objective(theta_prof))
                except FloatingPointError:
                    obj_prof = float('inf')
                neg_prof = -obj_prof if np.isfinite(obj_prof) else -np.inf
                if neg_prof > best_val:
                    best_val = neg_prof
            profile_neg_vc[i_c, j_v] = best_val

    for j_gamma, gamma_val in enumerate(gamma_grid):
        for i_v, V_val in enumerate(v_grid):
            theta_vec = jnp.asarray([gamma_val, V_val, c0], dtype=jnp.float64)
            try:
                obj = float(gmm_objective(theta_vec))
            except FloatingPointError:
                obj = float('inf')
            neg_obj_gamma_v[i_v, j_gamma] = -obj if np.isfinite(obj) else np.nan

            best_val = -np.inf
            for c_val in c_grid:
                theta_prof = jnp.asarray([gamma_val, V_val, c_val], dtype=jnp.float64)
                try:
                    obj_prof = float(gmm_objective(theta_prof))
                except FloatingPointError:
                    obj_prof = float('inf')
                neg_prof = -obj_prof if np.isfinite(obj_prof) else -np.inf
                if neg_prof > best_val:
                    best_val = neg_prof
            profile_neg_gamma_v[i_v, j_gamma] = best_val

    for i_c, c_val in enumerate(c_grid):
        for j_gamma, gamma_val in enumerate(gamma_grid):
            theta_vec = jnp.asarray([gamma_val, V0_nat[0], c_val], dtype=jnp.float64)
            try:
                obj = float(gmm_objective(theta_vec))
            except FloatingPointError:
                obj = float('inf')
            neg_obj_gamma_c[i_c, j_gamma] = -obj if np.isfinite(obj) else np.nan

            best_val = -np.inf
            for V_val in v_grid:
                theta_prof = jnp.asarray([gamma_val, V_val, c_val], dtype=jnp.float64)
                try:
                    obj_prof = float(gmm_objective(theta_prof))
                except FloatingPointError:
                    obj_prof = float('inf')
                neg_prof = -obj_prof if np.isfinite(obj_prof) else -np.inf
                if neg_prof > best_val:
                    best_val = neg_prof
            profile_neg_gamma_c[i_c, j_gamma] = best_val

    if not np.any(np.isfinite(neg_obj_vc)):
        raise RuntimeError("GMM objective evaluation failed for (V,c) grid.")

    idx = np.nanargmax(neg_obj_vc)
    i_best_c, i_best_v = np.unravel_index(idx, neg_obj_vc.shape)
    best_V = v_grid[i_best_v]
    best_c = c_grid[i_best_c]
    best_val = neg_obj_vc[i_best_c, i_best_v]
    print(f"Max (negative GMM objective) on grid: {best_val:.3f} at V={best_V:.3f}, c={best_c:.6f}")

    idx_gv = np.nanargmax(neg_obj_gamma_v)
    i_best_v_gv, i_best_gamma_gv = np.unravel_index(idx_gv, neg_obj_gamma_v.shape)
    best_gamma_gv = gamma_grid[i_best_gamma_gv]
    best_v_gv = v_grid[i_best_v_gv]
    best_val_gv = neg_obj_gamma_v[i_best_v_gv, i_best_gamma_gv]
    print(f"Max (negative GMM) on (gamma,V): {best_val_gv:.3f} at gamma={best_gamma_gv:.4f}, V={best_v_gv:.3f}")

    idx_gc = np.nanargmax(neg_obj_gamma_c)
    i_best_c_gc, i_best_gamma_gc = np.unravel_index(idx_gc, neg_obj_gamma_c.shape)
    best_gamma_gc = gamma_grid[i_best_gamma_gc]
    best_c_gc = c_grid[i_best_c_gc]
    best_val_gc = neg_obj_gamma_c[i_best_c_gc, i_best_gamma_gc]
    print(f"Max (negative GMM) on (gamma,c): {best_val_gc:.3f} at gamma={best_gamma_gc:.4f}, c={best_c_gc:.6f}")

    profile_vc_plot = np.where(np.isfinite(profile_neg_vc), profile_neg_vc, np.nan)
    profile_gv_plot = np.where(np.isfinite(profile_neg_gamma_v), profile_neg_gamma_v, np.nan)
    profile_gc_plot = np.where(np.isfinite(profile_neg_gamma_c), profile_neg_gamma_c, np.nan)

    prof_best_v = prof_best_c = None
    prof_best_val_vc = None
    if np.isfinite(profile_vc_plot).any():
        idx_prof_vc = np.nanargmax(profile_vc_plot)
        prof_c_idx, prof_v_idx = np.unravel_index(idx_prof_vc, profile_vc_plot.shape)
        prof_best_v = v_grid[prof_v_idx]
        prof_best_c = c_grid[prof_c_idx]
        prof_best_val_vc = profile_vc_plot[prof_c_idx, prof_v_idx]
        print(
            f"Profile -GMM over (V,c): {prof_best_val_vc:.3f} at V={prof_best_v:.3f}, c={prof_best_c:.6f}"
        )

    prof_best_gamma_gv = prof_best_v_gv = None
    prof_best_val_gv = None
    if np.isfinite(profile_gv_plot).any():
        idx_prof_gv = np.nanargmax(profile_gv_plot)
        prof_v_gv_idx, prof_gamma_gv_idx = np.unravel_index(idx_prof_gv, profile_gv_plot.shape)
        prof_best_gamma_gv = gamma_grid[prof_gamma_gv_idx]
        prof_best_v_gv = v_grid[prof_v_gv_idx]
        prof_best_val_gv = profile_gv_plot[prof_v_gv_idx, prof_gamma_gv_idx]
        print(
            f"Profile -GMM over (gamma,V): {prof_best_val_gv:.3f} at gamma={prof_best_gamma_gv:.4f}, V={prof_best_v_gv:.3f}"
        )

    prof_best_gamma_gc = prof_best_c_gc = None
    prof_best_val_gc_profile = None
    if np.isfinite(profile_gc_plot).any():
        idx_prof_gc = np.nanargmax(profile_gc_plot)
        prof_c_gc_idx, prof_gamma_gc_idx = np.unravel_index(idx_prof_gc, profile_gc_plot.shape)
        prof_best_gamma_gc = gamma_grid[prof_gamma_gc_idx]
        prof_best_c_gc = c_grid[prof_c_gc_idx]
        prof_best_val_gc_profile = profile_gc_plot[prof_c_gc_idx, prof_gamma_gc_idx]
        print(
            f"Profile -GMM over (gamma,c): {prof_best_val_gc_profile:.3f} at gamma={prof_best_gamma_gc:.4f}, c={prof_best_c_gc:.6f}"
        )

    V_mesh, C_mesh = np.meshgrid(v_grid, c_grid)
    Gamma_mesh_gv, V_mesh_gv = np.meshgrid(gamma_grid, v_grid)
    Gamma_mesh_gc, C_mesh_gc = np.meshgrid(gamma_grid, c_grid)

    def percentile_levels(Z, lower=5, upper=99, num=40):
        finite = Z[np.isfinite(Z)]
        if finite.size == 0:
            return None
        lo = np.percentile(finite, lower)
        hi = np.percentile(finite, upper)
        if np.isclose(lo, hi):
            hi = lo + 1e-6
        return np.linspace(lo, hi, num)

    def plot_surface(
        X,
        Y,
        Z,
        xlabel,
        ylabel,
        title,
        filename,
        baseline=None,
        extras=None,
        cmap='viridis',
        log_y=False,
        ax=None,
        save=True,
    ):
        levels = percentile_levels(Z)
        if levels is None:
            if ax is not None:
                ax.set_axis_off()
            if save and filename is not None:
                print(f"Skipping plot {filename}: insufficient finite data")
            return
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_fig = True
        else:
            fig = ax.figure
        contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.3, alpha=0.5)
        fig.colorbar(contour, ax=ax, label=title)
        if baseline is not None and np.all(np.isfinite(baseline)):
            ax.scatter(*baseline, color='red', marker='x', s=160, linewidths=2, zorder=6, label='Baseline')
        extras = [item for item in (extras or []) if item is not None]
        for label, point, style in extras:
            if point is None or not np.all(np.isfinite(point)):
                continue
            ax.scatter(
                point[0],
                point[1],
                facecolors=style.get('face', 'white'),
                edgecolors=style.get('edge', 'black'),
                marker=style.get('marker', 'o'),
                s=style.get('s', 220),
                linewidths=style.get('lw', 2),
                zorder=style.get('z', 7),
                label=label,
            )
        if log_y:
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(style='plain', axis='y', useMathText=False)
        ax.ticklabel_format(style='plain', axis='x', useMathText=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')
        if created_fig:
            fig.tight_layout()
        if save and filename is not None:
            path = os.path.join(args.out_dir, filename)
            fig.savefig(path, dpi=200)
            if created_fig:
                print(f"Saved contour plot to {path}")
        if created_fig:
            plt.close(fig)

    def plot_combined_panels(rows, out_dir: str, filename: str) -> None:
        if not rows:
            return
        n_rows = len(rows)
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        column_labels = ['Fixed other parameter', 'Profiled over other parameter']
        for col, label in enumerate(column_labels):
            fig.text(0.25 + 0.5 * col, 0.99, label, ha='center', va='top', fontsize=14, fontweight='bold')
        for row_idx, (fixed_plot, prof_plot) in enumerate(rows):
            for col_idx, info in enumerate((fixed_plot, prof_plot)):
                info_local = dict(info)
                info_local.setdefault('extras', info.get('extras'))
                info_local.update({'ax': axes[row_idx, col_idx], 'save': False})
                plot_surface(**info_local)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        combined_path = os.path.join(out_dir, filename)
        fig.savefig(combined_path, dpi=200)
        plt.close(fig)
        print(f"Saved combined GMM panels to {combined_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    extras_vc = [('Grid max', (best_V, best_c), {})]
    if prof_best_v is not None:
        extras_vc.append(('Profile max', (prof_best_v, prof_best_c), {'marker': 'D', 'face': 'cyan'}))
    extras_prof_vc = []
    if prof_best_v is not None:
        extras_prof_vc.append(('Profile max', (prof_best_v, prof_best_c), {'marker': 'D', 'face': 'cyan'}))

    extras_gv = [('Grid max', (best_gamma_gv, best_v_gv), {})]
    extras_prof_gv = []
    if prof_best_gamma_gv is not None:
        extras_prof_gv.append(('Profile max', (prof_best_gamma_gv, prof_best_v_gv), {'marker': 'D', 'face': 'cyan'}))

    extras_gc = [('Grid max', (best_gamma_gc, best_c_gc), {})]
    extras_prof_gc = []
    if prof_best_gamma_gc is not None:
        extras_prof_gc.append(('Profile max', (prof_best_gamma_gc, prof_best_c_gc), {'marker': 'D', 'face': 'cyan'}))

    rows = []

    vc_fixed = {
        'X': V_mesh,
        'Y': C_mesh,
        'Z': neg_obj_vc,
        'xlabel': 'V',
        'ylabel': 'Cutoff c',
        'title': f'-GMM objective (γ fixed at {gamma_fixed:.4f})',
        'filename': args.plot_filename,
        'baseline': (V_center, c_center),
        'extras': extras_vc,
        'log_y': True,
    }
    plot_surface(**vc_fixed)

    vc_profile = {
        'X': V_mesh,
        'Y': C_mesh,
        'Z': profile_vc_plot,
        'xlabel': 'V',
        'ylabel': 'Cutoff c',
        'title': 'Profile (-GMM objective) over (V, c) (max over γ)',
        'filename': 'profile_neg_gmm_v_c.png',
        'baseline': (V_center, c_center),
        'extras': extras_prof_vc,
        'cmap': 'magma',
        'log_y': True,
    }
    plot_surface(**vc_profile)
    rows.append((vc_fixed, vc_profile))

    gv_fixed = {
        'X': Gamma_mesh_gv,
        'Y': V_mesh_gv,
        'Z': neg_obj_gamma_v,
        'xlabel': 'γ',
        'ylabel': 'V',
        'title': '-GMM objective over (γ, V)',
        'filename': 'neg_gmm_gamma_v.png',
        'baseline': (gamma_fixed, V_center),
        'extras': extras_gv,
    }
    plot_surface(**gv_fixed)

    gv_profile = {
        'X': Gamma_mesh_gv,
        'Y': V_mesh_gv,
        'Z': profile_gv_plot,
        'xlabel': 'γ',
        'ylabel': 'V',
        'title': 'Profile (-GMM objective) over (γ, V) (max over c)',
        'filename': 'profile_neg_gmm_gamma_v.png',
        'baseline': (gamma_fixed, V_center),
        'extras': extras_prof_gv,
        'cmap': 'magma',
    }
    plot_surface(**gv_profile)
    rows.append((gv_fixed, gv_profile))

    gc_fixed = {
        'X': Gamma_mesh_gc,
        'Y': C_mesh_gc,
        'Z': neg_obj_gamma_c,
        'xlabel': 'γ',
        'ylabel': 'Cutoff c',
        'title': '-GMM objective over (γ, c)',
        'filename': 'neg_gmm_gamma_c.png',
        'baseline': (gamma_fixed, c_center),
        'extras': extras_gc,
        'log_y': True,
    }
    plot_surface(**gc_fixed)

    gc_profile = {
        'X': Gamma_mesh_gc,
        'Y': C_mesh_gc,
        'Z': profile_gc_plot,
        'xlabel': 'γ',
        'ylabel': 'Cutoff c',
        'title': 'Profile (-GMM objective) over (γ, c) (max over V)',
        'filename': 'profile_neg_gmm_gamma_c.png',
        'baseline': (gamma_fixed, c_center),
        'extras': extras_prof_gc,
        'cmap': 'magma',
        'log_y': True,
    }
    plot_surface(**gc_profile)
    rows.append((gc_fixed, gc_profile))

    plot_combined_panels(rows, args.out_dir, 'gmm_objective_panels.png')

    if args.save_grid:
        grid_path = os.path.join(args.out_dir, 'gmm_objective_vc_grid.npz')
        np.savez(
            grid_path,
            v_grid=v_grid,
            c_grid=c_grid,
            gamma_grid=gamma_grid,
            neg_obj_vc=neg_obj_vc,
            neg_obj_gamma_v=neg_obj_gamma_v,
            neg_obj_gamma_c=neg_obj_gamma_c,
            profile_neg_vc=profile_neg_vc,
            profile_neg_gamma_v=profile_neg_gamma_v,
            profile_neg_gamma_c=profile_neg_gamma_c,
            gamma=gamma_fixed,
            baseline_V=V_center,
            baseline_c=c_center,
        )
        print(f"Saved grid data to {grid_path}")


if __name__ == '__main__':
    main()
