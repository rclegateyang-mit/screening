#!/usr/bin/env python3
"""Plot the log-likelihood surface over (V, c) for J=1 with γ fixed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .jax_model import enable_x64, compute_choice_probabilities_jax
    from .helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
        compute_cutoffs,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from jax_model import enable_x64, compute_choice_probabilities_jax
    from helpers import (
        read_parameters,
        read_firms_data,
        read_workers_data,
        compute_worker_firm_distances,
        compute_cutoffs,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent.parent
    output_dir = root / "output"

    parser = argparse.ArgumentParser(
        description="Plot log-likelihood surface over (V, c) for J=1 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--workers_path', type=str, default=str(output_dir / 'workers_dataset.csv'))
    parser.add_argument('--firms_path', type=str, default=str(output_dir / 'equilibrium_firms.csv'))
    parser.add_argument('--params_path', type=str, default=str(output_dir / 'parameters_effective.csv'))
    parser.add_argument('--truth_path', type=str, default=None,
                        help='Optional JSON with true theta (fields gamma, V, A).')
    parser.add_argument('--out_dir', type=str, default=str(output_dir))
    parser.add_argument('--plot_filename', type=str, default='likelihood_v_c.png')
    parser.add_argument('--save_grid', action='store_true', help='Persist the evaluated grid to NPZ.')

    parser.add_argument('--v_min', type=float, default=None)
    parser.add_argument('--v_max', type=float, default=None)
    parser.add_argument('--v_steps', type=int, default=30)
    parser.add_argument('--c_min', type=float, default=None)
    parser.add_argument('--c_max', type=float, default=None)
    parser.add_argument('--c_steps', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=None,
                        help='Override gamma; defaults to params or truth if provided.')
    parser.add_argument('--gamma_min', type=float, default=0.0)
    parser.add_argument('--gamma_max', type=float, default=0.15)
    parser.add_argument('--gamma_steps', type=int, default=30)
    parser.add_argument('--threads', type=int, default=None,
                        help='Optional CPU thread pool size (XLA).')

    return parser.parse_args()


def log_likelihood(theta: jnp.ndarray,
                   X: jnp.ndarray,
                   choice_idx: jnp.ndarray,
                   aux: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    P = compute_choice_probabilities_jax(theta, X, aux)
    probs = jnp.take_along_axis(P, choice_idx[:, None], axis=1).squeeze(axis=1)
    safe = jnp.clip(probs, 1e-300, 1.0)
    return jnp.sum(jnp.log(safe))


def main() -> None:
    args = parse_args()

    if args.threads is not None:
        print("Set CPU threads by exporting before invocation:")
        print(f'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_cpu_thread_pool_size={int(args.threads)}"')

    enable_x64()

    params = read_parameters(args.params_path)
    firm_ids, w, Y, A_nat_data, xi, loc_firms, c_data = read_firms_data(args.firms_path)
    x_skill, ell_x, ell_y, chosen_firm = read_workers_data(args.workers_path)

    J = firm_ids.size
    if J != 1:
        raise ValueError(f"This visualisation only supports J=1; got J={J}.")

    w = np.asarray(w, dtype=float)
    xi = np.asarray(xi, dtype=float)
    c_data = np.asarray(c_data, dtype=float)
    A_nat_data = np.asarray(A_nat_data, dtype=float)

    beta = float(params.get('beta', 0.5))
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
    c0 = float(c_data[0]) if c_data.size else float(compute_cutoffs(w, Y, A_nat_data, beta)[0])
    A0 = float(A_nat_data[0])

    D_nat = compute_worker_firm_distances(ell_x, ell_y, loc_firms)

    aux: Dict = {
        'D_nat': jnp.asarray(D_nat, dtype=jnp.float64),
        'w': jnp.asarray(w, dtype=jnp.float64),
        'Y': jnp.asarray(Y, dtype=jnp.float64),
        'beta': beta,
        'phi': phi,
        'mu_a': mu_a,
        'sigma_a': sigma_a,
        'firm_ids': jnp.asarray(firm_ids, dtype=jnp.int32),
    }

    X = jnp.asarray(x_skill, dtype=jnp.float64)
    choice_idx = jnp.asarray(chosen_firm.astype(np.int32))

    log_likelihood_theta = jax.jit(log_likelihood)

    v_min = args.v_min if args.v_min is not None else float(V0_nat[0] - 2.0)
    v_max = args.v_max if args.v_max is not None else float(V0_nat[0] + 2.0)
    if v_min >= v_max:
        raise ValueError("Require v_min < v_max")

    if args.c_min is not None:
        c_min = max(args.c_min, 1e-6)
    else:
        c_min = c0 * 0.2

    if args.c_max is not None:
        c_max = max(args.c_max, c_min * 1.01)
    else:
        c_max = c0 * 5.0

    if c_min <= 0 or c_min >= c_max:
        raise ValueError("Require 0 < c_min < c_max")

    gamma_min = float(args.gamma_min)
    gamma_max = float(args.gamma_max)
    gamma_steps = int(args.gamma_steps)
    if gamma_min < 0:
        gamma_min = 0.0
    if gamma_max <= gamma_min:
        raise ValueError("Require gamma_min < gamma_max")

    v_grid = np.linspace(v_min, v_max, int(args.v_steps))
    c_grid = np.geomspace(c_min, c_max, int(args.c_steps))
    gamma_grid = np.linspace(gamma_min, gamma_max, gamma_steps)

    loglik = np.full((c_grid.size, v_grid.size), np.nan, dtype=float)
    loglik_gamma_v = np.full((v_grid.size, gamma_grid.size), np.nan, dtype=float)
    loglik_gamma_c = np.full((c_grid.size, gamma_grid.size), np.nan, dtype=float)

    profile_loglik_vc = np.full_like(loglik, -np.inf)
    profile_loglik_gamma_v = np.full_like(loglik_gamma_v, -np.inf)
    profile_loglik_gamma_c = np.full_like(loglik_gamma_c, -np.inf)

    for i_c, c_val in enumerate(c_grid):
        theta_base = jnp.asarray([gamma_fixed, v_grid[0], c_val], dtype=jnp.float64)
        _ = log_likelihood_theta(theta_base, X, choice_idx, aux)

        for j_v, V_val in enumerate(v_grid):
            theta = jnp.asarray([gamma_fixed, V_val, c_val], dtype=jnp.float64)
            try:
                ll_val = float(log_likelihood_theta(theta, X, choice_idx, aux))
            except FloatingPointError:
                ll_val = float('-inf')
            if not np.isfinite(ll_val):
                ll_val = float('-inf')
            loglik[i_c, j_v] = ll_val

            best_ll = -np.inf
            for gamma_val in gamma_grid:
                theta_prof = jnp.asarray([gamma_val, V_val, c_val], dtype=jnp.float64)
                try:
                    ll_prof = float(log_likelihood_theta(theta_prof, X, choice_idx, aux))
                except FloatingPointError:
                    ll_prof = float('-inf')
                if np.isfinite(ll_prof) and ll_prof > best_ll:
                    best_ll = ll_prof
            profile_loglik_vc[i_c, j_v] = best_ll

    c_for_gamma_v = c0

    for j_gamma, gamma_val in enumerate(gamma_grid):
        theta_base = jnp.asarray([gamma_val, v_grid[0], c_for_gamma_v], dtype=jnp.float64)
        _ = log_likelihood_theta(theta_base, X, choice_idx, aux)
        for i_v, V_val in enumerate(v_grid):
            theta = jnp.asarray([gamma_val, V_val, c_for_gamma_v], dtype=jnp.float64)
            try:
                ll_val = float(log_likelihood_theta(theta, X, choice_idx, aux))
            except FloatingPointError:
                ll_val = float('-inf')
            if not np.isfinite(ll_val):
                ll_val = float('-inf')
            loglik_gamma_v[i_v, j_gamma] = ll_val

            best_ll = -np.inf
            for c_val in c_grid:
                theta_prof = jnp.asarray([gamma_val, V_val, c_val], dtype=jnp.float64)
                try:
                    ll_prof = float(log_likelihood_theta(theta_prof, X, choice_idx, aux))
                except FloatingPointError:
                    ll_prof = float('-inf')
                if np.isfinite(ll_prof) and ll_prof > best_ll:
                    best_ll = ll_prof
            profile_loglik_gamma_v[i_v, j_gamma] = best_ll

    V_for_gamma_c = float(V0_nat[0])
    for i_c, c_val in enumerate(c_grid):
        theta_base = jnp.asarray([gamma_grid[0], V_for_gamma_c, c_val], dtype=jnp.float64)
        _ = log_likelihood_theta(theta_base, X, choice_idx, aux)
        for j_gamma, gamma_val in enumerate(gamma_grid):
            theta = jnp.asarray([gamma_val, V_for_gamma_c, c_val], dtype=jnp.float64)
            try:
                ll_val = float(log_likelihood_theta(theta, X, choice_idx, aux))
            except FloatingPointError:
                ll_val = float('-inf')
            if not np.isfinite(ll_val):
                ll_val = float('-inf')
            loglik_gamma_c[i_c, j_gamma] = ll_val

            best_ll = -np.inf
            for V_val in v_grid:
                theta_prof = jnp.asarray([gamma_val, V_val, c_val], dtype=jnp.float64)
                try:
                    ll_prof = float(log_likelihood_theta(theta_prof, X, choice_idx, aux))
                except FloatingPointError:
                    ll_prof = float('-inf')
                if np.isfinite(ll_prof) and ll_prof > best_ll:
                    best_ll = ll_prof
            profile_loglik_gamma_c[i_c, j_gamma] = best_ll

    finite_mask = np.isfinite(loglik)
    if not np.any(finite_mask):
        raise RuntimeError("Log-likelihood evaluation failed for all grid points.")

    max_idx = np.nanargmax(loglik)
    max_c_idx, max_v_idx = np.unravel_index(max_idx, loglik.shape)
    best_V = v_grid[max_v_idx]
    best_c = c_grid[max_c_idx]
    best_ll = loglik[max_c_idx, max_v_idx]

    finite_gv = np.isfinite(loglik_gamma_v)
    if not np.any(finite_gv):
        raise RuntimeError("Gamma-V log-likelihood evaluation failed for all grid points.")
    gv_idx = np.nanargmax(loglik_gamma_v)
    best_v_idx, best_gamma_idx = np.unravel_index(gv_idx, loglik_gamma_v.shape)
    best_gamma_gv = gamma_grid[best_gamma_idx]
    best_v_gv = v_grid[best_v_idx]
    best_ll_gv = loglik_gamma_v[best_v_idx, best_gamma_idx]

    finite_gc = np.isfinite(loglik_gamma_c)
    if not np.any(finite_gc):
        raise RuntimeError("Gamma-c log-likelihood evaluation failed for all grid points.")
    gc_idx = np.nanargmax(loglik_gamma_c)
    best_c_gc_idx, best_gamma_gc_idx = np.unravel_index(gc_idx, loglik_gamma_c.shape)
    best_gamma_gc = gamma_grid[best_gamma_gc_idx]
    best_c_gc = c_grid[best_c_gc_idx]
    best_ll_gc = loglik_gamma_c[best_c_gc_idx, best_gamma_gc_idx]

    print(f"Max log-likelihood on grid: {best_ll:.3f} at V={best_V:.3f}, c={best_c:.6f}")
    print(f"Max log-likelihood on (gamma,V) grid: {best_ll_gv:.3f} at gamma={best_gamma_gv:.4f}, V={best_v_gv:.3f}")
    print(f"Max log-likelihood on (gamma,c) grid: {best_ll_gc:.3f} at gamma={best_gamma_gc:.4f}, c={best_c_gc:.6f}")

    V0 = float(V0_nat[0])

    V_mesh, C_mesh = np.meshgrid(v_grid, c_grid)
    Gamma_mesh_gv, V_mesh_gv = np.meshgrid(gamma_grid, v_grid)
    Gamma_mesh_gc, C_mesh_gc = np.meshgrid(gamma_grid, c_grid)

    profile_vc_plot = np.where(np.isfinite(profile_loglik_vc), profile_loglik_vc, np.nan)
    profile_gv_plot = np.where(np.isfinite(profile_loglik_gamma_v), profile_loglik_gamma_v, np.nan)
    profile_gc_plot = np.where(np.isfinite(profile_loglik_gamma_c), profile_loglik_gamma_c, np.nan)

    if np.any(np.isfinite(profile_loglik_vc)):
        prof_vc_idx = np.nanargmax(profile_vc_plot)
        prof_c_idx, prof_v_idx = np.unravel_index(prof_vc_idx, profile_vc_plot.shape)
        prof_best_c = c_grid[prof_c_idx]
        prof_best_v = v_grid[prof_v_idx]
        prof_best_ll_vc = profile_vc_plot[prof_c_idx, prof_v_idx]
        print(f"Profile log-likelihood (V,c) max: {prof_best_ll_vc:.3f} at V={prof_best_v:.3f}, c={prof_best_c:.6f}")
    else:
        prof_best_v = None
        prof_best_c = None

    if np.any(np.isfinite(profile_loglik_gamma_v)):
        prof_gv_idx = np.nanargmax(profile_gv_plot)
        prof_v_gv_idx, prof_gamma_gv_idx = np.unravel_index(prof_gv_idx, profile_gv_plot.shape)
        prof_best_gamma_gv = gamma_grid[prof_gamma_gv_idx]
        prof_best_v_gv = v_grid[prof_v_gv_idx]
        prof_best_ll_gv = profile_gv_plot[prof_v_gv_idx, prof_gamma_gv_idx]
        print(f"Profile log-likelihood (gamma,V) max: {prof_best_ll_gv:.3f} at gamma={prof_best_gamma_gv:.4f}, V={prof_best_v_gv:.3f}")
    else:
        prof_best_gamma_gv = None
        prof_best_v_gv = None

    if np.any(np.isfinite(profile_loglik_gamma_c)):
        prof_gc_idx = np.nanargmax(profile_gc_plot)
        prof_c_gc_idx, prof_gamma_gc_idx = np.unravel_index(prof_gc_idx, profile_gc_plot.shape)
        prof_best_gamma_gc = gamma_grid[prof_gamma_gc_idx]
        prof_best_c_gc = c_grid[prof_c_gc_idx]
        prof_best_ll_gc = profile_gc_plot[prof_c_gc_idx, prof_gamma_gc_idx]
        print(f"Profile log-likelihood (gamma,c) max: {prof_best_ll_gc:.3f} at gamma={prof_best_gamma_gc:.4f}, c={prof_best_c_gc:.6f}")
    else:
        prof_best_gamma_gc = None
        prof_best_c_gc = None

    levels = 80

    def format_axes(ax, log_y=False):
        if log_y:
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(style='plain', axis='y', useMathText=False)
        ax.ticklabel_format(style='plain', axis='x', useMathText=False)

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
        *,
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
        if extras:
            for label, point, style in extras:
                if point is None or not np.all(np.isfinite(point)):
                    continue
                ax.scatter(
                    point[0], point[1],
                    facecolors=style.get('face', 'white'),
                    edgecolors=style.get('edge', 'black'),
                    marker=style.get('marker', 'o'),
                    s=style.get('s', 220),
                    linewidths=style.get('lw', 2),
                    zorder=style.get('z', 7),
                    label=label,
                )
        format_axes(ax, log_y=log_y)
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
        print(f"Saved combined likelihood panels to {combined_path}")

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
        'Z': loglik,
        'xlabel': 'V',
        'ylabel': 'Cutoff c',
        'title': f'Log-likelihood (γ fixed at {gamma_fixed:.4f})',
        'filename': args.plot_filename,
        'baseline': (V0, c0),
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
        'title': 'Profile log-likelihood (max over γ)',
        'filename': 'profile_loglik_V_c.png',
        'baseline': (V0, c0),
        'extras': extras_prof_vc,
        'cmap': 'magma',
        'log_y': True,
    }
    plot_surface(**vc_profile)
    rows.append((vc_fixed, vc_profile))

    gv_fixed = {
        'X': Gamma_mesh_gv,
        'Y': V_mesh_gv,
        'Z': loglik_gamma_v,
        'xlabel': 'γ',
        'ylabel': 'V',
        'title': 'Log-likelihood over (γ, V)',
        'filename': 'likelihood_gamma_V.png',
        'baseline': (gamma_fixed, V0),
        'extras': extras_gv,
    }
    plot_surface(**gv_fixed)

    gv_profile = {
        'X': Gamma_mesh_gv,
        'Y': V_mesh_gv,
        'Z': profile_gv_plot,
        'xlabel': 'γ',
        'ylabel': 'V',
        'title': 'Profile log-likelihood (max over c)',
        'filename': 'profile_loglik_gamma_V.png',
        'baseline': (gamma_fixed, V0),
        'extras': extras_prof_gv,
        'cmap': 'magma',
    }
    plot_surface(**gv_profile)
    rows.append((gv_fixed, gv_profile))

    gc_fixed = {
        'X': Gamma_mesh_gc,
        'Y': C_mesh_gc,
        'Z': loglik_gamma_c,
        'xlabel': 'γ',
        'ylabel': 'Cutoff c',
        'title': 'Log-likelihood over (γ, c)',
        'filename': 'likelihood_gamma_c.png',
        'baseline': (gamma_fixed, c0),
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
        'title': 'Profile log-likelihood (max over V)',
        'filename': 'profile_loglik_gamma_c.png',
        'baseline': (gamma_fixed, c0),
        'extras': extras_prof_gc,
        'cmap': 'magma',
        'log_y': True,
    }
    plot_surface(**gc_profile)
    rows.append((gc_fixed, gc_profile))

    plot_combined_panels(rows, args.out_dir, 'likelihood_panels.png')

    if args.save_grid:
        grid_path = os.path.join(args.out_dir, 'likelihood_vc_grid.npz')
        np.savez(
            grid_path,
            v_grid=v_grid,
            c_grid=c_grid,
            gamma_grid=gamma_grid,
            loglik=loglik,
            loglik_gamma_v=loglik_gamma_v,
            loglik_gamma_c=loglik_gamma_c,
            profile_loglik_vc=profile_loglik_vc,
            profile_loglik_gamma_v=profile_loglik_gamma_v,
            profile_loglik_gamma_c=profile_loglik_gamma_c,
            gamma=gamma_fixed,
            baseline_V=V0,
            baseline_c=c0,
            baseline_gamma=gamma_fixed,
        )
        print(f"Saved grid data to {grid_path}")


if __name__ == '__main__':
    main()
