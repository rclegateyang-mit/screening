#!/usr/bin/env python3
"""
Data Environment Item 4: Implied Wage Markdowns.

This script takes equilibrium firm outcomes, parameter values, and (optionally)
the simulated worker dataset to compute wage markdowns implied by the model.
It reports firm-level markdowns evaluated at firm-average skill and the market
average skill, and, when worker data are available, worker-level markdowns
evaluated at each worker's realised skill.

Outputs:
  - firm_markdowns.csv
  - worker_markdowns.csv (if worker data supplied)
  - worker_markdowns_vs_skill.png
  - worker_markdowns_by_firm.png
  - firm_markdowns_vs_average_skill.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from .. import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_MARKDOWNS
except ImportError:  # pragma: no cover - direct script execution
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
    from code import get_data_subdir, get_output_subdir, DATA_RAW, DATA_CLEAN, DATA_BUILD, OUTPUT_MARKDOWNS  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_parameters_csv(path: Path) -> Dict[str, float]:
    """Read key scalar parameters from a long-format CSV."""
    df = pd.read_csv(path)
    if not {"parameter", "value"}.issubset(df.columns):
        raise ValueError(f"Parameter file {path} must contain 'parameter' and 'value' columns.")
    params: Dict[str, float] = {}
    for _, row in df.iterrows():
        key = str(row["parameter"])
        try:
            params[key] = float(row["value"])
        except (TypeError, ValueError):
            continue
    required = {"eta", "alpha", "mu_s"}
    missing = required - params.keys()
    if missing:
        raise ValueError(f"Missing required parameters {missing} in {path}.")
    return params


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_firm_markdowns(equilibrium_df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    """Compute firm-level markdowns evaluated at Q_j and market-average skill."""
    required_cols = {"firm_id", "w", "L", "Q", "A"}
    missing = required_cols - set(equilibrium_df.columns)
    if missing:
        raise ValueError(f"Equilibrium data missing required columns: {missing}")

    alpha = params["alpha"]
    mu_s = params["mu_s"]
    N_workers = float(params.get("N_workers", 1.0))
    if not np.isfinite(N_workers) or N_workers <= 0:
        raise ValueError(f"N_workers must be positive; got {N_workers}")

    eps = 1e-12
    L_share = np.maximum(equilibrium_df["L"].to_numpy(dtype=float), eps)
    L_counts = np.maximum(L_share * N_workers, eps)
    Q = np.maximum(equilibrium_df["Q"].to_numpy(dtype=float), eps)
    w = equilibrium_df["w"].to_numpy(dtype=float)
    A = equilibrium_df["A"].to_numpy(dtype=float)
    if "qbar" in equilibrium_df.columns:
        qbar_vals = equilibrium_df["qbar"].to_numpy(dtype=float)
    else:
        qbar_vals = np.full_like(L_share, np.nan)
    firm_ids = equilibrium_df["firm_id"].to_numpy(dtype=int)

    # MRPL_j(s) = s * (1 - alpha) * A_j / (L_j * Q_j)^alpha
    LS_prod = np.maximum(L_counts * Q, eps)
    mrpl_scale = (1.0 - alpha) * A / np.power(LS_prod, alpha)

    mrpl_at_Qj = Q * mrpl_scale
    markdown_Qj = np.where(mrpl_at_Qj > eps, (mrpl_at_Qj - w) / mrpl_at_Qj, np.nan)

    Q_market = float(mu_s)
    mrpl_at_Qbar = Q_market * mrpl_scale
    markdown_Qbar = np.where(mrpl_at_Qbar > eps, (mrpl_at_Qbar - w) / mrpl_at_Qbar, np.nan)

    df = pd.DataFrame(
        {
            "firm_id": firm_ids,
            "A": A,
            "w": w,
            "L": L_share,
            "L_share": L_share,
            "L_workers": L_counts,
            "Q_j": Q,
            "qbar": qbar_vals,
            "markdown_at_Qj": markdown_Qj,
            "markdown_at_Q_market": markdown_Qbar,
        }
    )
    if qbar_vals is not None and {"elastic_L_w_w", "elastic_S_w_w"}.issubset(equilibrium_df.columns):
        eps_elastic = 1e-12
        elastic_L = equilibrium_df["elastic_L_w_w"].to_numpy(dtype=float)
        elastic_S = equilibrium_df["elastic_S_w_w"].to_numpy(dtype=float)
        denom = np.maximum(1.0 + elastic_L, eps_elastic)
        naive_markup = 1.0 / denom
        screening_markup = (1.0 - elastic_S) / denom
        df["elastic_L_w_w"] = elastic_L
        df["elastic_S_w_w"] = elastic_S
        df["markup_naive"] = naive_markup
        df["markup_screening"] = screening_markup
    if {"elastic_L_A_A", "elastic_w_A_A"}.issubset(equilibrium_df.columns):
        eps_pass = 1e-12
        elastic_L_A = equilibrium_df["elastic_L_A_A"].to_numpy(dtype=float)
        elastic_w_A = equilibrium_df["elastic_w_A_A"].to_numpy(dtype=float)
        denom_pass = np.maximum(1.0 + elastic_L_A / np.maximum(elastic_w_A, eps_pass), eps_pass)
        df["elastic_L_A_A"] = elastic_L_A
        df["elastic_w_A_A"] = elastic_w_A
        df["markup_naive_pass_through"] = 1.0 / denom_pass
    df.attrs["mrpl_scale"] = mrpl_scale
    return df


def compute_worker_markdowns(
    worker_df: pd.DataFrame,
    firm_markdowns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute worker-level markdowns using realised skills."""
    required_cols = {"s_skill", "chosen_firm"}
    if not required_cols.issubset(worker_df.columns):
        raise ValueError("Worker dataset must contain columns 's_skill' and 'chosen_firm'.")

    firm_ids = firm_markdowns["firm_id"].to_numpy(dtype=int)
    mrpl_scale = firm_markdowns.attrs["mrpl_scale"]
    wages = firm_markdowns["w"].to_numpy(dtype=float)
    firm_avg_skill = firm_markdowns["Q_j"].to_numpy(dtype=float)

    if {"elastic_L_w_w", "elastic_S_w_w"}.issubset(firm_markdowns.columns):
        elastic_L = firm_markdowns["elastic_L_w_w"].to_numpy(dtype=float)
        elastic_S = firm_markdowns["elastic_S_w_w"].to_numpy(dtype=float)
        eps_elastic = 1e-12
        denom = np.maximum(1.0 + elastic_L, eps_elastic)
        screening_markdowns = (1.0 - elastic_S) / denom
    elif "markup_screening" in firm_markdowns.columns:
        screening_markdowns = firm_markdowns["markup_screening"].to_numpy(dtype=float)
    else:
        raise ValueError("Firm markdown data must include elasticities to compute worker markdowns.")

    firm_index = {int(fid): idx for idx, fid in enumerate(firm_ids.tolist())}
    firm_col = "chosen_firm"

    # If reindexed workers are stored separately, fall back gracefully
    worker_firm_vals = worker_df[firm_col].dropna().astype(int)
    unique_firms = set(worker_firm_vals[worker_firm_vals > 0].unique().tolist())
    if not unique_firms.issubset(firm_index.keys()):
        fallback_col = "chosen_firm_original_id"
        if fallback_col in worker_df.columns:
            firm_col = fallback_col
            worker_firm_vals = worker_df[firm_col].dropna().astype(int)
            unique_firms = set(worker_firm_vals[worker_firm_vals > 0].unique().tolist())
        if not unique_firms.issubset(firm_index.keys()):
            raise ValueError("Worker firm identifiers do not match equilibrium firm IDs.")

    df = worker_df.copy()
    df["firm_index"] = df[firm_col].map(firm_index)
    valid = df["firm_index"].notna() & (df[firm_col] > 0)

    idx = df.loc[valid, "firm_index"].astype(int).to_numpy()
    s_skill = df.loc[valid, "s_skill"].to_numpy(dtype=float)
    mrpl_worker = s_skill * mrpl_scale[idx]
    wage_worker = wages[idx]

    df["mrpl_worker"] = np.nan
    df["wage_worker"] = np.nan
    df["markdown_worker"] = np.nan
    df["chosen_firm_avg_skill"] = np.nan
    df["r"] = np.nan
    df["screening_markdown_e"] = np.nan
    df.loc[valid, "mrpl_worker"] = mrpl_worker
    df.loc[valid, "wage_worker"] = wage_worker

    if idx.size > 0:
        chosen_avg_skill = firm_avg_skill[idx]
        df.loc[valid, "chosen_firm_avg_skill"] = chosen_avg_skill

        eps_skill = 1e-12
        r_vals = np.full_like(s_skill, np.nan)
        nonzero_skill = np.abs(chosen_avg_skill) > eps_skill
        r_vals[nonzero_skill] = s_skill[nonzero_skill] / chosen_avg_skill[nonzero_skill]
        df.loc[valid, "r"] = r_vals

        e_vals = screening_markdowns[idx]
        df.loc[valid, "screening_markdown_e"] = e_vals

        eps_r = 1e-12
        markdown_worker = np.full_like(s_skill, np.nan)
        valid_r = np.abs(r_vals) > eps_r
        markdown_worker[valid_r] = 1.0 - (1.0 - e_vals[valid_r]) / r_vals[valid_r]
        df.loc[valid, "markdown_worker"] = markdown_worker

    df.attrs["valid_mask"] = valid
    return df


def compute_noscreening_markdowns(
    firm_df_noscreen: pd.DataFrame,
    params: Dict[str, float],
) -> pd.DataFrame | None:
    if "firm_id" not in firm_df_noscreen.columns:
        return None
    has_L = "elastic_L_w_noscreening" in firm_df_noscreen.columns
    has_S = "elastic_S_w_noscreening" in firm_df_noscreen.columns
    has_L_fd = "elastic_L_w_noscreening_fd" in firm_df_noscreen.columns
    has_S_fd = "elastic_S_w_noscreening_fd" in firm_df_noscreen.columns
    if not ((has_L or has_L_fd) and (has_S or has_S_fd)):
        return None

    eps = 1e-12
    elastic_L = (
        firm_df_noscreen["elastic_L_w_noscreening_fd"].to_numpy(dtype=float)
        if has_L_fd
        else firm_df_noscreen["elastic_L_w_noscreening"].to_numpy(dtype=float)
    )
    elastic_S = (
        firm_df_noscreen["elastic_S_w_noscreening_fd"].to_numpy(dtype=float)
        if has_S_fd
        else firm_df_noscreen["elastic_S_w_noscreening"].to_numpy(dtype=float)
    )
    denom = np.maximum(1.0 + elastic_L, eps)
    df = pd.DataFrame(
        {
            "firm_id": firm_df_noscreen["firm_id"].to_numpy(dtype=int),
            "elastic_L_w_noscreening": elastic_L,
            "elastic_S_w_noscreening": elastic_S,
            "markup_no_screening_labor_only": 1.0 / denom,
            "markup_no_screening": (1.0 - elastic_S) / denom,
        }
    )
    if {"elastic_L_A_A_noscreening", "elastic_w_A_A_noscreening"}.issubset(
        firm_df_noscreen.columns
    ):
        elastic_L_A = firm_df_noscreen["elastic_L_A_A_noscreening"].to_numpy(dtype=float)
        elastic_w_A = firm_df_noscreen["elastic_w_A_A_noscreening"].to_numpy(dtype=float)
        denom_pass = np.maximum(1.0 + elastic_L_A / np.maximum(elastic_w_A, eps), eps)
        df["markup_no_screening_pass_through"] = 1.0 / denom_pass
    return df


def plot_worker_markdowns(worker_df: pd.DataFrame, out_path: Path) -> None:
    """Scatter markdowns vs worker skill."""
    valid = worker_df.attrs.get("valid_mask")
    if valid is None:
        valid = worker_df["markdown_worker"].notna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        worker_df.loc[valid, "s_skill"],
        worker_df.loc[valid, "markdown_worker"],
        s=14,
        alpha=0.35,
        edgecolor="none",
    )
    ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("Worker skill $s_i$")
    ax.set_ylabel(r"Markdown $1 - (1-e_j)/r_{ij}$")
    ax.set_title("Worker-Level Wage Markdown vs Skill")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_worker_wages_vs_skill(worker_df: pd.DataFrame, out_path: Path) -> bool:
    """Scatter realised wages vs worker skill."""
    valid = worker_df.attrs.get("valid_mask")
    if valid is None:
        valid = worker_df["wage_worker"].notna() & worker_df["s_skill"].notna()
    else:
        valid = valid & worker_df["wage_worker"].notna() & worker_df["s_skill"].notna()
    if not np.any(valid):
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        worker_df.loc[valid, "s_skill"],
        worker_df.loc[valid, "wage_worker"],
        s=14,
        alpha=0.35,
        edgecolor="none",
    )
    ax.set_xlabel("Worker skill $s_i$")
    ax.set_ylabel("Wage $w_i$")
    ax.set_title("Worker Wages vs Skill")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def _compute_binned_series(
    worker_df: pd.DataFrame, firm_df: pd.DataFrame, wage_cols: tuple[str, ...] = ("w", "w_noscreening")
) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """Compute decile-binned skill centers, unemployment rates, and mean wages."""
    if "s_skill" not in worker_df.columns:
        return None

    wage_col = next((c for c in wage_cols if c in firm_df.columns), None)
    if wage_col is None:
        return None

    firm_ids = firm_df["firm_id"].to_numpy(dtype=int)
    firm_wages = firm_df[wage_col].to_numpy(dtype=float)
    firm_index = {int(fid): idx for idx, fid in enumerate(firm_ids.tolist())}

    firm_col = "chosen_firm"
    worker_firm_vals = worker_df.get(firm_col)
    if worker_firm_vals is None:
        return None
    worker_firm_vals = worker_firm_vals.dropna().astype(int)
    unique_firms = set(worker_firm_vals[worker_firm_vals > 0].unique().tolist())
    if not unique_firms.issubset(firm_index.keys()):
        fallback_col = "chosen_firm_original_id"
        if fallback_col in worker_df.columns:
            worker_firm_vals = worker_df[fallback_col].dropna().astype(int)
            unique_firms = set(worker_firm_vals[worker_firm_vals > 0].unique().tolist())
            firm_col = fallback_col
    if not unique_firms.issubset(firm_index.keys()):
        return None

    df = worker_df.copy()
    df["firm_index"] = df[firm_col].map(firm_index)
    valid = df["s_skill"].notna()
    df = df.loc[valid].copy()
    if df.empty:
        return None

    df["wage_worker"] = np.nan
    mapped_idx = df["firm_index"].dropna().astype(int).to_numpy()
    df.loc[df["firm_index"].notna(), "wage_worker"] = firm_wages[mapped_idx]
    df["unemployed"] = df[firm_col].isna() | (df[firm_col] <= 0) | df["firm_index"].isna()

    # Decile bins; fall back to fewer bins if there are not enough unique skill values
    deciles = min(10, df["s_skill"].nunique())
    if deciles < 1:
        return None
    try:
        df["skill_bin"] = pd.qcut(df["s_skill"], q=deciles, duplicates="drop")
    except ValueError:
        return None

    grouped = df.groupby("skill_bin")
    bin_centers = grouped["s_skill"].mean()
    unemp_rate = grouped["unemployed"].mean()
    wage_mean = grouped["wage_worker"].mean()
    return bin_centers, unemp_rate, wage_mean


def binscatter_unemp_and_wage_overlay(
    worker_df: pd.DataFrame,
    worker_df_noscreen: pd.DataFrame | None,
    firm_df_screen: pd.DataFrame,
    firm_df_noscreen: pd.DataFrame,
    out_dir: Path,
) -> bool:
    """Overlay binned unemployment/wage vs skill for screening and no-screening equilibria."""
    stats_screen = _compute_binned_series(worker_df, firm_df_screen, ("w",))
    base_worker_df = worker_df_noscreen if worker_df_noscreen is not None else worker_df
    stats_noscreen = _compute_binned_series(base_worker_df, firm_df_noscreen, ("w_noscreening", "w"))
    if stats_screen is None or stats_noscreen is None:
        return False

    bins_s, unemp_s, wage_s = stats_screen
    bins_n, unemp_n, wage_n = stats_noscreen

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(bins_s, unemp_s, s=60, alpha=0.85, label="Unemployment (screening)")
    ax.scatter(bins_n, unemp_n, s=60, alpha=0.6, marker="s", label="Unemployment (no-screening)")
    ax.set_xlabel("Worker skill decile (mean $s_i$)")
    ax.set_ylabel("Unemployment rate")
    ax.set_title("Unemployment vs Skill (deciles)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "worker_unemployment_vs_skill_overlay.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(bins_s, wage_s, s=60, alpha=0.85, label="Wage (screening)")
    ax.scatter(bins_n, wage_n, s=60, alpha=0.6, marker="s", label="Wage (no-screening)")
    ax.set_xlabel("Worker skill decile (mean $s_i$)")
    ax.set_ylabel("Wage")
    ax.set_title("Wage vs Skill (deciles)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "worker_wage_vs_skill_overlay.png", dpi=200)
    plt.close(fig)
    return True


def _compute_avg_distance_to_workers(firm_df: pd.DataFrame, worker_df: pd.DataFrame | None) -> dict[int, float]:
    """Compute average worker–firm distance for each firm_id."""
    if worker_df is None or not {"ell_x", "ell_y"}.issubset(worker_df.columns):
        return {}
    if not {"x", "y", "firm_id"}.issubset(firm_df.columns):
        return {}
    worker_locations = worker_df[["ell_x", "ell_y"]].to_numpy(dtype=float)
    if worker_locations.size == 0:
        return {}
    distances = np.linalg.norm(
        worker_locations[:, None, :] - firm_df[["x", "y"]].to_numpy(dtype=float)[None, :, :],
        axis=2,
    )  # shape (N_workers, J)
    mean_dist = distances.mean(axis=0)
    return {int(fid): float(d) for fid, d in zip(firm_df["firm_id"].astype(int).to_numpy(), mean_dist)}


def plot_screen_vs_noscreen_by_index(
    firm_df_screen: pd.DataFrame,
    firm_df_noscreen: pd.DataFrame,
    out_path: Path,
    worker_df: pd.DataFrame | None = None,
) -> bool:
    """Compare screening vs no-screening L/w/S by firm index."""
    required_screen = {"firm_id", "L", "w", "Q_j"}
    required_noscreen = {"firm_id", "L_noscreening", "w_noscreening", "S_noscreening"}
    if not required_screen.issubset(firm_df_screen.columns) or not required_noscreen.issubset(
        firm_df_noscreen.columns
    ):
        return False

    df = firm_df_screen.merge(
        firm_df_noscreen[["firm_id", "L_noscreening", "w_noscreening", "S_noscreening"]],
        on="firm_id",
        how="inner",
    )
    if df.empty:
        return False

    # Sort by firm_id (natural order)
    df = df.sort_values("firm_id").reset_index(drop=True)

    dist_map = _compute_avg_distance_to_workers(firm_df_screen, worker_df)
    tfp = df["A"] if "A" in df.columns else None

    x = np.arange(df.shape[0])
    labels = []
    for i, row in df.iterrows():
        fid = int(row["firm_id"])
        parts = [f"{fid}"]
        if tfp is not None:
            parts.append(f"A={row['A']:.2f}")
        if fid in dist_map:
            parts.append(f"d={dist_map[fid]:.2f}")
        labels.append("\n".join(parts))

    series = [
        ("L", "L_noscreening", "Labor $L_j$"),
        ("w", "w_noscreening", "Wage $w_j$"),
        ("Q_j", "S_noscreening", "Average skill $Q_j$"),
    ]
    if "qbar" in df.columns:
        series.append(("qbar", None, r"Cutoff $\bar{q}_j$"))

    fig, axes = plt.subplots(len(series), 1, figsize=(10, 4 * len(series)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (col_s, col_ns, title) in zip(axes, series):
        ax.scatter(x, df[col_s], marker="o", label="Screening")
        if col_ns is not None and col_ns in df.columns:
            ax.scatter(x, df[col_ns], marker="s", label="No-screening")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.legend()
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Firm index / ID (with A, avg distance)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def plot_worker_markdowns_by_firm(
    worker_df: pd.DataFrame, firm_df: pd.DataFrame, out_path: Path
) -> bool:
    """Plot worker markdown distributions per firm and overlay screening markdowns."""
    if "firm_index" not in worker_df or "markdown_worker" not in worker_df:
        return False

    valid = worker_df.attrs.get("valid_mask")
    if valid is None:
        valid = worker_df["markdown_worker"].notna()
    valid_mask = (
        valid
        & worker_df["firm_index"].notna()
        & worker_df["markdown_worker"].notna()
    )
    if not np.any(valid_mask):
        return False

    if {"elastic_L_w_w", "elastic_S_w_w"}.issubset(firm_df.columns):
        elastic_L = firm_df["elastic_L_w_w"].to_numpy(dtype=float)
        elastic_S = firm_df["elastic_S_w_w"].to_numpy(dtype=float)
        eps_elastic = 1e-12
        denom = np.maximum(1.0 + elastic_L, eps_elastic)
        e_vals = (1.0 - elastic_S) / denom
    elif "markup_screening" in firm_df.columns:
        e_vals = firm_df["markup_screening"].to_numpy(dtype=float)
    else:
        return False

    num_firms = firm_df.shape[0]
    firm_positions = np.arange(num_firms)
    worker_positions = worker_df.loc[valid_mask, "firm_index"].astype(int).to_numpy()
    worker_markdowns = worker_df.loc[valid_mask, "markdown_worker"].to_numpy(dtype=float)

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.25, 0.25, size=worker_positions.shape[0])
    x_coords = worker_positions + jitter

    fig, ax = plt.subplots(figsize=(10, 6))
    worker_scatter = ax.scatter(
        x_coords,
        worker_markdowns,
        s=12,
        alpha=0.3,
        edgecolor="none",
        label="Worker markdowns",
    )
    finite_e = np.isfinite(e_vals)
    screening_scatter = ax.scatter(
        firm_positions[finite_e],
        e_vals[finite_e],
        color="tab:red",
        marker="_",
        s=220,
        linewidths=2.4,
        label=r"Markdown for average worker",
    )
    ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("Firm index (natural order)")
    ax.set_ylabel("Worker markdown")
    ax.set_title("Worker Markdown Distributions by Firm")
    ax.set_xlim(-0.5, num_firms - 0.5)
    ax.legend(handles=[worker_scatter, screening_scatter], loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def plot_firm_markdowns(firm_df: pd.DataFrame, out_path: Path) -> None:
    """Scatter markdowns vs firm-average skill."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        firm_df["Q_j"],
        firm_df["markdown_at_Qj"],
        marker="o",
        s=60,
        alpha=0.85,
        label="Using firm average skill $Q_j$",
    )
    ax.scatter(
        firm_df["Q_j"],
        firm_df["markdown_at_Q_market"],
        marker="s",
        s=60,
        alpha=0.85,
        label="Using market average skill $\\bar{Q}$",
    )
    ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("Firm average skill $Q_j$")
    ax.set_ylabel("Markdown")
    ax.set_title("Firm-Level Wage Markdowns")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_wage_vs_skill(firm_df: pd.DataFrame, out_path: Path) -> bool:
    """Scatter wages vs firm-average skill."""
    if not {"w", "Q_j"}.issubset(firm_df.columns):
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        firm_df["Q_j"],
        firm_df["w"],
        marker="o",
        s=60,
        alpha=0.85,
        label="Firm wage vs average skill",
    )
    ax.set_xlabel("Firm average skill $Q_j$")
    ax.set_ylabel("Wage $w_j$")
    ax.set_title("Wages vs Firm Average Skill")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def plot_firm_size_scatter(firm_df: pd.DataFrame, out_path: Path, title_suffix: str = "") -> bool:
    """
    Four-panel scatter: firm size vs wage, TFP, cutoff, and skill.
    Size is proxied by L (or L_noscreening for the no-screening dataset), plotted on a log scale.
    """
    size_col = "L" if "L" in firm_df.columns else ("L_noscreening" if "L_noscreening" in firm_df.columns else None)
    if size_col not in firm_df.columns:
        print("Skipping firm size scatter: no L column found.")
        return False

    size_vals = firm_df[size_col]
    plot_specs = [
        ("w", "Wage", "o"),
        ("A", "TFP", "s"),
        ("qbar", "Cutoff", "^"),
        ("Q_j", "Avg skill", "D"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    any_plotted = False
    for ax, spec in zip(axes, plot_specs):
        col, label, marker = spec
        if col not in firm_df.columns:
            ax.axis("off")
            continue
        ax.scatter(
            size_vals,
            firm_df[col],
            s=60,
            alpha=0.8,
            marker=marker,
            edgecolor="none",
        )
        ax.set_xscale("log")
        ax.set_xlabel(f"Firm size ({size_col})")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Firm Size{title_suffix}")
        ax.grid(True, alpha=0.2)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print("Skipping firm size scatter: no target columns available.")
        return False

    # Hide any remaining unused subplots
    for idx in range(len(plot_specs), axes.size):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def plot_markup_vs_cutoff(firm_df: pd.DataFrame, out_path: Path) -> bool:
    """Scatter wage markup proxies against cutoff costs."""
    required = {"qbar", "markup_naive", "markup_screening", "markup_naive_pass_through"}
    if not required.issubset(firm_df.columns):
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    screening_scatter = ax.scatter(
        firm_df["qbar"],
        firm_df["markup_screening"],
        marker="^",
        s=70,
        alpha=0.85,
        label=r"Screening $(1-\varepsilon_j^S)/(1+\varepsilon_j^L)$",
    )
    naive_scatter = ax.scatter(
        firm_df["qbar"],
        firm_df["markup_naive"],
        marker="o",
        s=60,
        alpha=0.85,
        label=r"Naive (correct elasticity) $1/(1+\varepsilon_j^L)$",
    )
    passthrough_scatter = ax.scatter(
        firm_df["qbar"],
        firm_df["markup_naive_pass_through"],
        marker="d",
        s=60,
        alpha=0.85,
        label=r"Naive pass-through $1/(1+\varepsilon_{j}^{L,A}/\varepsilon_{j}^{w,A})$",
    )
    ax.set_xlabel(r"Firm cutoff $\bar{q}_j$")
    ax.set_ylabel("Wage markdown")
    ax.set_ylim(0.0, 0.35)
    ax.set_title("Wage Markdowns in Naive and Screening Models")
    ax.legend(handles=[screening_scatter, naive_scatter, passthrough_scatter])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def _cutoff_quantiles_from_workers(cutoffs: np.ndarray, worker_df: pd.DataFrame) -> np.ndarray | None:
    if "s_skill" not in worker_df.columns:
        return None

    skill_vals = worker_df["s_skill"].dropna().to_numpy(dtype=float)
    if skill_vals.size == 0:
        return None

    skill_sorted = np.sort(skill_vals)
    quantiles = np.full_like(cutoffs, np.nan, dtype=float)
    valid = np.isfinite(cutoffs)
    if np.any(valid):
        ranks = np.searchsorted(skill_sorted, cutoffs[valid], side="right")
        quantiles[valid] = ranks / skill_sorted.size
    return quantiles


def plot_markup_vs_cutoff_quantiles(
    firm_df: pd.DataFrame,
    worker_df: pd.DataFrame,
    out_path: Path,
    firm_df_noscreen: pd.DataFrame | None = None,
    include_no_screening: bool = True,
    include_no_screening_labor_only: bool = False,
    include_naive: bool = True,
    include_labor_only: bool = True,
    include_screening: bool = True,
    title: str = "Wage Markdowns vs Cutoff Quantiles",
    regime_offset: float = 0.0,
) -> bool:
    """Plot markup proxies vs cutoff quantiles from the worker skill distribution."""
    required = {"qbar", "markup_naive", "markup_screening", "markup_naive_pass_through"}
    if not required.issubset(firm_df.columns):
        return False

    is_two_firm_screening_vs_no = (
        firm_df.shape[0] == 2
        and include_no_screening
        and include_no_screening_labor_only
        and include_labor_only
        and include_screening
        and not include_naive
        and firm_df_noscreen is not None
        and "A" in firm_df.columns
    )
    if is_two_firm_screening_vs_no:
        firm_ids = firm_df["firm_id"].to_numpy(dtype=int)
        A_vals = firm_df["A"].to_numpy(dtype=float)
        order = np.array([int(np.nanargmax(A_vals)), int(np.nanargmin(A_vals))], dtype=int)
        firm_ids_ordered = firm_ids[order]
        firm_df_noscreen_indexed = firm_df_noscreen.set_index("firm_id")
        needed_cols = {"markup_no_screening", "markup_no_screening_labor_only"}
        if not needed_cols.issubset(firm_df_noscreen_indexed.columns):
            return False
        noscreen_labor = firm_df_noscreen_indexed.reindex(firm_ids_ordered)[
            "markup_no_screening_labor_only"
        ].to_numpy(dtype=float)
        noscreen_full = firm_df_noscreen_indexed.reindex(firm_ids_ordered)[
            "markup_no_screening"
        ].to_numpy(dtype=float)
        noscreen_pass = None
        if "markup_no_screening_pass_through" in firm_df_noscreen_indexed.columns:
            noscreen_pass = firm_df_noscreen_indexed.reindex(firm_ids_ordered)[
                "markup_no_screening_pass_through"
            ].to_numpy(dtype=float)
        screen_labor = firm_df["markup_naive"].to_numpy(dtype=float)[order]
        screen_full = firm_df["markup_screening"].to_numpy(dtype=float)[order]
        screen_pass = firm_df["markup_naive_pass_through"].to_numpy(dtype=float)[order]
        noscreen_labor = np.round(noscreen_labor, 3)
        noscreen_full = np.round(noscreen_full, 3)
        screen_labor = np.round(screen_labor, 3)
        screen_full = np.round(screen_full, 3)
        screen_pass = np.round(screen_pass, 3)
        if noscreen_pass is not None:
            noscreen_pass = np.round(noscreen_pass, 3)
        series_all = [noscreen_labor, noscreen_full, screen_labor, screen_full, screen_pass]
        if noscreen_pass is not None:
            series_all.append(noscreen_pass)
        series_all = np.concatenate(series_all)
        if not np.all(np.isfinite(series_all)):
            return False
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        x_base = np.arange(2, dtype=float)
        offset = 0.18
        x_no = x_base - offset
        x_screen = x_base + offset
        bar_color = "#666666"
        triangle_color = "#004488"
        diamond_color = "#BB5566"
        screening_color = bar_color
        grey = bar_color
        bar_width = 0.18
        tri_offset = -0.035
        diamond_offset = 0.035
        ax.bar(
            x_no,
            noscreen_full,
            width=bar_width,
            color=bar_color,
            alpha=0.25,
            edgecolor="none",
            zorder=1,
        )
        ax.bar(
            x_screen,
            screen_full,
            width=bar_width,
            color=bar_color,
            alpha=0.25,
            edgecolor="none",
            zorder=1,
        )
        ax.scatter(
            x_no + tri_offset,
            noscreen_labor,
            marker="^",
            s=110,
            color=triangle_color,
            edgecolor=triangle_color,
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            x_no,
            noscreen_full,
            marker="s",
            s=110,
            color=bar_color,
            edgecolor=bar_color,
            linewidth=0.9,
            zorder=4,
        )
        ax.scatter(
            x_screen + tri_offset,
            screen_labor,
            marker="^",
            s=110,
            color=triangle_color,
            edgecolor=triangle_color,
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            x_screen,
            screen_full,
            marker="s",
            s=110,
            color=bar_color,
            edgecolor=bar_color,
            linewidth=0.9,
            zorder=4,
        )
        if noscreen_pass is not None:
            ax.scatter(
                x_no + diamond_offset,
                noscreen_pass,
                marker="D",
                s=90,
                color=diamond_color,
                edgecolor=diamond_color,
                linewidth=1.0,
                zorder=4,
            )
        ax.scatter(
            x_screen + diamond_offset,
            screen_pass,
            marker="D",
            s=90,
            color=diamond_color,
            edgecolor=diamond_color,
            linewidth=1.0,
            zorder=4,
        )
        ax.set_xticks(np.array([x_no[0], x_screen[0], x_no[1], x_screen[1]]))
        ax.set_xticklabels(["No screening", "Screening", "No screening", "Screening"])
        for idx, label in enumerate(("High-productivity firm", "Low-productivity firm")):
            ax.text(
                x_base[idx],
                -0.12,
                label,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=13,
            )
        ax.set_ylabel(r"Wage markdown ($\frac{MR-w}{MR}$)", fontsize=13)
        y_min = float(np.nanmin(series_all))
        y_max = float(np.nanmax(series_all))
        pad = 0.02
        y_lower = np.floor((y_min - pad) * 100.0) / 100.0
        y_upper = np.ceil((y_max + pad) * 100.0) / 100.0
        if y_lower == y_upper:
            y_lower -= 0.01
            y_upper += 0.01
        ax.set_ylim(0.0, y_upper)
        ax.grid(True, alpha=0.2)
        legend_handles = [
            Line2D([], [], linestyle="none", label=r"$\bf{Markdown}$"),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=bar_color,
                markeredgecolor=bar_color,
                markersize=8,
                label=r"True markdown: $(1-\varepsilon_j^Q)/(1+\varepsilon_j^L)$",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor=triangle_color,
                markeredgecolor=triangle_color,
                markersize=8,
                label=r"Conventional markdown: $1/(1+\varepsilon_j^L)$",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=diamond_color,
                markeredgecolor=diamond_color,
                markersize=7,
                label=r"Markdown implied by pass-through estimator: $1/(1+\hat\varepsilon_j^L)$",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            handlelength=1.2,
            handletextpad=0.6,
        )
        ax.set_xlim(-0.5, 1.5)
        quantiles = _cutoff_quantiles_from_workers(
            firm_df["qbar"].to_numpy(dtype=float), worker_df
        )
        if quantiles is not None and np.all(np.isfinite(quantiles)):
            quantiles_ordered = quantiles[order]
        else:
            quantiles_ordered = np.array([np.nan, np.nan])

        q_text_y = -0.06
        for idx, label in enumerate((r"$\bar q_j$", r"$\bar q_j$")):
            q_val_screen = quantiles_ordered[idx]
            if np.isfinite(q_val_screen):
                q_text = f"{label}={q_val_screen:.2f}"
                ax.text(
                    x_screen[idx],
                    q_text_y,
                    q_text,
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=10,
                    color="black",
                )
            ax.text(
                x_no[idx],
                q_text_y,
                f"{label}=0.00",
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=10,
                color="black",
            )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.24)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return True

    is_two_firm_screening_only = (
        firm_df.shape[0] == 2
        and not include_no_screening
        and include_naive
        and include_screening
        and not include_labor_only
        and "A" in firm_df.columns
    )
    if is_two_firm_screening_only:
        firm_ids = firm_df["firm_id"].to_numpy(dtype=int)
        A_vals = firm_df["A"].to_numpy(dtype=float)
        order = np.array([int(np.nanargmax(A_vals)), int(np.nanargmin(A_vals))], dtype=int)
        pass_through_vals = firm_df["markup_naive_pass_through"].to_numpy(dtype=float)[order]
        screen_labor_only = firm_df["markup_naive"].to_numpy(dtype=float)[order]
        screen_vals = firm_df["markup_screening"].to_numpy(dtype=float)[order]
        if not np.all(
            np.isfinite(np.concatenate([pass_through_vals, screen_labor_only, screen_vals]))
        ):
            return False
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        x_base = np.arange(2, dtype=float)
        offset = 0.06
        x_left = x_base - offset
        x_right = x_base + offset
        grey = "#4d4d4d"
        crimson = "#b22222"
        ax.vlines(
            x_right,
            np.minimum(screen_labor_only, screen_vals),
            np.maximum(screen_labor_only, screen_vals),
            color=crimson,
            alpha=0.35,
            linewidth=1.2,
            zorder=1,
        )
        ax.scatter(
            x_left,
            pass_through_vals,
            marker="o",
            s=80,
            facecolors="none",
            edgecolors=grey,
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            x_right,
            screen_labor_only,
            marker="o",
            s=80,
            facecolors="none",
            edgecolors=crimson,
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            x_right,
            screen_vals,
            marker="o",
            s=80,
            color=crimson,
            edgecolor="black",
            linewidth=0.4,
            zorder=4,
        )
        ax.set_xticks(x_base)
        ax.set_xticklabels(["Firm H", "Firm L"])
        ax.set_ylabel("Wage markdown")
        y_min = float(
            np.nanmin([pass_through_vals.min(), screen_labor_only.min(), screen_vals.min()])
        )
        y_max = float(
            np.nanmax([pass_through_vals.max(), screen_labor_only.max(), screen_vals.max()])
        )
        pad = 0.01
        y_lower = np.floor((y_min - pad) * 100.0) / 100.0
        y_upper = np.ceil((y_max + pad) * 100.0) / 100.0
        if y_lower == y_upper:
            y_lower -= 0.01
            y_upper += 0.01
        ax.set_ylim(y_lower, y_upper)
        ax.grid(True, alpha=0.2)
        legend_handles = [
            Line2D([], [], linestyle="none", label=r"$\bf{Markdown}$"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=grey,
                markeredgecolor=grey,
                markersize=8,
                label="Pass-through estimate",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=crimson,
                markeredgecolor=crimson,
                markersize=8,
                label="Screening truth",
            ),
            Line2D([], [], linestyle="none", label=r"$\bf{Markdown\ component}$"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=8,
                label="Labor only",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=8,
                label="Labor and skill",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            handlelength=1.2,
            handletextpad=0.6,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return True

    cutoffs = firm_df["qbar"].to_numpy(dtype=float)
    quantiles = _cutoff_quantiles_from_workers(cutoffs, worker_df)
    if quantiles is None:
        return False

    naive_vals = firm_df["markup_naive_pass_through"].to_numpy(dtype=float)
    labor_only_vals = firm_df["markup_naive"].to_numpy(dtype=float)
    screening_vals = firm_df["markup_screening"].to_numpy(dtype=float)
    noscreen_vals = None
    noscreen_labor_only_vals = None
    if (include_no_screening or include_no_screening_labor_only) and firm_df_noscreen is not None:
        if "firm_id" not in firm_df.columns:
            return False
        if include_no_screening and {"firm_id", "markup_no_screening"}.issubset(
            firm_df_noscreen.columns
        ):
            noscreen_map = firm_df_noscreen.set_index("firm_id")["markup_no_screening"]
            noscreen_vals = firm_df["firm_id"].map(noscreen_map).to_numpy(dtype=float)
        if include_no_screening_labor_only and {"firm_id", "markup_no_screening_labor_only"}.issubset(
            firm_df_noscreen.columns
        ):
            noscreen_labor_map = firm_df_noscreen.set_index("firm_id")[
                "markup_no_screening_labor_only"
            ]
            noscreen_labor_only_vals = firm_df["firm_id"].map(noscreen_labor_map).to_numpy(dtype=float)
    if include_no_screening and noscreen_vals is None:
        return False
    if include_no_screening_labor_only and noscreen_labor_only_vals is None:
        return False

    valid = np.isfinite(quantiles)
    if include_naive:
        valid = valid & np.isfinite(naive_vals)
    if include_labor_only:
        valid = valid & np.isfinite(labor_only_vals)
    if include_screening:
        valid = valid & np.isfinite(screening_vals)
    if include_no_screening and noscreen_vals is not None:
        valid = valid & np.isfinite(noscreen_vals)
    if include_no_screening_labor_only and noscreen_labor_only_vals is not None:
        valid = valid & np.isfinite(noscreen_labor_only_vals)
    if not np.any(valid):
        return False

    x_vals = quantiles[valid]
    y_series = []
    y_noscreen = None
    y_noscreen_labor_only = None
    y_naive = None
    y_labor = None
    y_screen = None
    if include_no_screening and noscreen_vals is not None:
        y_noscreen = noscreen_vals[valid]
        y_series.append(y_noscreen)
    if include_no_screening_labor_only and noscreen_labor_only_vals is not None:
        y_noscreen_labor_only = noscreen_labor_only_vals[valid]
        y_series.append(y_noscreen_labor_only)
    if include_naive:
        y_naive = naive_vals[valid]
        y_series.append(y_naive)
    if include_labor_only:
        y_labor = labor_only_vals[valid]
        y_series.append(y_labor)
    if include_screening:
        y_screen = screening_vals[valid]
        y_series.append(y_screen)

    y_min = np.minimum.reduce(y_series)
    y_max = np.maximum.reduce(y_series)

    fig, ax = plt.subplots(figsize=(8, 5))
    has_screening = include_naive or include_labor_only or include_screening
    has_no_screening = include_no_screening or include_no_screening_labor_only
    offset = regime_offset if (regime_offset != 0.0 and has_screening and has_no_screening) else 0.0
    x_no = np.clip(x_vals - offset, 0.0, 1.0)
    x_screen = np.clip(x_vals + offset, 0.0, 1.0)
    drew_regime_lines = False
    if include_no_screening and include_no_screening_labor_only and y_noscreen is not None and y_noscreen_labor_only is not None:
        ax.vlines(
            x_no,
            np.minimum(y_noscreen, y_noscreen_labor_only),
            np.maximum(y_noscreen, y_noscreen_labor_only),
            color="#1f77b4",
            alpha=0.35,
            linewidth=1.0,
            zorder=1,
        )
        drew_regime_lines = True
    if include_labor_only and include_screening:
        ax.vlines(
            x_screen,
            np.minimum(y_labor, y_screen),
            np.maximum(y_labor, y_screen),
            color="#2ca02c",
            alpha=0.35,
            linewidth=1.0,
            zorder=1,
        )
        drew_regime_lines = True
    if not drew_regime_lines:
        ax.vlines(x_vals, y_min, y_max, color="0.5", alpha=0.35, linewidth=1.0, zorder=1)
    handles = []
    use_regime_palette = (
        include_no_screening
        and include_no_screening_labor_only
        and include_labor_only
        and include_screening
    )
    if include_no_screening_labor_only and y_noscreen_labor_only is not None:
        noscreen_labor_scatter = ax.scatter(
            x_no,
            y_noscreen_labor_only,
            marker="s",
            s=70,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.4,
            label="No screening: labor only" if use_regime_palette else "No screening: labor only",
            zorder=2,
        )
        handles.append(noscreen_labor_scatter)
    if include_no_screening and y_noscreen is not None:
        noscreen_marker = "o" if use_regime_palette else "D"
        noscreen_label = "No screening: labor and skill" if use_regime_palette else "No screening"
        noscreen_scatter = ax.scatter(
            x_no,
            y_noscreen,
            marker=noscreen_marker,
            s=70,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.4,
            label=noscreen_label,
            zorder=2,
        )
        handles.append(noscreen_scatter)
    if include_naive:
        naive_scatter = ax.scatter(
            x_screen,
            y_naive,
            marker="s",
            s=70,
            color="red",
            edgecolor="black",
            linewidth=0.4,
            label="Screening, naive",
            zorder=3,
        )
        handles.append(naive_scatter)
    if include_labor_only:
        labor_color = "#2ca02c" if use_regime_palette else "#f1c40f"
        labor_label = "Screening: labor only" if use_regime_palette else "Screening, labor only"
        labor_scatter = ax.scatter(
            x_screen,
            y_labor,
            marker="s" if use_regime_palette else "^",
            s=70,
            color=labor_color,
            edgecolor="black",
            linewidth=0.4,
            label=labor_label,
            zorder=4,
        )
        handles.append(labor_scatter)
    if include_screening:
        screening_color = "#2ca02c" if use_regime_palette else "#2ca02c"
        screening_label = (
            "Screening: labor and skill" if use_regime_palette else "Screening, labor and skill"
        )
        screening_scatter = ax.scatter(
            x_screen,
            y_screen,
            marker="o" if use_regime_palette else "o",
            s=70,
            color=screening_color,
            edgecolor="black",
            linewidth=0.4,
            label=screening_label,
            zorder=5,
        )
        handles.append(screening_scatter)
    ax.set_xlabel("Cutoff quantile in worker skill distribution")
    ax.set_ylabel("Wage markdown")
    ax.set_xlim(0.0, 1.0)
    y_span = float(np.nanmax(y_max - y_min))
    pad = 0.05 * y_span if y_span > 0 else 0.02
    y_lower = float(np.nanmin(y_min)) - pad
    y_upper = float(np.nanmax(y_max)) + pad
    ax.set_ylim(y_lower, y_upper)
    if "firm_id" in firm_df.columns and "A" in firm_df.columns:
        firm_ids_all = firm_df["firm_id"].to_numpy(dtype=int)
        A_all = firm_df["A"].to_numpy(dtype=float)
        if firm_ids_all.size == 2 and A_all.size == 2:
            firm_ids_valid = firm_ids_all[valid]
            high_firm_id = firm_ids_all[int(np.nanargmax(A_all))]
            low_firm_id = firm_ids_all[int(np.nanargmin(A_all))]
            label_pad = 0.02 * y_span if y_span > 0 else 0.02
            for label, firm_id in (("Firm H", high_firm_id), ("Firm L", low_firm_id)):
                match_idx = np.where(firm_ids_valid == firm_id)[0]
                if match_idx.size == 0:
                    continue
                idx = int(match_idx[0])
                y_candidates = []
                for series in (y_noscreen, y_noscreen_labor_only, y_naive, y_labor, y_screen):
                    if series is not None and np.isfinite(series[idx]):
                        y_candidates.append(series[idx])
                if not y_candidates:
                    continue
                y_label = float(np.nanmax(y_candidates)) + label_pad
                ax.text(
                    x_vals[idx],
                    y_label,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(handles=handles)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def plot_markdowns_vs_firm_index(firm_df: pd.DataFrame, out_path: Path) -> bool:
    """Plot markdown proxies against firm index."""
    required = {"markup_naive", "markup_screening", "markup_naive_pass_through"}
    if not required.issubset(firm_df.columns):
        return False

    idx = np.arange(firm_df.shape[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(idx, firm_df["markup_screening"], marker="^", label="Screening")
    ax.scatter(idx, firm_df["markup_naive"], marker="o", label="Naive (correct elasticity)")
    ax.scatter(idx, firm_df["markup_naive_pass_through"], marker="d", label="Naive pass-through")
    ax.set_xlabel("Firm index (natural order)")
    ax.set_ylabel("Wage markdown")
    ax.set_ylim(0.0, 0.35)
    ax.set_title("Markdown Comparisons Across Firms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    raw_dir = get_data_subdir(DATA_RAW, create=True)
    clean_dir = get_data_subdir(DATA_CLEAN, create=True)
    build_dir = get_data_subdir(DATA_BUILD, create=True)
    plot_dir = get_output_subdir(OUTPUT_MARKDOWNS, create=True)
    parser = argparse.ArgumentParser(
        description="Compute implied wage markdowns from equilibrium outcomes."
    )
    parser.add_argument(
        "--equilibrium_path",
        type=str,
        default=str(clean_dir / "equilibrium_firms.csv"),
        help="CSV with equilibrium firm outcomes.",
    )
    parser.add_argument(
        "--equilibrium_noscreening_path",
        type=str,
        default=str(clean_dir / "equilibrium_firms_noscreening.csv"),
        help="Optional CSV with no-screening equilibrium outcomes.",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default=str(raw_dir / "parameters_effective.csv"),
        help="CSV with model parameters.",
    )
    parser.add_argument(
        "--workers_path",
        type=str,
        default=str(build_dir / "workers_dataset.csv"),
        help="Optional worker dataset CSV (skips worker-level markdowns if missing).",
    )
    parser.add_argument(
        "--workers_noscreening_path",
        type=str,
        default=str(build_dir / "workers_dataset_noscreening.csv"),
        help="Optional worker dataset CSV for no-screening equilibrium (used in overlay plots if present).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(build_dir),
        help="Directory to write markdown data CSVs.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=str(plot_dir),
        help="Directory to write markdown plots (defaults to output/markdowns/).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    equilibrium_path = Path(args.equilibrium_path)
    equilibrium_noscreening_path = Path(args.equilibrium_noscreening_path)
    params_path = Path(args.params_path)
    workers_path = Path(args.workers_path) if args.workers_path is not None else None
    workers_noscreen_path = (
        Path(args.workers_noscreening_path) if args.workers_noscreening_path is not None else None
    )
    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)

    print("Implied Wage Markdown Computation (Data Environment Item 4)")
    print("=" * 70)
    print(f"Equilibrium data: {equilibrium_path}")
    print(f"Equilibrium (no-screening) data: {equilibrium_noscreening_path if equilibrium_noscreening_path.exists() else '(not found)'}")
    print(f"Parameters:       {params_path}")
    if workers_path is not None:
        print(f"Worker data:      {workers_path}")
    if workers_noscreen_path is not None:
        print(f"Worker data (no-screening): {workers_noscreen_path if workers_noscreen_path.exists() else '(not found)'}")
    print(f"Output directory: {out_dir}")
    print(f"Plot directory:   {plot_dir}")

    if not equilibrium_path.exists():
        print(f"Error: equilibrium file {equilibrium_path} does not exist.")
        return 1
    if not params_path.exists():
        print(f"Error: parameter file {params_path} does not exist.")
        return 1

    ensure_directory(out_dir)
    ensure_directory(plot_dir)

    firm_df_noscreen = None
    if equilibrium_noscreening_path.exists():
        try:
            firm_df_noscreen = pd.read_csv(equilibrium_noscreening_path)
            print(f"[INFO] Loaded no-screening equilibrium from {equilibrium_noscreening_path}")
        except Exception as exc:
            print(f"[WARN] Could not read no-screening equilibrium: {exc}")

    # Read data
    equilibrium_df = pd.read_csv(equilibrium_path)
    params = read_parameters_csv(params_path)
    worker_df: pd.DataFrame | None = None
    worker_df_noscreen: pd.DataFrame | None = None

    # Firm-level markdowns
    firm_markdowns = compute_firm_markdowns(equilibrium_df, params)
    firm_output = out_dir / "firm_markdowns.csv"
    firm_markdowns.to_csv(firm_output, index=False)
    print(f"[OK] Firm markdowns written to {firm_output}")
    firm_markdowns_noscreen = None
    if firm_df_noscreen is not None:
        try:
            firm_markdowns_noscreen = compute_noscreening_markdowns(firm_df_noscreen, params)
            if firm_markdowns_noscreen is None:
                print("[WARN] Could not compute no-screening markdowns (missing columns).")
        except Exception as exc:
            print(f"[WARN] Could not compute no-screening markdowns: {exc}")

    # Plot firm markdowns
    firm_plot_path = plot_dir / "firm_markdowns_vs_average_skill.png"
    plot_firm_markdowns(firm_markdowns, firm_plot_path)
    print(f"[OK] Firm markdown plot saved to {firm_plot_path}")

    wage_skill_plot_path = plot_dir / "wages_vs_skill.png"
    if plot_wage_vs_skill(firm_markdowns, wage_skill_plot_path):
        print(f"[OK] Wage vs skill plot saved to {wage_skill_plot_path}")
    else:
        print("[WARN] Wage vs skill plot skipped (missing columns).")

    size_plot_path = plot_dir / "firm_size_scatter.png"
    if plot_firm_size_scatter(firm_markdowns, size_plot_path):
        print(f"[OK] Firm size scatter saved to {size_plot_path}")
    else:
        print("[WARN] Firm size scatter skipped (missing data).")
    if firm_df_noscreen is not None:
        size_plot_ns_path = plot_dir / "firm_size_scatter_noscreening.png"
        if plot_firm_size_scatter(firm_df_noscreen, size_plot_ns_path, title_suffix=" (no-screening)"):
            print(f"[OK] Firm size scatter (no-screening) saved to {size_plot_ns_path}")
        else:
            print("[WARN] Firm size scatter (no-screening) skipped (missing data).")

    markup_plot_path = plot_dir / "markup_vs_cutoff.png"
    if plot_markup_vs_cutoff(firm_markdowns, markup_plot_path):
        print(f"[OK] Markup vs cutoff plot saved to {markup_plot_path}")
    else:
        print("[WARN] Elasticity columns not found; skipping markup vs cutoff plot.")

    idx_plot_path = plot_dir / "firm_markdowns_vs_firm_index.png"
    if plot_markdowns_vs_firm_index(firm_markdowns, idx_plot_path):
        print(f"[OK] Markdown vs firm index plot saved to {idx_plot_path}")
    else:
        print("[WARN] Elasticity columns not found; skipping firm index plot.")

    # Worker-level markdowns (optional)
    if workers_path is not None and workers_path.exists():
        worker_df = pd.read_csv(workers_path)
        worker_df_noscreen = None
        if workers_noscreen_path is not None and workers_noscreen_path.exists():
            try:
                worker_df_noscreen = pd.read_csv(workers_noscreen_path)
            except Exception as exc:
                print(f"[WARN] Could not read worker no-screening dataset: {exc}")
        quantile_screening_vs_no_path = plot_dir / "markup_vs_cutoff_quantiles_screening_vs_noscreening.png"
        if plot_markup_vs_cutoff_quantiles(
            firm_markdowns,
            worker_df,
            quantile_screening_vs_no_path,
            firm_markdowns_noscreen,
            include_no_screening=True,
            include_no_screening_labor_only=True,
            include_naive=False,
            include_labor_only=True,
            include_screening=True,
            title="Markdowns vs Cutoff Quantiles (No Screening vs Screening)",
            regime_offset=0.006,
        ):
            print(
                "[OK] Markup vs cutoff quantiles (screening vs no screening) plot saved to "
                f"{quantile_screening_vs_no_path}"
            )
        else:
            print("[WARN] Skipping screening vs no-screening quantiles plot (missing data).")
        quantile_screening_only_path = plot_dir / "markup_vs_cutoff_quantiles_screening_only.png"
        if plot_markup_vs_cutoff_quantiles(
            firm_markdowns,
            worker_df,
            quantile_screening_only_path,
            firm_markdowns_noscreen,
            include_no_screening=False,
            include_naive=True,
            include_labor_only=False,
            include_screening=True,
            title="Markdowns vs Cutoff Quantiles (Screening: Naive vs Full)",
        ):
            print(
                "[OK] Markup vs cutoff quantiles (screening only) plot saved to "
                f"{quantile_screening_only_path}"
            )
        else:
            print("[WARN] Skipping screening-only quantiles plot (missing data).")
        try:
            worker_markdowns = compute_worker_markdowns(worker_df, firm_markdowns)
        except ValueError as exc:
            print(f"[WARN] Skipping worker markdowns: {exc}")
        else:
            worker_output = out_dir / "worker_markdowns.csv"
            worker_markdowns.to_csv(worker_output, index=False)
            print(f"[OK] Worker markdowns written to {worker_output}")

            worker_plot_path = plot_dir / "worker_markdowns_vs_skill.png"
            plot_worker_markdowns(worker_markdowns, worker_plot_path)
            print(f"[OK] Worker markdown plot saved to {worker_plot_path}")

            worker_wage_plot_path = plot_dir / "worker_wages_vs_skill.png"
            if plot_worker_wages_vs_skill(worker_markdowns, worker_wage_plot_path):
                print(f"[OK] Worker wages vs skill plot saved to {worker_wage_plot_path}")
            else:
                print("[WARN] Worker wages vs skill plot skipped (missing data).")

            worker_firm_plot_path = plot_dir / "worker_markdowns_by_firm.png"
            if plot_worker_markdowns_by_firm(worker_markdowns, firm_markdowns, worker_firm_plot_path):
                print(f"[OK] Worker markdown by firm plot saved to {worker_firm_plot_path}")
            else:
                print("[WARN] Could not create worker-by-firm markdown plot (missing data).")

        # Binned plots using no-screening equilibrium if available (overlay with screening)
        if firm_df_noscreen is not None:
            try:
                if binscatter_unemp_and_wage_overlay(worker_df, worker_df_noscreen, firm_markdowns, firm_df_noscreen, plot_dir):
                    print(f"[OK] Overlay unemployment/wage vs skill plots saved to {plot_dir}")
                else:
                    print("[WARN] Could not create overlay binscatter (missing data).")
            except Exception as exc:
                print(f"[WARN] Skipping overlay binscatter due to error: {exc}")
    else:
        # Fallback: try default data/build/workers_dataset.csv for distance calculations and labels
        fallback_worker = get_data_subdir(DATA_BUILD) / "workers_dataset.csv"
        if fallback_worker.exists():
            try:
                worker_df = pd.read_csv(fallback_worker)
                print(f"[INFO] Loaded worker data from fallback {fallback_worker} for distance summaries.")
            except Exception as exc:
                worker_df = None
                print(f"[WARN] Could not read fallback worker dataset {fallback_worker}: {exc}")
        else:
            print("[WARN] Worker dataset not provided or missing; skipping worker-level markdowns.")

    if firm_df_noscreen is not None:
        compare_path = plot_dir / "firm_screen_vs_noscreen_by_index.png"
        if plot_screen_vs_noscreen_by_index(firm_markdowns, firm_df_noscreen, compare_path, worker_df):
            print(f"[OK] Screening vs no-screening (by firm index) plot saved to {compare_path}")
        else:
            print("[WARN] Could not create screening vs no-screening firm index plot (missing data).")

    print("Completed implied wage markdown diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
