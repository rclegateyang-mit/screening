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
  - firm_markdowns_vs_average_skill.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .. import get_data_dir
except ImportError:  # pragma: no cover - direct script execution
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
    from code import get_data_dir  # type: ignore


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
    required = {"alpha", "beta", "mu_s"}
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
    """Compute firm-level markdowns evaluated at S_j and market-average skill."""
    required_cols = {"firm_id", "w", "L", "S", "A"}
    missing = required_cols - set(equilibrium_df.columns)
    if missing:
        raise ValueError(f"Equilibrium data missing required columns: {missing}")

    beta = params["beta"]
    mu_s = params["mu_s"]
    N_workers = float(params.get("N_workers", 1.0))
    if not np.isfinite(N_workers) or N_workers <= 0:
        raise ValueError(f"N_workers must be positive; got {N_workers}")

    eps = 1e-12
    L_share = np.maximum(equilibrium_df["L"].to_numpy(dtype=float), eps)
    L_counts = np.maximum(L_share * N_workers, eps)
    S = np.maximum(equilibrium_df["S"].to_numpy(dtype=float), eps)
    w = equilibrium_df["w"].to_numpy(dtype=float)
    A = equilibrium_df["A"].to_numpy(dtype=float)
    if "c" in equilibrium_df.columns:
        c_vals = equilibrium_df["c"].to_numpy(dtype=float)
    else:
        c_vals = np.full_like(L_share, np.nan)
    firm_ids = equilibrium_df["firm_id"].to_numpy(dtype=int)

    # MRPL_j(s) = s * (1 - beta) * A_j / (L_j * S_j)^beta
    LS_prod = np.maximum(L_counts * S, eps)
    mrpl_scale = (1.0 - beta) * A / np.power(LS_prod, beta)

    mrpl_at_Sj = S * mrpl_scale
    markdown_Sj = np.where(mrpl_at_Sj > eps, (mrpl_at_Sj - w) / mrpl_at_Sj, np.nan)

    S_market = float(mu_s)
    mrpl_at_Sbar = S_market * mrpl_scale
    markdown_Sbar = np.where(mrpl_at_Sbar > eps, (mrpl_at_Sbar - w) / mrpl_at_Sbar, np.nan)

    df = pd.DataFrame(
        {
            "firm_id": firm_ids,
            "A": A,
            "w": w,
            "L": L_share,
            "L_share": L_share,
            "L_workers": L_counts,
            "S_j": S,
            "c": c_vals,
            "markdown_at_Sj": markdown_Sj,
            "markdown_at_S_market": markdown_Sbar,
        }
    )
    if c_vals is not None and {"elastic_L_w_w", "elastic_S_w_w"}.issubset(equilibrium_df.columns):
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

    firm_index = {int(fid): idx for idx, fid in enumerate(firm_ids.tolist())}
    firm_col = "chosen_firm"

    # If reindexed workers are stored separately, fall back gracefully
    unique_firms = set(worker_df[firm_col].dropna().astype(int).unique().tolist())
    if not unique_firms.issubset(firm_index.keys()):
        fallback_col = "chosen_firm_original_id"
        if fallback_col in worker_df.columns:
            firm_col = fallback_col
            unique_firms = set(worker_df[firm_col].dropna().astype(int).unique().tolist())
        if not unique_firms.issubset(firm_index.keys()):
            raise ValueError("Worker firm identifiers do not match equilibrium firm IDs.")

    df = worker_df.copy()
    df["firm_index"] = df[firm_col].map(firm_index)
    valid = df["firm_index"].notna() & (df[firm_col] > 0)

    idx = df.loc[valid, "firm_index"].astype(int).to_numpy()
    s_skill = df.loc[valid, "s_skill"].to_numpy(dtype=float)
    mrpl_worker = s_skill * mrpl_scale[idx]
    wage_worker = wages[idx]
    eps = 1e-12
    markdown_worker = np.where(mrpl_worker > eps, (mrpl_worker - wage_worker) / mrpl_worker, np.nan)

    df["mrpl_worker"] = np.nan
    df["wage_worker"] = np.nan
    df["markdown_worker"] = np.nan
    df.loc[valid, "mrpl_worker"] = mrpl_worker
    df.loc[valid, "wage_worker"] = wage_worker
    df.loc[valid, "markdown_worker"] = markdown_worker
    df.attrs["valid_mask"] = valid
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
    ax.set_ylabel("Markdown $(\\text{MRPL}_j(s_i) - w_j) / \\text{MRPL}_j(s_i)$")
    ax.set_title("Worker-Level Wage Markdown vs Skill")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_firm_markdowns(firm_df: pd.DataFrame, out_path: Path) -> None:
    """Scatter markdowns vs firm-average skill."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        firm_df["S_j"],
        firm_df["markdown_at_Sj"],
        marker="o",
        s=60,
        alpha=0.85,
        label="Using firm average skill $S_j$",
    )
    ax.scatter(
        firm_df["S_j"],
        firm_df["markdown_at_S_market"],
        marker="s",
        s=60,
        alpha=0.85,
        label="Using market average skill $\\bar{S}$",
    )
    ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("Firm average skill $S_j$")
    ax.set_ylabel("Markdown")
    ax.set_title("Firm-Level Wage Markdowns")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_markup_vs_cutoff(firm_df: pd.DataFrame, out_path: Path) -> bool:
    """Scatter wage markup proxies against cutoff costs."""
    required = {"c", "markup_naive", "markup_screening", "markup_naive_pass_through"}
    if not required.issubset(firm_df.columns):
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    screening_scatter = ax.scatter(
        firm_df["c"],
        firm_df["markup_screening"],
        marker="^",
        s=70,
        alpha=0.85,
        label=r"Screening $(1-\varepsilon_j^S)/(1+\varepsilon_j^L)$",
    )
    naive_scatter = ax.scatter(
        firm_df["c"],
        firm_df["markup_naive"],
        marker="o",
        s=60,
        alpha=0.85,
        label=r"Naive (correct elasticity) $1/(1+\varepsilon_j^L)$",
    )
    passthrough_scatter = ax.scatter(
        firm_df["c"],
        firm_df["markup_naive_pass_through"],
        marker="d",
        s=60,
        alpha=0.85,
        label=r"Naive pass-through $1/(1+\varepsilon_{j}^{L,A}/\varepsilon_{j}^{w,A})$",
    )
    ax.set_xlabel("Firm cutoff $c_j$")
    ax.set_ylabel("Wage markdown")
    ax.set_ylim(0.0, 0.35)
    ax.set_title("Wage Markdowns in Naive and Screening Models")
    ax.legend(handles=[screening_scatter, naive_scatter, passthrough_scatter])
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
    data_dir = get_data_dir(create=True)
    parser = argparse.ArgumentParser(
        description="Compute implied wage markdowns from equilibrium outcomes."
    )
    parser.add_argument(
        "--equilibrium_path",
        type=str,
        default=str(data_dir / "equilibrium_firms.csv"),
        help="CSV with equilibrium firm outcomes.",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default=str(data_dir / "parameters_effective.csv"),
        help="CSV with model parameters.",
    )
    parser.add_argument(
        "--workers_path",
        type=str,
        default=str(data_dir / "workers_dataset.csv"),
        help="Optional worker dataset CSV (skips worker-level markdowns if missing).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(data_dir),
        help="Directory to write markdown diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    equilibrium_path = Path(args.equilibrium_path)
    params_path = Path(args.params_path)
    workers_path = Path(args.workers_path) if args.workers_path is not None else None
    out_dir = Path(args.out_dir)

    print("Implied Wage Markdown Computation (Data Environment Item 4)")
    print("=" * 70)
    print(f"Equilibrium data: {equilibrium_path}")
    print(f"Parameters:       {params_path}")
    if workers_path is not None:
        print(f"Worker data:      {workers_path}")
    print(f"Output directory: {out_dir}")

    if not equilibrium_path.exists():
        print(f"Error: equilibrium file {equilibrium_path} does not exist.")
        return 1
    if not params_path.exists():
        print(f"Error: parameter file {params_path} does not exist.")
        return 1

    ensure_directory(out_dir)

    # Read data
    equilibrium_df = pd.read_csv(equilibrium_path)
    params = read_parameters_csv(params_path)

    # Firm-level markdowns
    firm_markdowns = compute_firm_markdowns(equilibrium_df, params)
    firm_output = out_dir / "firm_markdowns.csv"
    firm_markdowns.to_csv(firm_output, index=False)
    print(f"[OK] Firm markdowns written to {firm_output}")

    # Plot firm markdowns
    firm_plot_path = out_dir / "firm_markdowns_vs_average_skill.png"
    plot_firm_markdowns(firm_markdowns, firm_plot_path)
    print(f"[OK] Firm markdown plot saved to {firm_plot_path}")

    markup_plot_path = out_dir / "markup_vs_cutoff.png"
    if plot_markup_vs_cutoff(firm_markdowns, markup_plot_path):
        print(f"[OK] Markup vs cutoff plot saved to {markup_plot_path}")
    else:
        print("[WARN] Elasticity columns not found; skipping markup vs cutoff plot.")

    idx_plot_path = out_dir / "firm_markdowns_vs_firm_index.png"
    if plot_markdowns_vs_firm_index(firm_markdowns, idx_plot_path):
        print(f"[OK] Markdown vs firm index plot saved to {idx_plot_path}")
    else:
        print("[WARN] Elasticity columns not found; skipping firm index plot.")

    # Worker-level markdowns (optional)
    if workers_path is not None and workers_path.exists():
        worker_df = pd.read_csv(workers_path)
        try:
            worker_markdowns = compute_worker_markdowns(worker_df, firm_markdowns)
        except ValueError as exc:
            print(f"[WARN] Skipping worker markdowns: {exc}")
        else:
            worker_output = out_dir / "worker_markdowns.csv"
            worker_markdowns.to_csv(worker_output, index=False)
            print(f"[OK] Worker markdowns written to {worker_output}")

            worker_plot_path = out_dir / "worker_markdowns_vs_skill.png"
            plot_worker_markdowns(worker_markdowns, worker_plot_path)
            print(f"[OK] Worker markdown plot saved to {worker_plot_path}")
    else:
        print("[WARN] Worker dataset not provided or missing; skipping worker-level markdowns.")

    print("Completed implied wage markdown diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
