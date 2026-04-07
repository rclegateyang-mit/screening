#!/usr/bin/env python3
"""GMM moments test: 6 moment specifications x 2 setups for (alpha, sigma_e).

Tests the moment conditions from docs/methods/gmm_moments.md:
  - Combined residual r^C (no structural residual, gamma0 cancels)
  - Production residual r^P (residual = ln A, needs demeaned instruments)
  - Screening FOC residual r^S (residual = -ln A, needs demeaned instruments)

Two setups:
  (a) True tilde_q, solver initialized at true (alpha, sigma_e)
  (b) MLE-estimated tilde_q, solver initialized at naive (alpha, sigma_e)

Usage::

    cd code/
    python -m screening.analysis.scaling.run_gmm_moments [--smoke] [--skip_data] [--skip_mle]
"""

from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = os.environ.get(_var, "1")
os.environ["XLA_FLAGS"] = os.environ.get(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false"
)
os.environ["JAX_ENABLE_X64"] = "1"

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRUE_PARAMS = {
    "tau": 0.4, "alpha": 0.2, "eta": 5.0,
    "gamma0": -7.0, "gamma1": 0.94,
    "sigma_e": 0.135, "mu_v": 10.84, "sigma_v": 0.25,
    "sigma_A": 0.37, "sigma_xi": 0.35, "rho_Axi": 0.44,
    "sigma_z1": 0.2, "sigma_z2": 0.175,
}
TRUE_TG = TRUE_PARAMS["gamma1"] / TRUE_PARAMS["sigma_e"]

PROJ_ROOT = Path(__file__).resolve().parents[3]  # .../code/
BASE_DIR = Path("/tmp/gmm_moments_test")


# ---------------------------------------------------------------------------
# Subprocess helper (from run_two_step.py)
# ---------------------------------------------------------------------------


def _run(cmd: List[str], label: str, cwd: Path | None = None, timeout: int = 7200,
         env_extra: Dict[str, str] | None = None):
    """Run subprocess, raise on failure."""
    print(f"  [{label}] {' '.join(str(c) for c in cmd[:6])}...")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        print(f"  FAILED ({label}):")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        raise RuntimeError(f"Subprocess failed: {label}")


# ---------------------------------------------------------------------------
# Phase 1: Data generation
# ---------------------------------------------------------------------------


def generate_data(M: int, N: int, J: int, seed: int = 12345):
    """Run 01_prep_data -> 02_solve_equilibrium -> 03_draw_workers."""
    d = BASE_DIR
    raw = d / "raw"
    clean = d / "clean"
    build = d / "build"

    py = sys.executable
    p = TRUE_PARAMS
    data_env = {"SCREENING_DATA_DIR": str(d)}

    label = f"M={M} N={N} J={J}"

    _run([
        py, "-m", "screening.simulate.01_prep_data",
        "--J", str(J), "--N_workers", str(N), "--M", str(M),
        "--seed", str(seed),
        "--out_dir", str(raw),
        "--tau", str(p["tau"]), "--alpha", str(p["alpha"]),
        "--eta", str(p["eta"]),
        "--gamma0", str(p["gamma0"]), "--gamma1", str(p["gamma1"]),
        "--sigma_e", str(p["sigma_e"]),
        "--mu_v", str(p["mu_v"]), "--sigma_v", str(p["sigma_v"]),
        "--sigma_A", str(p["sigma_A"]),
        "--sigma_xi", str(p["sigma_xi"]), "--rho_Axi", str(p["rho_Axi"]),
        "--sigma_z1", str(p["sigma_z1"]), "--sigma_z2", str(p["sigma_z2"]),
        "--quad_n_x", "50", "--quad_n_y", "50",
        "--conduct_mode", "1",
    ], label=f"prep {label}", cwd=PROJ_ROOT, env_extra=data_env)

    par = min(M, 30)
    _run([
        py, "-m", "screening.clean.02_solve_equilibrium",
        "--M", str(M), "--parallel_markets", str(par),
        "--firms_path", str(raw / "firms.csv"),
        "--support_path", str(raw / "support_points.csv"),
        "--params_path", str(raw / "parameters_effective.csv"),
        "--out_dir", str(clean),
        "--conduct_mode", "1", "--use_lsq", "--max_iter", "50000",
    ], label=f"equil {label}", cwd=PROJ_ROOT, env_extra=data_env,
       timeout=max(3600, M * 180))

    _run([
        py, "-m", "screening.build.03_draw_workers",
        "--M", str(M), "--seed", str(seed),
        "--params_path", str(raw / "parameters_effective.csv"),
        "--firms_path", str(clean / "equilibrium_firms.csv"),
        "--out_dir", str(build),
        "--drop_below_n", "5",
    ], label=f"draw {label}", cwd=PROJ_ROOT, env_extra=data_env)


# ---------------------------------------------------------------------------
# Phase 2: MLE
# ---------------------------------------------------------------------------


def run_mle(M: int, mpi_np: int):
    """Run distributed MLE via mpirun."""
    d = BASE_DIR
    est = d / "est"
    est.mkdir(parents=True, exist_ok=True)
    n_ranks = min(M, mpi_np)

    cmd = [
        "mpirun", "--oversubscribe", "-np", str(n_ranks),
        sys.executable, "-m", "screening.analysis.mle.run_distributed",
        "--firms_path", str(d / "clean" / "equilibrium_firms.csv"),
        "--workers_path", str(d / "build" / "workers_dataset.csv"),
        "--params_path", str(d / "raw" / "parameters_effective.csv"),
        "--out_dir", str(est), "--M", str(M),
        "--inner_maxiter", "500", "--inner_tol", "1e-7",
        "--outer_maxiter", "50", "--outer_tol", "1e-6",
    ]
    _run(cmd, label=f"MLE M={M}", cwd=PROJ_ROOT,
         env_extra={"SCREENING_DATA_DIR": str(d)},
         timeout=max(3600, M * 300))


# ---------------------------------------------------------------------------
# Phase 3: GMM moments test
# ---------------------------------------------------------------------------


def _load_setup_data(
    use_true_delta: bool,
) -> Tuple[List[dict], dict, float, float]:
    """Load market data for one setup, returning market_list, meta, tau, tg."""
    d = BASE_DIR
    firms_path = str(d / "clean" / "equilibrium_firms.csv")
    workers_path = str(d / "build" / "workers_dataset.csv")
    params_path = str(d / "raw" / "parameters_effective.csv")

    if use_true_delta:
        inner_df = None
        tau_fixed = TRUE_PARAMS["tau"]
        tg_fixed = TRUE_TG
    else:
        inner_df = pd.read_csv(d / "est" / "mle_distributed_inner_estimates.csv")
        with open(d / "est" / "mle_distributed_estimates.json") as f:
            mle = json.load(f)
        tau_fixed = mle["theta_G"][0]
        tg_fixed = mle["theta_G"][1]

    # Import and call the shared loader
    sys.path.insert(0, str(PROJ_ROOT))
    from screening.analysis.gmm.run_from_mle import load_market_data_with_inner
    market_list, meta = load_market_data_with_inner(
        firms_path, workers_path, params_path,
        inner_df, M_subset=None, use_true_delta=use_true_delta,
    )
    return market_list, meta, tau_fixed, tg_fixed


def _augment_markets(market_list: List[dict]):
    """Add z2_dm (within-market demeaned z2) to each market dict."""
    for md in market_list:
        md["z2_dm"] = md["z2"] - np.mean(md["z2"])


def _naive_alpha_sigma_e(market_list: List[dict], tg_hat: float) -> np.ndarray:
    """Compute naive (alpha, sigma_e) from wage regression + production 2SLS."""
    # Wage regression for sigma_e
    all_ln_w, all_v = [], []
    for md in market_list:
        mask = md["choice_idx"] > 0
        if np.any(mask):
            j_idx = md["choice_idx"][mask] - 1
            all_ln_w.append(np.log(np.maximum(md["w"][j_idx], 1e-300)))
            all_v.append(md["v"][mask])
    pooled_ln_w = np.concatenate(all_ln_w)
    pooled_v = np.concatenate(all_v)
    X = np.column_stack([np.ones(len(pooled_v)), pooled_v])
    beta, _, _, _ = np.linalg.lstsq(X, pooled_ln_w, rcond=None)
    sigma_e_init = float(np.sqrt(np.mean((pooled_ln_w - X @ beta) ** 2)))
    sigma_e_init = max(sigma_e_init, 0.01)

    # Production 2SLS for alpha
    gamma0_hat = float(beta[0])
    gamma1_hat = sigma_e_init * tg_hat
    all_ln_R, all_ln_QL, all_Z = [], [], []
    for md in market_list:
        Q_j = np.zeros(md["J"])
        for j in range(md["J"]):
            mask_j = md["choice_idx"] == (j + 1)
            if np.any(mask_j):
                Q_j[j] = np.mean(np.exp(gamma0_hat + gamma1_hat * md["v"][mask_j]))
        Q_j[Q_j == 0] = np.median(Q_j[Q_j > 0]) if np.any(Q_j > 0) else 1.0
        QL = Q_j * np.maximum(md["L"], 1e-300)
        all_ln_R.append(np.log(np.maximum(md["R"], 1e-300)))
        all_ln_QL.append(np.log(np.maximum(QL, 1e-300)))
        all_Z.append(np.column_stack([np.ones(md["J"]), md["z2"]]))
    pooled_ln_R = np.concatenate(all_ln_R)
    pooled_ln_QL = np.concatenate(all_ln_QL)
    pooled_Z = np.vstack(all_Z)
    coef_s1, _, _, _ = np.linalg.lstsq(pooled_Z, pooled_ln_QL, rcond=None)
    ln_QL_hat = pooled_Z @ coef_s1
    X_s2 = np.column_stack([np.ones(len(pooled_ln_R)), ln_QL_hat])
    coef_s2, _, _, _ = np.linalg.lstsq(X_s2, pooled_ln_R, rcond=None)
    alpha_init = float(np.clip(1.0 - coef_s2[1], 0.05, 0.95))

    return np.array([alpha_init, sigma_e_init], dtype=np.float64)


# ---------------------------------------------------------------------------
# Moment specifications
# ---------------------------------------------------------------------------

# Each spec: name, label, list of (equation, instruments_fn, demeaned_flag)
# instruments_fn(md) -> (J, K_eq) instrument matrix for that equation in market md

SPECS = [
    {
        "name": "C_internal",
        "label": "Combined, internal instruments",
        "moments": [
            ("C", lambda md: np.column_stack([
                np.ones(md["J"]), md["L"], md["R"], md["w"],
                md["v_bar"], md["tilde_q_hat"],
            ])),
        ],
    },
    {
        "name": "C_amenity",
        "label": "Combined, amenity shock",
        "moments": [
            ("C", lambda md: np.column_stack([np.ones(md["J"]), md["z2"]])),
        ],
    },
    {
        "name": "CS_amenity",
        "label": "Combined + screening, amenity",
        "moments": [
            ("C", lambda md: np.ones((md["J"], 1))),
            ("S", lambda md: md["z2_dm"].reshape(-1, 1)),
        ],
    },
    {
        "name": "CS_internal_amenity",
        "label": "Combined + screening, internal + amenity",
        "moments": [
            ("C", lambda md: np.column_stack([
                np.ones(md["J"]), md["L"], md["R"], md["w"],
                md["v_bar"], md["tilde_q_hat"],
            ])),
            ("S", lambda md: md["z2_dm"].reshape(-1, 1)),
        ],
    },
    {
        "name": "CP_amenity",
        "label": "Combined + production, amenity",
        "moments": [
            ("C", lambda md: np.ones((md["J"], 1))),
            ("P", lambda md: md["z2_dm"].reshape(-1, 1)),
        ],
    },
    {
        "name": "CP_internal_amenity",
        "label": "Combined + production, internal + amenity",
        "moments": [
            ("C", lambda md: np.column_stack([
                np.ones(md["J"]), md["L"], md["R"], md["w"],
                md["v_bar"], md["tilde_q_hat"],
            ])),
            ("P", lambda md: md["z2_dm"].reshape(-1, 1)),
        ],
    },
]


def _n_moments(spec: dict) -> int:
    """Count total moment conditions for a specification (use first market)."""
    # Can't evaluate lambdas without data; sum the static column counts
    # from the spec structure. We compute this dynamically at runtime.
    return -1  # placeholder; computed from instrument matrices


def _run_one_spec(
    spec: dict,
    market_list: List[dict],
    tau_fixed: float,
    tg_fixed: float,
    theta0: np.ndarray,
) -> dict:
    """Run GMM optimization for one moment specification.

    Args:
        spec: moment specification dict
        market_list: list of per-market data dicts
        tau_fixed: fixed tau value
        tg_fixed: fixed tilde_gamma value
        theta0: (2,) initial [alpha, sigma_e]

    Returns:
        dict with results (alpha_hat, sigma_e_hat, time, iters, etc.)
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from screening.analysis.lib.model_components import (
        compute_tilde_Q_M, compute_residual_C, compute_residual_P,
        compute_residual_S,
    )

    # Build per-market instrument matrices and compute total K
    per_market_instruments = []
    for md in market_list:
        inst_blocks = []
        for eq_name, inst_fn in spec["moments"]:
            inst_blocks.append(inst_fn(md))
        Z_m = np.hstack(inst_blocks)  # (J_m, K)
        per_market_instruments.append(Z_m)

    K = per_market_instruments[0].shape[1]
    J_total = sum(md["J"] for md in market_list)

    # Compute 2SLS weighting matrix: W = (Z'Z / J_total)^{-1}
    if K > 2:
        ZtZ = np.zeros((K, K), dtype=np.float64)
        for Z_m in per_market_instruments:
            ZtZ += Z_m.T @ Z_m
        ZtZ /= J_total
        try:
            W = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            W = np.eye(K, dtype=np.float64)
    else:
        W = np.eye(K, dtype=np.float64)

    # Determine which residuals are needed
    eq_names = set(eq for eq, _ in spec["moments"])
    need_C = "C" in eq_names
    need_P = "P" in eq_names
    need_S = "S" in eq_names

    # Precompute column ranges for each equation's instruments
    eq_col_ranges = []
    col_start = 0
    for eq_name, inst_fn in spec["moments"]:
        K_eq = per_market_instruments[0].shape[1]  # will recalculate
        # Get from the actual instrument block
        inst_block = inst_fn(market_list[0])
        K_eq = inst_block.shape[1]
        eq_col_ranges.append((eq_name, col_start, col_start + K_eq))
        col_start += K_eq

    # JIT the per-market residual computation
    @jax.jit
    def _market_residuals(theta, tau, tg, delta, tq, v, D, cidx, w, R, L):
        alpha, sigma_e = theta[0], theta[1]
        tilde_Q = compute_tilde_Q_M(sigma_e, tg, tau, delta, tq, v, D, cidx)
        r_C = compute_residual_C(sigma_e, alpha, tq, tilde_Q, w, R, L)
        r_P = compute_residual_P(alpha, tilde_Q, R, L)
        r_S = compute_residual_S(sigma_e, alpha, tq, tilde_Q, w, L)
        return r_C, r_P, r_S

    # Convert market data to JAX arrays (once)
    tau_j = jnp.float64(tau_fixed)
    tg_j = jnp.float64(tg_fixed)
    jax_markets = []
    for md in market_list:
        jax_markets.append({
            "delta": jnp.array(md["delta_hat"]),
            "tq": jnp.array(md["tilde_q_hat"]),
            "v": jnp.array(md["v"]),
            "D": jnp.array(md["D"]),
            "cidx": jnp.array(md["choice_idx"]),
            "w": jnp.array(md["w"]),
            "R": jnp.array(md["R"]),
            "L": jnp.array(md["L"]),
        })

    # Map equation name to residual index
    eq_idx = {"C": 0, "P": 1, "S": 2}

    def gmm_obj(params_unc):
        """GMM objective in unconstrained space."""
        alpha_v = 0.01 + 0.98 / (1.0 + np.exp(-params_unc[0]))
        sigma_e_v = np.exp(params_unc[1])
        theta_jax = jnp.array([alpha_v, sigma_e_v])

        g_bar = np.zeros(K, dtype=np.float64)
        for i, (md, jm, Z_m) in enumerate(zip(
            market_list, jax_markets, per_market_instruments
        )):
            r_C, r_P, r_S = _market_residuals(
                theta_jax, tau_j, tg_j,
                jm["delta"], jm["tq"], jm["v"], jm["D"], jm["cidx"],
                jm["w"], jm["R"], jm["L"],
            )
            residuals = {
                "C": np.asarray(r_C),
                "P": np.asarray(r_P),
                "S": np.asarray(r_S),
            }

            # Build moment vector for this market
            g_m = np.zeros(K, dtype=np.float64)
            for eq_name, c_lo, c_hi in eq_col_ranges:
                Z_eq = Z_m[:, c_lo:c_hi]  # (J, K_eq)
                r_eq = residuals[eq_name]  # (J,)
                g_m[c_lo:c_hi] = Z_eq.T @ r_eq / md["J"]

            g_bar += md["omega"] * g_m

        Q = float(g_bar @ W @ g_bar)
        return Q

    # Transform init to unconstrained
    from scipy.special import logit as sp_logit
    p0 = np.array([
        sp_logit(np.clip((theta0[0] - 0.01) / 0.98, 1e-6, 1 - 1e-6)),
        np.log(max(theta0[1], 1e-6)),
    ])

    from scipy.optimize import minimize as sp_minimize

    bounds = [(None, None), (np.log(0.01), None)]

    t0 = time.perf_counter()
    result = sp_minimize(
        gmm_obj, p0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-8},
    )
    elapsed = time.perf_counter() - t0

    alpha_hat = 0.01 + 0.98 / (1.0 + np.exp(-result.x[0]))
    sigma_e_hat = np.exp(result.x[1])

    # Final moments at solution
    theta_final_unc = result.x
    Q_final = result.fun

    # J-statistic for overidentification
    df = K - 2
    if df > 0:
        J_stat = J_total * Q_final
        from scipy.stats import chi2
        p_value = float(1.0 - chi2.cdf(J_stat, df=df))
    else:
        J_stat = np.nan
        p_value = np.nan

    return {
        "spec_name": spec["name"],
        "spec_label": spec["label"],
        "K": K,
        "df": df,
        "alpha_hat": float(alpha_hat),
        "sigma_e_hat": float(sigma_e_hat),
        "alpha_init": float(theta0[0]),
        "sigma_e_init": float(theta0[1]),
        "Q_final": float(Q_final),
        "J_stat": float(J_stat) if df > 0 else None,
        "p_value": float(p_value) if df > 0 else None,
        "converged": bool(result.success),
        "n_iter": int(result.nit),
        "n_fev": int(result.nfev),
        "time_s": elapsed,
    }


def run_moments_test() -> List[dict]:
    """Run all 6 specs x 2 setups."""
    results = []

    for setup_name, use_true in [("true", True), ("estimated", False)]:
        print(f"\n{'='*60}")
        print(f"Setup: {setup_name} tilde_q")
        print(f"{'='*60}")

        # Load data
        t0 = time.perf_counter()
        try:
            market_list, meta, tau_fixed, tg_fixed = _load_setup_data(use_true)
        except Exception as e:
            print(f"  Failed to load data for setup '{setup_name}': {e}")
            if not use_true:
                print("  (MLE results may not exist yet — run without --skip_mle)")
            continue

        _augment_markets(market_list)

        tp = meta["true_params"]
        M = meta["M"]
        J_total = meta["total_J"]
        load_time = time.perf_counter() - t0
        print(f"  Loaded M={M}, J_total={J_total} ({load_time:.1f}s)")
        print(f"  tau={tau_fixed:.6f}, tg={tg_fixed:.6f}")

        # Initial values
        if use_true:
            theta0 = np.array([tp["alpha"], tp["sigma_e"]], dtype=np.float64)
            print(f"  Init (true): alpha={theta0[0]}, sigma_e={theta0[1]}")
        else:
            theta0 = _naive_alpha_sigma_e(market_list, tg_fixed)
            print(f"  Init (naive): alpha={theta0[0]:.4f}, sigma_e={theta0[1]:.4f}")

        # Run each specification
        for i, spec in enumerate(SPECS):
            print(f"\n  [{i+1}/6] {spec['name']}: {spec['label']}")
            try:
                res = _run_one_spec(spec, market_list, tau_fixed, tg_fixed, theta0)
                res["setup"] = setup_name
                res["M"] = M
                res["J_total"] = J_total

                cvg_str = "Y" if res["converged"] else "N"
                j_str = f"J={res['J_stat']:.2f} p={res['p_value']:.3f}" if res["J_stat"] is not None else "N/A"
                print(f"    alpha={res['alpha_hat']:.6f} (err={res['alpha_hat']-tp['alpha']:+.6f})"
                      f"  sigma_e={res['sigma_e_hat']:.6f} (err={res['sigma_e_hat']-tp['sigma_e']:+.6f})")
                print(f"    cvg={cvg_str}  iters={res['n_iter']}  Q={res['Q_final']:.2e}"
                      f"  {j_str}  ({res['time_s']:.1f}s)")

                results.append(res)
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback; traceback.print_exc()
                results.append({
                    "setup": setup_name, "spec_name": spec["name"],
                    "spec_label": spec["label"],
                    "K": 0, "df": 0,
                    "alpha_hat": np.nan, "sigma_e_hat": np.nan,
                    "alpha_init": float(theta0[0]), "sigma_e_init": float(theta0[1]),
                    "Q_final": np.nan, "J_stat": None, "p_value": None,
                    "converged": False, "n_iter": 0, "n_fev": 0,
                    "time_s": 0, "M": M, "J_total": J_total,
                })

    return results


# ---------------------------------------------------------------------------
# Phase 4: Results
# ---------------------------------------------------------------------------


def print_results(results: List[dict]):
    """Print markdown table of results."""
    if not results:
        print("\nNo results to display.")
        return

    print(f"\n{'='*60}")
    print("GMM Moments Test Results")
    print(f"{'='*60}")
    print(f"\nTrue: alpha={TRUE_PARAMS['alpha']}, sigma_e={TRUE_PARAMS['sigma_e']}")

    hdr = (
        "| Setup | Spec | K | time_s | iters | cvg "
        "| alpha_hat | alpha_err | se_hat | se_err "
        "| Q | J_stat | p_val |"
    )
    sep = "|" + "|".join(["---"] * 13) + "|"
    print(f"\n{hdr}")
    print(sep)

    for r in results:
        alpha_err = r["alpha_hat"] - TRUE_PARAMS["alpha"]
        se_err = r["sigma_e_hat"] - TRUE_PARAMS["sigma_e"]
        j_str = f"{r['J_stat']:.2f}" if r.get("J_stat") is not None else "N/A"
        p_str = f"{r['p_value']:.3f}" if r.get("p_value") is not None else "N/A"
        cvg = "Y" if r["converged"] else "N"

        cells = [
            r["setup"],
            r["spec_name"],
            str(r["K"]),
            f"{r['time_s']:.1f}",
            str(r["n_iter"]),
            cvg,
            f"{r['alpha_hat']:.6f}",
            f"{alpha_err:+.6f}",
            f"{r['sigma_e_hat']:.6f}",
            f"{se_err:+.6f}",
            f"{r['Q_final']:.2e}",
            j_str,
            p_str,
        ]
        print("| " + " | ".join(cells) + " |")


def save_results(results: List[dict], out_path: Path):
    """Save results to JSON."""
    # Convert NaN to None for JSON
    clean = []
    for r in results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                cr[k] = None
            else:
                cr[k] = v
        clean.append(cr)

    summary = {
        "true_params": {
            "alpha": TRUE_PARAMS["alpha"],
            "sigma_e": TRUE_PARAMS["sigma_e"],
            "tau": TRUE_PARAMS["tau"],
            "gamma0": TRUE_PARAMS["gamma0"],
            "gamma1": TRUE_PARAMS["gamma1"],
        },
        "specs": [s["name"] for s in SPECS],
        "results": clean,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global BASE_DIR

    parser = argparse.ArgumentParser(
        description="GMM moments test: 6 specs x 2 setups for (alpha, sigma_e)",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test with M=5")
    parser.add_argument("--skip_data", action="store_true",
                        help="Skip data generation (assume data exists)")
    parser.add_argument("--skip_mle", action="store_true",
                        help="Skip MLE (assume estimates exist)")
    parser.add_argument("--base_dir", type=str, default=str(BASE_DIR))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--N", type=int, default=2500)
    parser.add_argument("--J", type=int, default=100)
    parser.add_argument("--mpi_np", type=int, default=20)
    args = parser.parse_args()

    BASE_DIR = Path(args.base_dir)
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    M = 5 if args.smoke else args.M
    N = args.N
    J = args.J

    total_t0 = time.perf_counter()
    print(f"GMM Moments Test: M={M}, N={N}, J={J}")
    print(f"Base dir: {BASE_DIR}")

    # Phase 1: Data generation
    if not args.skip_data:
        print(f"\n{'='*60}")
        print("Phase 1: Data generation")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        try:
            generate_data(M, N, J, seed=args.seed)
            print(f"  Done ({time.perf_counter() - t0:.0f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            return

    # Phase 2: MLE
    if not args.skip_mle:
        print(f"\n{'='*60}")
        print("Phase 2: Distributed MLE")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        try:
            run_mle(M, mpi_np=args.mpi_np)
            print(f"  Done ({time.perf_counter() - t0:.0f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            return

    # Phase 3: GMM moments test
    print(f"\n{'='*60}")
    print("Phase 3: GMM moments test (6 specs x 2 setups)")
    print(f"{'='*60}")
    results = run_moments_test()

    # Phase 4: Results
    print_results(results)
    save_results(results, BASE_DIR / "gmm_moments_summary.json")

    total_time = time.perf_counter() - total_t0
    print(f"\nTotal time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
