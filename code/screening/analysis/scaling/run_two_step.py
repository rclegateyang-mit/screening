#!/usr/bin/env python3
"""Two-step scaling test: micro MLE + structural GMM.

Step 1 — Distributed MLE for (tau, tilde_gamma, delta, tilde_q)
Step 2a — Linear IV (2SLS) for eta
Step 2b — GMM for (alpha, sigma_e) using A-free moment
Step 2c — Overidentification test

Usage::

    python -m screening.analysis.scaling.run_two_step [--smoke] [--skip_datagen] [--skip_mle]
"""

from __future__ import annotations

import argparse
import json
import os
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
BASE_DIR = Path("/tmp/two_step_scaling")

M_SCALING = [
    {"M": 10,  "N": 2500,  "J": 100},
    {"M": 50,  "N": 2500,  "J": 100},
    {"M": 100, "N": 2500,  "J": 100},
    {"M": 200, "N": 2500,  "J": 100},
]
NJ_SCALING = [
    {"M": 100, "N": 1000,  "J": 100},
    {"M": 100, "N": 2500,  "J": 100},  # shared
    {"M": 100, "N": 5000,  "J": 100},
    {"M": 100, "N": 10000, "J": 100},
]

MAX_CORES = 72  # 75% of 96


def _unique_configs(configs: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for c in configs:
        key = (c["M"], c["N"], c["J"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _cfg_dir(cfg: dict) -> Path:
    return BASE_DIR / f"M{cfg['M']}_N{cfg['N']}"


def _cfg_label(cfg: dict) -> str:
    return f"M={cfg['M']} N={cfg['N']} J={cfg['J']}"


# ---------------------------------------------------------------------------
# Phase 1: Data generation
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


def generate_data(cfg: dict, seed: int = 12345):
    """Run 01_prep_data → 02_solve_equilibrium → 03_draw_workers."""
    d = _cfg_dir(cfg)
    raw = d / "raw"
    clean = d / "clean"
    build = d / "build"

    py = sys.executable
    M, N, J = cfg["M"], cfg["N"], cfg["J"]
    p = TRUE_PARAMS
    # Set SCREENING_DATA_DIR so scripts resolve per-market paths correctly
    data_env = {"SCREENING_DATA_DIR": str(d)}

    # --- Stage 1: prep ---
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
    ], label=f"prep {_cfg_label(cfg)}", cwd=PROJ_ROOT, env_extra=data_env)

    # --- Stage 2: equilibrium ---
    par = min(M, 48)
    _run([
        py, "-m", "screening.clean.02_solve_equilibrium",
        "--M", str(M), "--parallel_markets", str(par),
        "--firms_path", str(raw / "firms.csv"),
        "--support_path", str(raw / "support_points.csv"),
        "--params_path", str(raw / "parameters_effective.csv"),
        "--out_dir", str(clean),
        "--conduct_mode", "1", "--use_lsq", "--max_iter", "50000",
    ], label=f"equil {_cfg_label(cfg)}", cwd=PROJ_ROOT, env_extra=data_env,
       timeout=max(3600, M * 180))

    # --- Stage 3: draw workers ---
    _run([
        py, "-m", "screening.build.03_draw_workers",
        "--M", str(M), "--seed", str(seed),
        "--params_path", str(raw / "parameters_effective.csv"),
        "--firms_path", str(clean / "equilibrium_firms.csv"),
        "--out_dir", str(build),
        "--drop_below_n", "5",
    ], label=f"draw {_cfg_label(cfg)}", cwd=PROJ_ROOT, env_extra=data_env)


# ---------------------------------------------------------------------------
# Phase 2: MLE
# ---------------------------------------------------------------------------


def run_mle(cfg: dict):
    """Run distributed MLE via mpirun."""
    d = _cfg_dir(cfg)
    est = d / "est"
    est.mkdir(parents=True, exist_ok=True)
    M = cfg["M"]
    n_ranks = min(M, 48)

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
    _run(cmd, label=f"MLE {_cfg_label(cfg)}", cwd=PROJ_ROOT,
         env_extra={"SCREENING_DATA_DIR": str(d)},
         timeout=max(3600, M * 300))


# ---------------------------------------------------------------------------
# Phase 3: GMM (inline)
# ---------------------------------------------------------------------------


def _load_market_arrays(
    firms_df: pd.DataFrame, workers_df: pd.DataFrame,
    M_markets: List[int],
) -> List[Dict[str, np.ndarray]]:
    """Load per-market arrays needed for GMM."""
    from scipy.spatial.distance import cdist

    market_arrays = []
    for mid in M_markets:
        fdf = firms_df[firms_df["market_id"] == mid].sort_values("firm_id")
        wdf = workers_df[workers_df["market_id"] == mid]

        J_m = len(fdf)
        v = wdf["x_skill"].values.astype(np.float64)
        choice = wdf["chosen_firm"].values.astype(np.int32)
        w = fdf["w"].values.astype(np.float64)
        R = fdf["Y"].values.astype(np.float64)

        loc_f = fdf[["x", "y"]].values
        loc_w = np.column_stack([wdf["ell_x"].values, wdf["ell_y"].values])
        D = cdist(loc_w, loc_f, metric="euclidean").astype(np.float64)

        counts = np.bincount(choice, minlength=J_m + 1).astype(np.float64)
        L = counts[1:]

        z1 = fdf["z1"].values.astype(np.float64) if "z1" in fdf.columns else np.zeros(J_m)
        z2 = fdf["z2"].values.astype(np.float64) if "z2" in fdf.columns else np.zeros(J_m)

        market_arrays.append({
            "v": v, "choice": choice, "D": D, "w": w, "R": R, "L": L,
            "z1": z1, "z2": z2, "J": J_m, "N": len(wdf),
        })
    return market_arrays


def _compute_tQ_all(
    sigma_e_val: float, tg_hat: float, tau_hat: float,
    inner_df: pd.DataFrame, market_arrays: List[dict],
    markets: List[int], compute_tQ_jit,
) -> np.ndarray:
    """Compute tilde_Q for all firms across markets."""
    import jax.numpy as jnp

    tQ_list = []
    for i, mid in enumerate(markets):
        idf = inner_df[inner_df["market_id"] == mid].sort_values("firm_j")
        ma = market_arrays[i]
        delta_hat = jnp.asarray(idf["delta_hat"].values, dtype=jnp.float64)
        tq_hat = jnp.asarray(idf["tilde_q_hat"].values, dtype=jnp.float64)
        v = jnp.asarray(ma["v"], dtype=jnp.float64)
        D = jnp.asarray(ma["D"], dtype=jnp.float64)
        choice = jnp.asarray(ma["choice"], dtype=jnp.int32)

        tQ = compute_tQ_jit(
            jnp.float64(sigma_e_val), jnp.float64(tg_hat),
            jnp.float64(tau_hat), delta_hat, tq_hat, v, D, choice,
        )
        tQ_list.append(np.asarray(tQ, dtype=np.float64))
    return np.concatenate(tQ_list)


def run_gmm(cfg: dict) -> dict:
    """Run GMM step 2a/2b/2c on one config's MLE output."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from screening.analysis.lib.model_components import compute_tilde_Q_M

    d = _cfg_dir(cfg)
    firms_df = pd.read_csv(d / "clean" / "equilibrium_firms.csv")
    workers_df = pd.read_csv(d / "build" / "workers_dataset.csv")
    inner_df = pd.read_csv(d / "est" / "mle_distributed_inner_estimates.csv")

    with open(d / "est" / "mle_distributed_estimates.json") as f:
        mle = json.load(f)
    tau_hat = mle["theta_G"][0]
    tg_hat = mle["theta_G"][1]

    firms_df = firms_df.sort_values(["market_id", "firm_id"]).reset_index(drop=True)
    workers_df = workers_df.sort_values("market_id").reset_index(drop=True)
    markets = sorted(firms_df["market_id"].unique())

    market_arrays = _load_market_arrays(firms_df, workers_df, markets)

    # JIT compile compute_tilde_Q_M (compilation keyed on J)
    compute_tQ_jit = jax.jit(compute_tilde_Q_M)

    # ===== Step 2a: 2SLS for eta (within-market demeaned) =====
    all_delta_dm, all_ln_w_dm, all_z1_dm = [], [], []
    for i, mid in enumerate(markets):
        idf = inner_df[inner_df["market_id"] == mid].sort_values("firm_j")
        ma = market_arrays[i]
        delta_m = idf["delta_hat"].values
        ln_w_m = np.log(np.maximum(ma["w"], 1e-300))
        z1_m = ma["z1"]
        # Demean within market to use only cross-firm variation
        all_delta_dm.append(delta_m - delta_m.mean())
        all_ln_w_dm.append(ln_w_m - ln_w_m.mean())
        all_z1_dm.append(z1_m - z1_m.mean())

    delta_pool = np.concatenate(all_delta_dm)
    ln_w_pool = np.concatenate(all_ln_w_dm)
    z1_pool_dm = np.concatenate(all_z1_dm)
    J_total = len(delta_pool)

    # 2SLS (no intercept — already demeaned): delta_dm = eta * ln_w_dm + xi_dm
    # Stage 1: ln_w_dm on z1_dm
    coef1 = float(np.dot(z1_pool_dm, ln_w_pool) / np.dot(z1_pool_dm, z1_pool_dm))
    ln_w_hat = coef1 * z1_pool_dm
    # Stage 2: delta_dm on ln_w_hat
    eta_hat = float(np.dot(ln_w_hat, delta_pool) / np.dot(ln_w_hat, ln_w_hat))
    eta_const = 0.0  # absorbed by market FE

    # Naive init for eta (same 2SLS, for reporting)
    eta_init = eta_hat  # 2SLS IS the estimator, init = estimate

    # ===== Step 2b: GMM for (alpha, sigma_e) =====
    # Naive inits from wage regression
    all_ln_w_matched, all_v_matched = [], []
    for i, mid in enumerate(markets):
        ma = market_arrays[i]
        mask = ma["choice"] > 0
        if np.any(mask):
            j_idx = ma["choice"][mask] - 1
            all_ln_w_matched.append(np.log(np.maximum(ma["w"][j_idx], 1e-300)))
            all_v_matched.append(ma["v"][mask])
    pooled_ln_w = np.concatenate(all_ln_w_matched)
    pooled_v = np.concatenate(all_v_matched)
    X_ols = np.column_stack([np.ones(len(pooled_v)), pooled_v])
    beta_ols, _, _, _ = np.linalg.lstsq(X_ols, pooled_ln_w, rcond=None)
    sigma_e_init = float(np.sqrt(np.mean((pooled_ln_w - X_ols @ beta_ols) ** 2)))
    sigma_e_init = max(sigma_e_init, 0.01)

    # Naive alpha from production 2SLS
    all_ln_R, all_ln_QL, all_Z2 = [], [], []
    gamma0_hat = float(beta_ols[0])
    gamma1_hat = sigma_e_init * tg_hat
    for i, mid in enumerate(markets):
        ma = market_arrays[i]
        idf = inner_df[inner_df["market_id"] == mid].sort_values("firm_j")
        # Approximate Q from worker skill averages
        choice_np = ma["choice"]
        v_np = ma["v"]
        Q_j = np.zeros(ma["J"])
        for j in range(ma["J"]):
            mask_j = choice_np == (j + 1)
            if np.any(mask_j):
                Q_j[j] = np.mean(np.exp(gamma0_hat + gamma1_hat * v_np[mask_j]))
        Q_j[Q_j == 0] = np.median(Q_j[Q_j > 0]) if np.any(Q_j > 0) else 1.0
        QL = Q_j * np.maximum(ma["L"], 1e-300)
        all_ln_R.append(np.log(np.maximum(ma["R"], 1e-300)))
        all_ln_QL.append(np.log(np.maximum(QL, 1e-300)))
        all_Z2.append(np.column_stack([np.ones(ma["J"]), ma["z2"]]))

    pooled_ln_R = np.concatenate(all_ln_R)
    pooled_ln_QL = np.concatenate(all_ln_QL)
    pooled_Z2 = np.vstack(all_Z2)
    coef_s1, _, _, _ = np.linalg.lstsq(pooled_Z2, pooled_ln_QL, rcond=None)
    ln_QL_hat = pooled_Z2 @ coef_s1
    X_s2 = np.column_stack([np.ones(J_total), ln_QL_hat])
    coef_s2, _, _, _ = np.linalg.lstsq(X_s2, pooled_ln_R, rcond=None)
    alpha_init = float(1.0 - coef_s2[1])
    alpha_init = np.clip(alpha_init, 0.05, 0.95)

    # Pool tq and build instruments for GMM
    all_tq, all_w_firm, all_L_firm, all_R_firm, all_z2_firm = [], [], [], [], []
    for i, mid in enumerate(markets):
        idf = inner_df[inner_df["market_id"] == mid].sort_values("firm_j")
        ma = market_arrays[i]
        all_tq.append(idf["tilde_q_hat"].values)
        all_w_firm.append(ma["w"])
        all_L_firm.append(ma["L"])
        all_R_firm.append(ma["R"])
        all_z2_firm.append(ma["z2"])

    tq_pool = np.concatenate(all_tq)
    w_pool = np.concatenate(all_w_firm)
    L_pool = np.concatenate(all_L_firm)
    R_pool = np.concatenate(all_R_firm)
    z2_pool = np.concatenate(all_z2_firm)

    # GMM instruments: Z = [1, tq - mean(tq)]
    tq_dm = tq_pool - np.mean(tq_pool)
    Z_gmm = np.column_stack([np.ones(J_total), tq_dm])
    W_gmm = np.linalg.inv(Z_gmm.T @ Z_gmm / J_total)

    def gmm_obj(params_unc):
        """GMM objective in unconstrained space."""
        # Transform: alpha = sigmoid(p[0]), sigma_e = exp(p[1])
        alpha_v = 0.01 + 0.98 / (1.0 + np.exp(-params_unc[0]))
        sigma_e_v = np.exp(params_unc[1])

        tQ_all = _compute_tQ_all(
            sigma_e_v, tg_hat, tau_hat, inner_df,
            market_arrays, markets, compute_tQ_jit,
        )
        ln_tQ = np.log(np.maximum(tQ_all, 1e-300))
        ln_w = np.log(np.maximum(w_pool, 1e-300))
        ln_L = np.log(np.maximum(L_pool, 1e-300))
        ln_R = np.log(np.maximum(R_pool, 1e-300))

        # A-free moment: sigma_e*tq - ln(tQ) - ln(w) - ln(L) + ln(1-alpha) + ln(R)
        m3 = sigma_e_v * tq_pool - ln_tQ - ln_w - ln_L + np.log(1.0 - alpha_v) + ln_R

        g = Z_gmm.T @ m3 / J_total  # (2,)
        Q = float(g @ W_gmm @ g)
        return Q

    # Transform init to unconstrained
    from scipy.special import logit as sp_logit
    p0 = np.array([
        sp_logit(np.clip((alpha_init - 0.01) / 0.98, 1e-6, 1 - 1e-6)),
        np.log(max(sigma_e_init, 1e-6)),
    ])

    from scipy.optimize import minimize as sp_minimize
    # Bounds: alpha unconstrained (sigmoid), sigma_e >= 0.01 (exp)
    bounds_gmm = [(None, None), (np.log(0.01), None)]

    gmm_t0 = time.perf_counter()
    gmm_result = sp_minimize(
        gmm_obj, p0, method="L-BFGS-B", bounds=bounds_gmm,
        options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-8},
    )
    gmm_time = time.perf_counter() - gmm_t0

    alpha_hat = 0.01 + 0.98 / (1.0 + np.exp(-gmm_result.x[0]))
    sigma_e_hat = np.exp(gmm_result.x[1])

    # ===== Step 2c: Overidentification test =====
    # Compute tQ at estimated sigma_e
    tQ_final = _compute_tQ_all(
        sigma_e_hat, tg_hat, tau_hat, inner_df,
        market_arrays, markets, compute_tQ_jit,
    )
    ln_tQ = np.log(np.maximum(tQ_final, 1e-300))
    ln_w = np.log(np.maximum(w_pool, 1e-300))
    ln_L = np.log(np.maximum(L_pool, 1e-300))
    ln_R = np.log(np.maximum(R_pool, 1e-300))

    # ln(A)^prod = ln(R) - (1-alpha)*ln(tQ) - (1-alpha)*ln(L)  (+ constants)
    ln_A_prod = ln_R - (1 - alpha_hat) * ln_tQ - (1 - alpha_hat) * ln_L

    # ln(A)^foc = ln(w) - sigma_e*tq + alpha*ln(tQ) + alpha*ln(L) - ln(1-alpha)
    ln_A_foc = (ln_w - sigma_e_hat * tq_pool
                + alpha_hat * ln_tQ + alpha_hat * ln_L
                - np.log(1.0 - alpha_hat))

    # z2 demeaned for overid test
    z2_dm = z2_pool - np.mean(z2_pool)

    def _t_stat(z, ln_A):
        prod = z * ln_A
        m = np.mean(prod)
        s = np.std(prod, ddof=1)
        return float(np.sqrt(J_total) * m / max(s, 1e-300))

    t_prod = _t_stat(z2_dm, ln_A_prod)
    t_foc = _t_stat(z2_dm, ln_A_foc)

    return {
        "eta_hat": eta_hat, "eta_init": eta_init, "eta_const": eta_const,
        "alpha_hat": alpha_hat, "alpha_init": alpha_init,
        "sigma_e_hat": sigma_e_hat, "sigma_e_init": sigma_e_init,
        "overid_t_prod": t_prod, "overid_t_foc": t_foc,
        "gmm_converged": gmm_result.success,
        "gmm_nit": gmm_result.nit,
        "gmm_obj": float(gmm_result.fun),
        "gmm_time_s": gmm_time,
    }


# ---------------------------------------------------------------------------
# Phase 4: Collect and output
# ---------------------------------------------------------------------------


def collect_one(cfg: dict, gmm: dict) -> dict:
    """Collect MLE + GMM results for one config."""
    d = _cfg_dir(cfg)
    with open(d / "est" / "mle_distributed_estimates.json") as f:
        mle = json.load(f)

    rec = mle.get("recovery", {})
    hist = mle.get("history", [])
    sci = mle.get("scipy_result", {})
    n_outer = mle.get("n_outer_iters", 0)
    max_outer = 50  # from CLI

    # MLE init values from naive init (recompute would be expensive; use first history entry)
    tau_init = rec.get("tau_hat", mle["theta_G"][0])  # fallback
    tg_init = rec.get("tg_hat", mle["theta_G"][1])
    # Better: read from mle json if available
    # The naive init is printed but not stored — approximate from true values
    # Actually, let's re-derive from the inner_df init

    n_conv = hist[-1]["inner_converged"] if hist else 0
    n_markets = mle.get("M", cfg["M"])

    return {
        "M": cfg["M"], "N": cfg["N"], "J": cfg["J"],
        "NJ": cfg["N"] // cfg["J"],
        "tau_true": TRUE_PARAMS["tau"],
        "tau_hat": mle["theta_G"][0],
        "tg_true": TRUE_TG,
        "tg_hat": mle["theta_G"][1],
        "delta_corr": rec.get("delta_corr", 0),
        "delta_rmse": rec.get("delta_rmse", 0),
        "tq_corr": rec.get("tq_corr", 0),
        "tq_rmse": rec.get("tq_rmse", 0),
        "eta_true": TRUE_PARAMS["eta"],
        "eta_hat": gmm["eta_hat"],
        "alpha_true": TRUE_PARAMS["alpha"],
        "alpha_hat": gmm["alpha_hat"],
        "alpha_init": gmm["alpha_init"],
        "se_true": TRUE_PARAMS["sigma_e"],
        "se_hat": gmm["sigma_e_hat"],
        "se_init": gmm["sigma_e_init"],
        "overid_t_prod": gmm["overid_t_prod"],
        "overid_t_foc": gmm["overid_t_foc"],
        "mle_iters": n_outer,
        "mle_max": max_outer,
        "mle_cvg": n_conv,
        "mle_M": n_markets,
        "gmm_iters": gmm["gmm_nit"],
        "gmm_max": 200,
        "mle_time": mle.get("timings", {}).get("total_time_sec", 0),
        "gmm_time": gmm["gmm_time_s"],
    }


def print_table(rows: List[dict], title: str):
    """Print markdown table."""
    if not rows:
        return
    print(f"\n### {title}\n")

    hdr = (
        "| M | N/J | tau_hat | tg_hat | d_corr | d_RMSE | tq_corr | tq_RMSE "
        "| eta_hat | alpha_hat(init) | se_hat(init) "
        "| t_prod | t_foc | MLE_it | MLE_cvg | GMM_it | wall_s |"
    )
    sep = "|" + "|".join(["---"] * 17) + "|"
    print(hdr)
    print(sep)

    for r in rows:
        cells = [
            f"{r['M']}",
            f"{r['NJ']}",
            f"{r['tau_hat']:.4f}",
            f"{r['tg_hat']:.3f}",
            f"{r['delta_corr']:.3f}",
            f"{r['delta_rmse']:.3f}",
            f"{r['tq_corr']:.3f}",
            f"{r['tq_rmse']:.3f}",
            f"{r['eta_hat']:.3f}",
            f"{r['alpha_hat']:.3f}({r['alpha_init']:.3f})",
            f"{r['se_hat']:.4f}({r['se_init']:.4f})",
            f"{r['overid_t_prod']:.2f}",
            f"{r['overid_t_foc']:.2f}",
            f"{r['mle_iters']}/{r['mle_max']}",
            f"{r['mle_cvg']}/{r['mle_M']}",
            f"{r['gmm_iters']}/{r['gmm_max']}",
            f"{r['mle_time'] + r['gmm_time']:.0f}",
        ]
        print("| " + " | ".join(cells) + " |")

    print(f"\nTrue: tau={TRUE_PARAMS['tau']}, tg={TRUE_TG:.3f}, "
          f"eta={TRUE_PARAMS['eta']}, alpha={TRUE_PARAMS['alpha']}, "
          f"sigma_e={TRUE_PARAMS['sigma_e']}")


def print_notes():
    """Print equations and init notes."""
    print("\n### Equations\n")
    print("**Step 1 (Micro MLE):** Maximize choice likelihood over "
          "(tau, tg, delta_m, tq_m).")
    print("  Distributed: outer L-BFGS over (tau, tg), inner L-BFGS "
          "over (delta, tq) per market.\n")
    print("**Step 2a (Linear IV):** delta_j = c + eta*ln(w_j) + xi_j, "
          "IV = (1, z1_j).\n")
    print("**Step 2b (GMM):** m3_j = sigma_e*tq_j - ln(tQ_j(sigma_e)) "
          "- ln(w_j) - ln(L_j) + ln(1-alpha) + ln(R_j) = 0")
    print("  Instruments Z = [1, tq_j - mean(tq_j)]. "
          "W = (Z'Z/J)^{-1}.\n")
    print("**Step 2c (Overid):** "
          "t = sqrt(J)*mean(z2*ln_A)/std(z2*ln_A), ~N(0,1) under H0.")
    print("  ln(A)^prod and ln(A)^foc should agree if model is correct.\n")
    print("### Initializations\n")
    print("- tau, delta: MNL logit on choices (naive_init step 0e)")
    print("- tg: OLS ln(w)~v, beta1/sigma_e_hat (naive_init step 0a)")
    print("- tq: tg*quantile_05(v matched to j) (naive_init step 0d)")
    print("- eta: 2SLS delta on ln(w) with z1 (naive_init step 0f)")
    print("- alpha: 2SLS ln(R) on ln(QL) with z2 (naive_init step 0c)")
    print("- sigma_e: wage regression residual std (naive_init step 0a)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global BASE_DIR

    parser = argparse.ArgumentParser(
        description="Two-step scaling test: MLE + GMM",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Run only M=10 N=2500 (quick test)")
    parser.add_argument("--skip_datagen", action="store_true",
                        help="Skip data generation (assume data exists)")
    parser.add_argument("--skip_mle", action="store_true",
                        help="Skip MLE (assume estimates exist)")
    parser.add_argument("--skip_gmm", action="store_true",
                        help="Skip GMM (only run datagen + MLE)")
    parser.add_argument("--base_dir", type=str, default=str(BASE_DIR))
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    BASE_DIR = Path(args.base_dir)
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        all_configs = [{"M": 10, "N": 2500, "J": 100}]
        m_configs = all_configs
        nj_configs = []
    else:
        all_configs = _unique_configs(M_SCALING + NJ_SCALING)
        m_configs = M_SCALING
        nj_configs = NJ_SCALING

    total_t0 = time.perf_counter()

    # ---- Phase 1: Data generation ----
    if not args.skip_datagen:
        print("=" * 60)
        print("Phase 1: Data generation")
        print("=" * 60)
        for cfg in all_configs:
            t0 = time.perf_counter()
            print(f"\n--- {_cfg_label(cfg)} ---")
            try:
                generate_data(cfg, seed=args.seed)
                print(f"  Done ({time.perf_counter() - t0:.0f}s)")
            except Exception as e:
                print(f"  FAILED: {e}")
                return

    # ---- Phase 2: MLE ----
    if not args.skip_mle:
        print("\n" + "=" * 60)
        print("Phase 2: Distributed MLE")
        print("=" * 60)
        for cfg in all_configs:
            t0 = time.perf_counter()
            print(f"\n--- {_cfg_label(cfg)} ---")
            try:
                run_mle(cfg)
                print(f"  Done ({time.perf_counter() - t0:.0f}s)")
            except Exception as e:
                print(f"  FAILED: {e}")
                return

    # ---- Phase 3: GMM ----
    m_results, nj_results = [], []
    if not args.skip_gmm:
        print("\n" + "=" * 60)
        print("Phase 3: GMM (inline)")
        print("=" * 60)

        gmm_cache: Dict[Tuple[int, int, int], dict] = {}
        for cfg in all_configs:
            t0 = time.perf_counter()
            print(f"\n--- {_cfg_label(cfg)} ---")
            try:
                gmm = run_gmm(cfg)
                gmm_cache[(cfg["M"], cfg["N"], cfg["J"])] = gmm
                print(f"  eta={gmm['eta_hat']:.3f}  alpha={gmm['alpha_hat']:.3f}  "
                      f"se={gmm['sigma_e_hat']:.4f}  "
                      f"t_prod={gmm['overid_t_prod']:.2f}  "
                      f"t_foc={gmm['overid_t_foc']:.2f}  "
                      f"({time.perf_counter() - t0:.0f}s)")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()
                gmm_cache[(cfg["M"], cfg["N"], cfg["J"])] = {
                    "eta_hat": np.nan, "eta_init": np.nan, "eta_const": np.nan,
                    "alpha_hat": np.nan, "alpha_init": np.nan,
                    "sigma_e_hat": np.nan, "sigma_e_init": np.nan,
                    "overid_t_prod": np.nan, "overid_t_foc": np.nan,
                    "gmm_converged": False, "gmm_nit": 0,
                    "gmm_obj": np.nan, "gmm_time_s": 0,
                }

        # ---- Phase 4: Output ----
        print("\n" + "=" * 60)
        print("Phase 4: Results")
        print("=" * 60)

        for cfg in m_configs:
            key = (cfg["M"], cfg["N"], cfg["J"])
            if key in gmm_cache:
                try:
                    m_results.append(collect_one(cfg, gmm_cache[key]))
                except Exception as e:
                    print(f"  Could not collect {_cfg_label(cfg)}: {e}")

        for cfg in nj_configs:
            key = (cfg["M"], cfg["N"], cfg["J"])
            if key in gmm_cache:
                try:
                    nj_results.append(collect_one(cfg, gmm_cache[key]))
                except Exception as e:
                    print(f"  Could not collect {_cfg_label(cfg)}: {e}")

        print_table(m_results, "Table 1: M Scaling (J=100, N=2500)")
        if nj_results:
            print_table(nj_results, "Table 2: N/J Scaling (J=100, M=100)")
        print_notes()

        # Save summary JSON
        summary = {
            "m_scaling": m_results,
            "nj_scaling": nj_results,
            "true_params": {**TRUE_PARAMS, "tilde_gamma": TRUE_TG},
            "total_time_s": time.perf_counter() - total_t0,
        }
        summary_path = BASE_DIR / "two_step_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=lambda x: None if np.isnan(x) else x)
        print(f"\nSummary written to {summary_path}")

    total_time = time.perf_counter() - total_t0
    print(f"\nTotal wall time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
