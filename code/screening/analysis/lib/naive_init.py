"""Pooled naive initialization for the hybrid augmented-Lagrangian estimator.

Seven-step procedure that derives all starting values from data alone:
  0a. Wage regression -> gamma0, sigma_e, tilde_gamma
  0b. Firm average skill -> Q_j
  0c. Production function 2SLS -> alpha
  0d. Conditional logit -> tau, delta_j  (no tg dependency)
  0e. Coarse tg grid scan -> refined tilde_gamma  (uses tau, delta from 0d)
  0f. Screening thresholds -> tilde_q_j  (uses tg from 0e)
  0g. Wage equation 2SLS -> eta

See docs/naive_initialization.md for detailed documentation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize as sp_minimize


def _tsls_coeff(y: np.ndarray, x_endog: np.ndarray, Z: np.ndarray) -> float:
    """2SLS coefficient on x_endog.

    Args:
        y: (n,) dependent variable
        x_endog: (n,) endogenous regressor
        Z: (n, k) instrument matrix (should include constant)

    Returns:
        Scalar coefficient on x_endog.
    """
    # Stage 1: project x_endog onto Z
    coef1, _, _, _ = np.linalg.lstsq(Z, x_endog, rcond=None)
    x_hat = Z @ coef1

    # Stage 2: regress y on (ones, x_hat)
    X2 = np.column_stack([np.ones(len(y)), x_hat])
    coef2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
    return float(coef2[1])


def _mnl_tau_delta(
    D: np.ndarray,
    choice_idx: np.ndarray,
    J: int,
    N: int,
    tau0: float = 0.05,
    reg_delta: float = 1e-6,
    maxiter: int = 500,
) -> Tuple[float, np.ndarray]:
    """MNL logit for (tau, delta) ignoring screening.

    Returns (tau_hat, delta_hat).
    """
    choice_np = np.asarray(choice_idx, dtype=np.int32)
    D_np = np.asarray(D, dtype=np.float64)

    # Starting values from shares
    counts_all = np.bincount(np.clip(choice_np, 0, J), minlength=J + 1)
    s0 = counts_all[0] / max(N, 1)
    s = counts_all[1:] / max(N, 1)
    eps = 1e-12
    delta0 = np.log(np.maximum(s, eps)) - np.log(np.maximum(s0, eps))
    theta0 = np.concatenate(([tau0], delta0))

    # Sufficient statistics
    mask_in = choice_np > 0
    j_idx = (choice_np[mask_in] - 1).astype(int)
    d_chosen = D_np[mask_in, j_idx] if np.any(mask_in) else np.array([])
    counts_firms = (np.bincount(j_idx, minlength=J).astype(float)
                    if np.any(mask_in) else np.zeros(J))

    def nll_and_grad(theta_vec):
        theta_vec = np.asarray(theta_vec, float).ravel()
        tau_v = float(theta_vec[0])
        delta_vals = theta_vec[1:]
        U = delta_vals[None, :] - tau_v * D_np
        a = np.maximum(0.0, np.max(U, axis=1))
        exp_U_shift = np.exp(U - a[:, None])
        denom_scaled = np.exp(-a) + exp_U_shift.sum(axis=1)
        log_denom = a + np.log(denom_scaled)
        P = np.exp(U - log_denom[:, None])
        nll = (np.sum(log_denom)
               - (delta_vals @ counts_firms - tau_v * np.sum(d_chosen))
               + 0.5 * reg_delta * (delta_vals @ delta_vals))
        g_delta = P.sum(axis=0) - counts_firms + reg_delta * delta_vals
        g_tau = -float(np.sum(P * D_np)) + np.sum(d_chosen)
        return nll, np.concatenate(([g_tau], g_delta.astype(float)))

    bounds = [(0.0, 1.0)] + [(-np.inf, np.inf)] * J
    res = sp_minimize(nll_and_grad, theta0, method="L-BFGS-B", jac=True,
                      bounds=bounds,
                      options={"maxiter": maxiter, "ftol": 1e-10, "iprint": -1})
    return float(res.x[0]), res.x[1:].copy()


def _compute_tq_from_tg(
    tilde_gamma: float,
    market_datas: list,
    quantile: float = 0.05,
) -> list:
    """Compute per-market tq vectors from a candidate tilde_gamma.

    For each firm j, tq_j = tilde_gamma * quantile(v[matched to j], q).
    Firms with no matched workers get the market median tq.

    Args:
        tilde_gamma: Candidate standardized skill loading.
        market_datas: List of HybridMarketData.
        quantile: Low quantile of matched-worker skill (default 0.05).

    Returns:
        List of per-market tq arrays (each shape (J,)).
    """
    tilde_qs = []
    for md in market_datas:
        choice_np = np.asarray(md.choice_idx, dtype=np.int32)
        v_np = np.asarray(md.v)
        tq = np.empty(md.J, dtype=np.float64)
        for j in range(md.J):
            mask_j = choice_np == (j + 1)
            if np.any(mask_j):
                v_low = float(np.quantile(v_np[mask_j], quantile))
                tq[j] = tilde_gamma * v_low
            else:
                tq[j] = np.nan
        # Fill firms with no workers using market median
        valid = tq[np.isfinite(tq)]
        if valid.size > 0:
            fallback = float(np.median(valid))
        else:
            fallback = tilde_gamma * float(np.median(v_np))
        tq[~np.isfinite(tq)] = fallback
        tilde_qs.append(tq.astype(np.float64))
    return tilde_qs


def _coarse_tg_scan(
    tg_naive: float,
    tau: float,
    deltas: list,
    market_datas: list,
    gamma0: float,
    alpha: float,
    sigma_e: float,
    n_grid: int = 7,
    max_markets: int = 10,
    inner_maxiter: int = 500,
) -> float:
    """Profiled grid scan over tilde_gamma candidates.

    For each tg on a log-spaced grid, runs per-market inner solves
    (optimizing delta and tq via L-BFGS) at fixed (tau, tg).  This
    gives the profiled NLL — min_{delta,tq} NLL(tau, tg, delta, tq) —
    which is a faithful objective for selecting tg.

    Args:
        tg_naive: Naive tilde_gamma from wage regression.
        tau: Distance decay from MNL logit (Step 0d).
        deltas: Per-market MNL delta vectors (used as initial values).
        market_datas: List of HybridMarketData.
        gamma0: Skill intercept from Step 0a.
        alpha: Production elasticity from Step 0c.
        sigma_e: Skill noise std from Step 0a.
        n_grid: Number of grid points (default 7).
        max_markets: Max markets to evaluate (default 10, first-N).
        inner_maxiter: Max L-BFGS iterations per inner solve (default 500).

    Returns:
        Best tilde_gamma candidate (lowest profiled NLL).
    """
    from screening.analysis.mle.distributed_worker import solve_market, MarketData

    # Build log-spaced grid including tg_naive itself.
    # Asymmetric range: naive is biased low (sigma_e biased high),
    # so extend further above than below.
    tg_lo = tg_naive / 2.0
    tg_hi = tg_naive * 4.0
    grid = np.geomspace(tg_lo, tg_hi, n_grid)
    # Ensure tg_naive is in the grid (replace closest point)
    closest_idx = int(np.argmin(np.abs(grid - tg_naive)))
    grid[closest_idx] = tg_naive

    # Subsample markets and convert HybridMarketData → MarketData
    sub_hmds = market_datas[:max_markets]
    sub_deltas = deltas[:max_markets]
    sub_mds = []
    for hmd in sub_hmds:
        sub_mds.append(MarketData(
            market_id=hmd.market_id,
            X_m=np.asarray(hmd.v, dtype=np.float64),
            choice_m=np.asarray(hmd.choice_idx, dtype=np.int32),
            D_m=np.asarray(hmd.D, dtype=np.float64),
            w_m=np.asarray(hmd.w, dtype=np.float64),
            Y_m=np.asarray(hmd.R, dtype=np.float64),
            labor_m=np.asarray(hmd.L, dtype=np.float64),
            gamma0=gamma0, alpha=alpha, sigma_e=sigma_e,
            J_per=hmd.J, N_per=hmd.N,
            z1_m=np.asarray(hmd.z1, dtype=np.float64),
            z2_m=np.asarray(hmd.z2, dtype=np.float64),
        ))

    best_idx = -1
    best_nll = np.inf
    nlls = np.empty(len(grid))

    for k, tg_cand in enumerate(grid):
        theta_G = np.array([tau, float(tg_cand)], dtype=np.float64)
        tq_list = _compute_tq_from_tg(float(tg_cand), sub_hmds)
        total_nll = 0.0
        n_conv = 0
        for md, delta, tq in zip(sub_mds, sub_deltas, tq_list):
            theta_m_init = np.concatenate([delta, tq]).astype(np.float64)
            result = solve_market(
                md, theta_G, theta_m_init,
                inner_maxiter=inner_maxiter, inner_tol=1e-7,
                compute_hessian=False,
            )
            total_nll += result.nll_m
            n_conv += int(result.inner_converged)
        nlls[k] = total_nll
        if total_nll < best_nll:
            best_nll = total_nll
            best_idx = k
        print(f"    tg={tg_cand:7.3f}  NLL={total_nll:12.0f}  "
              f"conv={n_conv}/{len(sub_mds)}")

    best_tg = float(grid[best_idx])

    # Guard: if the best is at a grid boundary, the profiled landscape
    # may still be monotone.  Fall back to naive.
    at_boundary = (best_idx == 0) or (best_idx == len(grid) - 1)
    if at_boundary:
        print(f"  Coarse tg scan: naive={tg_naive:.2f}, "
              f"grid=[{grid.min():.2f}..{grid.max():.2f}], "
              f"best={best_tg:.2f} at boundary -> keeping naive "
              f"(NLL@naive={nlls[closest_idx]:.0f}, "
              f"NLL@boundary={best_nll:.0f})")
        best_tg = tg_naive
    else:
        print(f"  Coarse tg scan: naive={tg_naive:.2f}, "
              f"grid=[{grid.min():.2f}..{grid.max():.2f}], "
              f"best={best_tg:.2f} (NLL={best_nll:.0f})")

    return best_tg


def compute_pooled_naive_init(
    market_datas: list,
    coarse_tg_scan: bool = True,
) -> Tuple[np.ndarray, list, list]:
    """Pooled 7-step naive initialization.

    Args:
        market_datas: List of HybridMarketData (all markets, rank 0 pre-scatter).
        coarse_tg_scan: If True (default), run a coarse grid scan over
            tilde_gamma candidates using the micro NLL after the MNL step.

    Returns:
        theta_G_init: (6,) array [tau, tilde_gamma, alpha, sigma_e, eta, gamma0]
        deltas: list of per-market delta vectors
        tilde_qs: list of per-market tilde_q vectors
    """
    # ==================================================================
    # Step 0a: Wage regression (pooled workers)
    # ==================================================================
    all_ln_w = []
    all_v = []
    for md in market_datas:
        mask = np.asarray(md.choice_idx, dtype=np.int32) > 0
        if not np.any(mask):
            continue
        v_matched = np.asarray(md.v)[mask]
        j_idx = np.asarray(md.choice_idx, dtype=np.int32)[mask] - 1
        w_arr = np.asarray(md.w)
        ln_w_matched = np.log(np.maximum(w_arr[j_idx], 1e-300))
        all_ln_w.append(ln_w_matched)
        all_v.append(v_matched)

    pooled_ln_w = np.concatenate(all_ln_w)
    pooled_v = np.concatenate(all_v)

    # OLS: ln_w = beta0 + beta1 * v + residual
    X_ols = np.column_stack([np.ones(len(pooled_v)), pooled_v])
    beta, _, _, _ = np.linalg.lstsq(X_ols, pooled_ln_w, rcond=None)
    gamma0 = float(beta[0])
    beta1 = float(beta[1])
    residuals = pooled_ln_w - X_ols @ beta
    sigma_e = float(np.sqrt(np.mean(residuals ** 2)))
    sigma_e = max(sigma_e, 1e-6)
    tilde_gamma = beta1 / sigma_e

    # ==================================================================
    # Step 0b: Firm average skill (per-market)
    # ==================================================================
    gamma1_est = sigma_e * tilde_gamma  # = beta1
    Q_per_market = []
    for md in market_datas:
        J = md.J
        choice_np = np.asarray(md.choice_idx, dtype=np.int32)
        v_np = np.asarray(md.v)
        Q_j = np.zeros(J, dtype=np.float64)
        for j in range(J):
            mask_j = choice_np == (j + 1)
            if np.any(mask_j):
                Q_j[j] = np.mean(np.exp(gamma0 + gamma1_est * v_np[mask_j]))
        # Firms with no workers: median of positive Q values
        pos_mask = Q_j > 0
        if not np.all(pos_mask) and np.any(pos_mask):
            Q_j[~pos_mask] = np.median(Q_j[pos_mask])
        elif not np.any(pos_mask):
            Q_j[:] = 1.0
        Q_per_market.append(Q_j)

    # ==================================================================
    # Step 0c: Production function 2SLS (pooled firms)
    #   R = A * (QL)^{1-alpha}  =>  ln R = ln A + (1-alpha)*ln(QL)
    #   2SLS coefficient on ln(QL) estimates (1-alpha)
    # ==================================================================
    all_ln_R = []
    all_ln_QL = []
    all_Z_prod = []
    for md, Q_j in zip(market_datas, Q_per_market):
        R_np = np.asarray(md.R)
        L_np = np.asarray(md.L)
        z2_np = np.asarray(md.z2)
        QL = Q_j * np.maximum(L_np, 1e-300)
        all_ln_R.append(np.log(np.maximum(R_np, 1e-300)))
        all_ln_QL.append(np.log(np.maximum(QL, 1e-300)))
        all_Z_prod.append(np.column_stack([np.ones(md.J), z2_np]))

    pooled_ln_R = np.concatenate(all_ln_R)
    pooled_ln_QL = np.concatenate(all_ln_QL)
    pooled_Z_prod = np.vstack(all_Z_prod)

    one_minus_alpha = _tsls_coeff(pooled_ln_R, pooled_ln_QL, pooled_Z_prod)
    alpha = 1.0 - one_minus_alpha
    if not (0.01 <= alpha <= 0.99):
        alpha = 0.25

    # ==================================================================
    # Step 0d: Conditional logit (per-market)
    #   No dependency on tg — moved before tq computation so that
    #   (tau, delta) are available for the coarse tg scan.
    # ==================================================================
    deltas = []
    tau_sum = 0.0
    total_N = 0
    for md in market_datas:
        tau_hat, delta_hat = _mnl_tau_delta(md.D, md.choice_idx, md.J, md.N)
        deltas.append(delta_hat.astype(np.float64))
        tau_sum += tau_hat * md.N
        total_N += md.N
    tau = tau_sum / max(total_N, 1)

    # ==================================================================
    # Step 0e: Coarse tg grid scan (optional)
    #   Uses (tau, delta) from 0d to evaluate micro NLL over a log-spaced
    #   grid of tg candidates. Picks the tg with lowest NLL.
    # ==================================================================
    if coarse_tg_scan:
        tilde_gamma = _coarse_tg_scan(
            tilde_gamma, tau, deltas, market_datas,
            gamma0, alpha, sigma_e,
        )

    # ==================================================================
    # Step 0f: Screening thresholds (per-market)
    #   Uses the (possibly scan-refined) tilde_gamma.
    # ==================================================================
    tilde_qs = _compute_tq_from_tg(tilde_gamma, market_datas)

    # ==================================================================
    # Step 0g: Wage equation 2SLS (pooled firms)
    #   delta_j = eta * ln w_j + xi_j,  instrument: z1
    # ==================================================================
    all_delta = []
    all_ln_w_firm = []
    all_Z_wage = []
    for md, delta in zip(market_datas, deltas):
        w_np = np.asarray(md.w)
        z1_np = np.asarray(md.z1)
        all_delta.append(delta)
        all_ln_w_firm.append(np.log(np.maximum(w_np, 1e-300)))
        all_Z_wage.append(np.column_stack([np.ones(md.J), z1_np]))

    pooled_delta = np.concatenate(all_delta)
    pooled_ln_w_firm = np.concatenate(all_ln_w_firm)
    pooled_Z_wage = np.vstack(all_Z_wage)

    eta = _tsls_coeff(pooled_delta, pooled_ln_w_firm, pooled_Z_wage)

    # ==================================================================
    # Assemble theta_G
    # ==================================================================
    theta_G = np.array([tau, tilde_gamma, alpha, sigma_e, eta, gamma0],
                       dtype=np.float64)
    return theta_G, deltas, tilde_qs
