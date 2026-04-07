# Naive Initialization Procedure

Implementation: `code/screening/analysis/lib/naive_init.py` -> `compute_pooled_naive_init()`

## Overview

The naive initializer produces starting values for all parameters from data
alone, with no dependence on true/known parameter values. It pools worker
and firm microdata across all markets to estimate global parameters, then
computes per-market local parameters. The procedure is designed so each
step uses only observables and estimates from prior steps.

Parameters initialized: theta_G = [tau, tilde_gamma, alpha, sigma_e, eta, gamma0]
plus per-market delta (J,) and tilde_q (J,) vectors.

## Step 0a: Skill parameters from wages

**Data**: All matched workers (choice_idx > 0) across all markets.
**Regression**: OLS of ln(w_i) on (1, v_i) where w_i is the wage of
worker i's chosen firm.

| Estimate | Formula | Rationale |
|----------|---------|-----------|
| gamma0   | intercept (beta_0) | In the model, E[ln w \| v] ~ gamma0 + gamma1*v for matched workers. The intercept recovers the skill distribution location. |
| sigma_e  | sqrt(residual variance) | The residual captures the idiosyncratic skill component e_i ~ N(0, sigma_e^2). |
| tilde_gamma | slope / sigma_e = beta_1 / sigma_e | The slope beta_1 ~ gamma1 = sigma_e * tilde_gamma, so dividing by sigma_e recovers the standardized skill coefficient. |

**Why wages identify skill parameters**: In the model, worker wages are
determined by their skill q_i = exp(gamma0 + gamma1*v_i + e_i). The wage
regression exploits the fact that v_i is observed and the noise e_i is
independent of v_i.

## Step 0b: Firm average skill

**Data**: Per-market firm-worker matches.
**Formula**: Q_j = (1/L_j) sum_{i matched to j} exp(gamma0 + sigma_e * tilde_gamma * v_i)

This estimates the average skill level at firm j using the skill parameters
from Step 0a. Since we don't observe e_i, we use the conditional expectation
E[q_i | v_i] = exp(gamma0 + gamma1*v_i + sigma_e^2/2), approximated here
by the sample average of exp(gamma0 + gamma1*v_i) across matched workers.

Firms with no matched workers (L_j = 0) are assigned the market median Q_j.

## Step 0c: Production function (2SLS)

**Data**: All firms across all markets.
**Regression**: 2SLS of ln(R_j) on ln(Q_j * L_j) instrumented with (1, z2_j).

The production function is R_j = A_j * (Q_j * L_j)^{1-alpha}, so
ln(R_j) = ln(A_j) + (1-alpha) * ln(Q_j * L_j). Since Q_j is estimated
(measurement error) and A_j may correlate with inputs, we instrument
with z2_j (a supply-side instrument for firm productivity).

The coefficient on ln(Q_j * L_j) is (1-alpha), so alpha = 1 - coefficient.
Clamped to (0, 1); defaults to 0.25 if the 2SLS estimate falls
outside this range.

## Step 0d: Preference parameters (conditional logit)

**Data**: Per-market worker choice microdata.
**Model**: Standard MNL ignoring screening:
  P(y_i = j) = exp(delta_j - tau * d_ij) / (1 + sum_k exp(delta_k - tau * d_ik))

Estimated per-market via L-BFGS-B with analytic gradient. The pooled tau
is the N-weighted average across markets.

**Why ignore screening**: At the initialization stage, we don't yet have
reliable screening thresholds. The MNL treats all firms as feasible for
all workers. This biases delta and tau but provides reasonable starting
values for the iterative solver.

**Why before screening thresholds**: The MNL logit has no dependency on
tilde_gamma, so it can run before the tg scan. This provides (tau, delta)
for evaluating the micro NLL in the coarse grid scan (Step 0e).

## Step 0e: Profiled tilde_gamma grid scan

**Data**: Subsample of markets (first 10 by default).
**Depends on**: tau and delta from Step 0d, tilde_gamma_naive from Step 0a,
gamma0 and sigma_e from Step 0a, alpha from Step 0c.

The naive sigma_e from the wage regression is biased upward (~0.26 vs true
0.135). Since tilde_gamma = gamma1/sigma_e, this halves the naive tg estimate,
placing tq in a gradient dead zone and trapping the solver at a suboptimal basin.

**Algorithm** (profiled inner solves):
1. Build a log-spaced grid of 5 points from tg_naive/3 to tg_naive*3,
   always including tg_naive itself.
2. Subsample up to 10 markets (deterministic: first N).
3. For each candidate tg:
   a. Set theta_G = [tau_naive, tg_candidate].
   b. For each market, initialize theta_m = [delta_MNL, tq_quantile(tg)]
      and run `solve_market()` (L-BFGS inner solve for delta and tq at
      fixed theta_G, maxiter=200). This profiles out (delta, tq).
   c. Sum per-market optimized NLLs → profiled NLL at this tg.
4. Return the tg with lowest profiled NLL. If the best is at a grid
   boundary, fall back to tg_naive (monotone landscape = unreliable).

The profiled objective is faithful: each tg is evaluated at its own
optimized (delta, tq), not at MNL deltas that ignore screening.
Cost: ~3-8 min (JIT compilation per unique J, then ~2-5s per inner solve).

**CLI**: `--skip_coarse_tg_scan` disables this step and uses the naive tg
from Step 0a directly.

## Step 0f: Screening thresholds

**Data**: Per-market firm-worker matches.
**Formula**: tq_j = tilde_gamma * quantile(v[matched to j], 0.05)

The screening cutoff determines the minimum skill to qualify for firm j,
so the lowest-skilled matched workers bound the cutoff from above. Using
the 5th percentile of matched-worker skill (scaled by tilde_gamma) places
tq in the region where screening is active and the NLL gradient is nonzero.

Uses the (possibly scan-refined) tilde_gamma from Step 0e. Firms with no
matched workers get the market median tq as fallback.

## Step 0g: Wage equation (2SLS)

**Data**: All firms across all markets.
**Regression**: 2SLS of delta_j on ln(w_j) instrumented with (1, z1_j).

In the model, delta_j = eta * ln(w_j) + xi_j, where xi_j is a firm
amenity. Since w_j is endogenous (correlated with xi_j through sorting),
we instrument with z1_j (a demand-side instrument for wages).

The coefficient on ln(w_j) is eta (wage elasticity of mean utility).

## Parameter vector

theta_G = [tau, tilde_gamma, alpha, sigma_e, eta, gamma0]

| Index | Parameter | Step | Typical range |
|-------|-----------|------|---------------|
| 0 | tau | 0d | (0, 1) |
| 1 | tilde_gamma | 0a/0e | (0, 20) |
| 2 | alpha | 0c | (0, 1) |
| 3 | sigma_e | 0a | (0, 2) |
| 4 | eta | 0g | (0, 20) |
| 5 | gamma0 | 0a | (-2, 2) |
