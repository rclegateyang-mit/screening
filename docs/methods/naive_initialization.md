# Naive Initialization Procedure

Implementation: `code/estimation/naive_init.py` -> `compute_pooled_naive_init()`

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

## Step 0d: Screening thresholds

**Data**: Per-market firm observables.
**Formula**: ln(qbar_j) = ln(w_j) + ln(Q_j) + ln(L_j) - ln(R_j) - ln(1 - alpha)

Derived from the screening FOC (estimation.tex eq. (2)): at the optimal
cutoff, the marginal worker's skill equals the firm's wage-to-marginal-
product ratio. Rearranging gives ln(qbar_j) in terms of observables
and the production parameter alpha.

**Standardization**: tilde_q_j = (ln(qbar_j) - gamma0) / sigma_e

This converts to the standardized units used in the likelihood, where
tilde_q measures cutoffs in standard deviations of the skill distribution.

## Step 0e: Preference parameters (conditional logit)

**Data**: Per-market worker choice microdata.
**Model**: Standard MNL ignoring screening:
  P(y_i = j) = exp(delta_j - tau * d_ij) / (1 + sum_k exp(delta_k - tau * d_ik))

Estimated per-market via L-BFGS-B with analytic gradient. The pooled tau
is the N-weighted average across markets.

**Why ignore screening**: At the initialization stage, we don't yet have
reliable screening thresholds. The MNL treats all firms as feasible for
all workers. This biases delta and tau but provides reasonable starting
values for the iterative solver.

## Step 0f: Wage equation (2SLS)

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
| 0 | tau | 0e | (0, 1) |
| 1 | tilde_gamma | 0a | (0, 20) |
| 2 | alpha | 0c | (0, 1) |
| 3 | sigma_e | 0a | (0, 2) |
| 4 | eta | 0f | (0, 20) |
| 5 | gamma0 | 0a | (-2, 2) |
