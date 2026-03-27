# Model Components: Standardized Spec M Parametrization

Reference implementation: `code/estimation/model_components.py`

This document describes the building blocks of the hybrid MLE/GMM objective under Specification M (log-multiplicative skill), using the standardized parametrization from `estimation.tex`. The existing `jax_model.py` remains intact; the new module provides equivalent computations in a cleaner interface that separates micro-identified from macro-identified parameters.

## Standardized Parameters

The key insight is that match probabilities depend on the skill distribution only through standardized quantities:

$$\tilde\gamma = \frac{\gamma_1}{\sigma_e}, \qquad \tilde q_j = \frac{\ln\bar q_j - \gamma_0}{\sigma_e}$$

The threshold function becomes $t_j(v_i) = \tilde q_j - \tilde\gamma\, v_i$ in standardized units.

### Parameter partition

| Parameter set | Parameters | Identified by |
|---------------|-----------|---------------|
| **Micro** | $\tau,\; \tilde\gamma,\; \{\delta_j\},\; \{\tilde q_j\}$ | Match likelihood |
| **Macro** | $\gamma_0,\; \sigma_e,\; \alpha,\; \eta$ | Firm-level moments |

The micro likelihood depends *only* on the first set. The structural parameters $(\gamma_0, \sigma_e)$ enter only through the macro moments. Given estimates, the structural skill parameters are recovered as $\gamma_1 = \tilde\gamma\,\sigma_e$ and $\ln\bar q_j = \gamma_0 + \sigma_e\,\tilde q_j$.

### Why standardize?

1. **Fewer parameters in the likelihood.** The micro objective depends on $(\tau, \tilde\gamma, \delta, \tilde q)$, not on $(\gamma_0, \gamma_1, \sigma_e)$ separately.
2. **Clean separation.** $(\gamma_0, \sigma_e, \alpha, \eta)$ appear only in the GMM penalty, enabling two-step or concentrated estimation.
3. **No `aux` dict.** The old code threaded `gamma0`, `gamma1`, `sigma_e` through an auxiliary dictionary. The standardized interface takes all inputs as explicit arguments.

## Objective Function

The overall estimator (estimation.tex eq 4) is:

$$\hat\theta = \arg\min_\theta \; \underbrace{-\sum_{i=1}^N \ln\sigma_{y_i}(v_i, d_i;\,\theta_{\text{micro}})}_{\text{micro: negative log-likelihood}} \;+\; \underbrace{\tfrac{1}{2}\,\hat m(\theta)'\,\hat W\,\hat m(\theta)}_{\text{macro: GMM penalty}}$$

where $\theta_{\text{micro}} = (\tau, \tilde\gamma, \{\delta_j\}, \{\tilde q_j\})$ and $\theta = (\theta_{\text{micro}}, \gamma_0, \sigma_e, \alpha, \eta)$.

The five functions in `model_components.py` compute the constituent pieces of this objective.

## Function 1: `choice_probabilities`

```python
choice_probabilities(tau, tilde_gamma, delta, tilde_q, v, D) -> P_nat  # (N, J+1)
```

Computes $\sigma_j(v_i, d_i;\,\theta_{\text{micro}})$ for all workers and firms.

### Algorithm

Firms are sorted by ascending $\tilde q_j$, yielding nested choice sets. For each worker $i$, the probability of choosing firm $j$ sums over eligibility intervals $k = 0, \ldots, J$:

$$\sigma_j(v_i, d_i) = \sum_{k=0}^{J} \mathbf{1}[\tilde q_j \leq \tilde q_{o^{-1}(k)}] \;\cdot\; \underbrace{\bigl[\Phi(z_{k+1}^i) - \Phi(z_k^i)\bigr]}_{\text{interval probability}} \;\cdot\; \underbrace{\frac{\exp(\delta_j - \tau d_{ij})}{1 + \sum_{m=1}^k \exp(\delta_{o^{-1}(m)} - \tau d_{i,o^{-1}(m)})}}_{\text{logit share given choice set}}$$

where $z_k^i = \tilde q_{o^{-1}(k)} - \tilde\gamma\, v_i$ are the standardized z-values.

The key equivalence with the old code: $(\ln\bar q_j - \gamma_0 - \gamma_1 v_i) / \sigma_e = \tilde q_j - \tilde\gamma\, v_i$, so no structural parameters appear.

### Implementation

1. Sort firms by ascending $\tilde q$ to get `order_idx`, `inv_order`, `tilde_q_sorted`.
2. Pad cutoffs: $[-10^6,\; \tilde q_{\text{sorted}},\; +10^6]$.
3. Compute z-values directly: `z_hi = tilde_q_pad[1:] - tilde_gamma * v`, `z_lo = tilde_q_pad[:-1] - tilde_gamma * v`.
4. Interval probabilities: `p_x = Phi(z_hi) - Phi(z_lo)`, clipped and normalized.
5. Logit utilities: outside option = 1, inside = $\exp(\delta_j - \tau d_{ij})$.
6. Cumulative denominator and suffix sum to accumulate over eligible intervals.
7. Reorder from sorted to natural firm order; clip and normalize.

Returns `P_nat` of shape $(N, J{+}1)$ with column 0 = outside option.

## Function 2: `per_obs_nll`

```python
per_obs_nll(P_nat, choice_idx) -> nll  # (N,)
```

Computes $-\ln\sigma_{y_i}$ for each worker: extracts the probability of the observed choice and takes the negative log. The micro component of the objective is `jnp.sum(per_obs_nll(...))`.

## Function 3: `compute_tilde_Q_M`

```python
compute_tilde_Q_M(sigma_e, tilde_gamma, tau, delta, tilde_q, v, D, choice_idx) -> tilde_Q  # (J,)
```

Computes standardized average skill per firm under Spec M (estimation.tex eq 11):

$$\tilde Q_j^M(\sigma_e) = \frac{1}{L_j} \sum_{i:\,y_i=j} \exp(\sigma_e\,\tilde\gamma\, v_i) \;\cdot\; \mathbb{E}\!\left[\exp(\sigma_e\,\tilde e_i) \;\Big|\; y_i = j,\, v_i,\, d_i\right]$$

where $\tilde e_i \sim N(0,1)$ is the standardized unobserved skill.

### Why this depends on $\sigma_e$

Under Spec M, skill in levels is $q_i = \exp(\gamma_0 + \gamma_1 v_i + e_i)$, so average skill involves $\mathbb{E}[\exp(\sigma_e \tilde e)]$ in each truncation interval. The convexity of the exponential means that larger $\sigma_e$ inflates average skill more at high-threshold firms (where the truncation is more severe). When $\sigma_e = 0$, there is no within-firm skill dispersion and $\tilde Q_j = 1$ for all firms.

Structural recovery: $Q_j = e^{\gamma_0} \cdot \tilde Q_j^M(\sigma_e)$.

### Posterior expectation algorithm

For a worker matched to firm $j$, the posterior expectation $\mathbb{E}[\exp(\sigma_e \tilde e_i) \mid y_i = j]$ uses the joint-probability-tensor approach from estimation.tex eq 12. For each eligibility interval $k$ with standardized bounds $[a_{ik}, b_{ik}]$:

$$\mathbb{E}\!\left[\exp(\sigma_e\,\tilde e) \;\Big|\; \tilde e \in [a, b]\right] = \exp\!\left(\tfrac{\sigma_e^2}{2}\right) \cdot \frac{\Phi(b - \sigma_e) - \Phi(a - \sigma_e)}{\Phi(b) - \Phi(a)}$$

The implementation computes:
1. **Shifted z-values**: `z_hi_shifted = z_hi - sigma_e`, `z_lo_shifted = z_lo - sigma_e`.
2. **Exponential factor**: $\exp(\sigma_e \tilde\gamma v_i + \sigma_e^2/2)$ per worker.
3. **Shifted interval mass**: `p_x_shifted = Phi(z_hi_shifted) - Phi(z_lo_shifted)`.
4. **Unnormalized skill contribution**: `E_q_in_interval = exp_factor * p_x_shifted`.
5. **Joint probability tensor**: logit probability $\times$ interval mass $\times$ eligibility mask, summed over intervals and workers, then divided by model labor $L_j$.

## Function 4: `compute_gmm_moments_M`

```python
compute_gmm_moments_M(delta, tilde_q, tilde_Q, gamma0, sigma_e, alpha, eta,
                       w, R, L, z1, z2, z3) -> m_vec  # (4,)
```

Computes the 4 sample moments from Spec M (estimation.tex eqs 9--12). These identify the macro parameters $(\eta, \alpha, \sigma_e, \gamma_0)$.

### Moment 1: Preference (identifies $\eta$)

$$\hat m_1 = \frac{1}{J}\sum_j z_j^1 \cdot (\delta_j - \eta\ln w_j)$$

The residual $\xi_j = \delta_j - \eta\ln w_j$ is the unobserved amenity. The instrument $z_j^1$ is a TFP shifter (which moves wages without directly affecting amenities). At the true parameters, $\mathbb{E}[\xi_j \mid z_j^1] = 0$.

### Moment 2: Screening FOC (identifies $\alpha$)

$$\hat m_2 = \frac{1}{J}\sum_j z_j^2 \cdot \bigl[\ln(1{-}\alpha) + (1{-}\alpha)\gamma_0 + \sigma_e\tilde q_j - \alpha\ln\tilde Q_j - \alpha\ln L_j - \ln w_j\bigr]$$

This is the log screening FOC with the residual $-\ln A_j$. The instrument $z_j^2$ is an amenity shifter (which moves labor supply without directly affecting TFP). At the true parameters, $\mathbb{E}[\ln A_j \mid z_j^2] = 0$.

### Moment 3: $A_j$-free / revenue (identifies $\sigma_e$)

$$\hat m_3 = \frac{1}{J}\sum_j z_j^3 \cdot \bigl[\sigma_e\tilde q_j - \ln\tilde Q_j - \ln w_j - \ln L_j + \ln(1{-}\alpha) + \ln R_j\bigr]$$

This eliminates $A_j$ entirely by substituting in observed revenue $R_j = A_j (L_j Q_j)^{1-\alpha}$. The location parameter $\gamma_0$ cancels (it enters $\ln\bar q_j$ and $\ln Q_j$ symmetrically under Spec M). Different values of $\sigma_e$ generate different gaps between $\sigma_e \tilde q_j$ and $\ln \tilde Q_j$ across firms. Any instrument with cross-firm variation suffices (e.g., $z_j^3 = \tilde q_j$ or $z_j^3 = 1$).

### Moment 4: TFP normalization (identifies $\gamma_0$)

$$\hat m_4 = \frac{1}{J}\sum_j \bigl[\ln(1{-}\alpha) + (1{-}\alpha)\gamma_0 + \sigma_e\tilde q_j - \alpha\ln\tilde Q_j - \alpha\ln L_j - \ln w_j\bigr]$$

This is moment 2 evaluated at $z_j^2 = 1$. Given $(\alpha, \sigma_e)$ from moments 2--3, it pins $\gamma_0$ in closed form. Imposing $\mathbb{E}[\ln A_j] = 0$ is necessary because shifting $\gamma_0$ by $c$ under Spec M is absorbed by $A_j \to A_j e^{-c(1-\alpha)}$.

### Instrument requirements

All instruments must be passed as concrete arrays (not `None`) to avoid JIT recompilation branches. Use `jnp.ones(J)` for an unconditional moment.

## Function 5: `compute_penalty`

```python
compute_penalty(m_vec, W) -> scalar
```

Returns $\frac{1}{2}\,\hat m'\,\hat W\,\hat m$. This is the macro component of the objective.

## Assembling the Full Objective

To evaluate the combined objective at a parameter vector:

```python
# 1. Micro: choice probabilities (depends on micro params only)
P = choice_probabilities(tau, tilde_gamma, delta, tilde_q, v, D)
nll = jnp.sum(per_obs_nll(P, choice_idx))

# 2. Macro: standardized average skill (depends on sigma_e too)
tilde_Q = compute_tilde_Q_M(sigma_e, tilde_gamma, tau, delta, tilde_q,
                             v, D, choice_idx)

# 3. Macro: moment vector (depends on all structural params)
m_vec = compute_gmm_moments_M(delta, tilde_q, tilde_Q,
                                gamma0, sigma_e, alpha, eta,
                                w, R, L, z1, z2, z3)

# 4. Macro: penalty
penalty = compute_penalty(m_vec, W)

# 5. Combined objective
objective = nll + penalty
```

All functions are compatible with `jax.jit` and `jax.grad`.

## Comparison with `jax_model.py`

| Aspect | `jax_model.py` | `model_components.py` |
|--------|-----------------|------------------------|
| Choice prob inputs | `(tau, delta, ln_qbar)` + `aux{gamma0, gamma1, sigma_e}` | `(tau, tilde_gamma, delta, tilde_q)` -- no aux dict |
| z-values | $({\ln\bar q_j - \gamma_0 - \gamma_1 v_i})/{\sigma_e}$ | $\tilde q_j - \tilde\gamma\, v_i$ (equivalent) |
| Average skill | $Q_j = \mathbb{E}[q \mid \text{matched}]$ in levels | $\tilde Q_j = Q_j / e^{\gamma_0}$ -- standardized |
| Moments | Single screening-FOC residual (J-vector) | 4 scalar Spec M moments with instruments |
| Macro params | `[tau, alpha, gamma1, sigma_e, lambda_e]` (5 globals) | Micro: `(tau, tilde_gamma)`. Macro: `(gamma_0, sigma_e, alpha, eta)` |
| `lambda_e` | In theta, shifts $Q$ | Gone -- absorbed into $\gamma_0$ |
| `eta` | Fixed from params file | Identified by moment 1 |
