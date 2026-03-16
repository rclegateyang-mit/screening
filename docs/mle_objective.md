# Penalized MLE Estimator

Reference implementation: `code/estimation/run_mle_penalty_phi_sigma_jax.py`

This document describes the objective function minimized by the penalized maximum likelihood estimator, using the notation from the NSF project description.

## Parameter Vector

$$\theta = \bigl(\tau,\; \alpha,\; \gamma,\; \sigma_e,\; \lambda,\; \delta_1,\ldots,\delta_J,\; \bar{q}_1,\ldots,\bar{q}_J\bigr)$$

| Symbol | Code name | Constraint | Description |
|--------|-----------|------------|-------------|
| $\tau$ | `gamma` | $(0,1)$ | Commuting disutility |
| $\alpha$ | `beta` | $(0,1)$ | Curvature of production, $F(H)=H^{1-\alpha}$ |
| $\gamma$ | `phi` | $>0$ | Loading of observed skill proxy on log skill |
| $\sigma_e$ | `sigma_skill` | $>0$ | Scale of unobserved skill component |
| $\lambda$ | `lambda_skill` | $\mathbb{R}$ | Location of skill distribution |
| $\delta_j$ | `V_j` | $\mathbb{R}$ | Firm $j$ fixed effect, $\delta_j \equiv \eta \ln w_j + \xi_j$ |
| $\bar{q}_j$ | `c_j` | $>0$ | Firm $j$ screening threshold |

Observed data per worker: skill proxy $v_i$ (`x_skill`), location $\ell_i$ (`ell_x`, `ell_y`), matched firm $\mu(i)$ (`chosen_firm`).
Observed data per firm: wage $w_j$, revenue $R_j$ (`Y`), employment $L_j$, location.

## Objective

$$\hat\theta = \arg\min_\theta\;\Sigma(\theta) \;=\; \underbrace{-\sum_{i=1}^{N}\ln\sigma_{\mu(i)}(v_i, d_i;\,\theta)}_{\text{negative log-likelihood}} \;+\; \underbrace{\tfrac{1}{2}\,m(\theta)^\prime\, W\, m(\theta)}_{\text{screening FOC penalty}}$$

This is the mixed GMM-Likelihood objective of Grieco, Murry, Pinkse & Sagl (2025), combining micro-level match moments (the likelihood) with macro-level screening moments (the penalty). The matrix $W$ weights the penalty (defaults to the identity).

## First Term: Micro Match Likelihood

The negative log-likelihood sums over all $N$ workers. Each worker $i$'s contribution is $-\ln \sigma_{\mu(i)}$, where the match probability comes from equation (5) of the proposal:

$$\sigma_j(v_i,d_i;\,\theta) \;=\; \sum_{\substack{O\subseteq\mathcal{J},\; j\in O}} \Pr\bigl(O(q_i)=O\mid v_i\bigr) \;\cdot\; \Pr\bigl(j=\arg\max_{k\in O} U_{ik}\mid d_i\bigr)$$

### Choice set probability

Worker $i$'s skill is $\ln q_i = \gamma v_i + e_i$ with $e_i \sim N(\lambda, \sigma_e^2)$. Her choice set is $O(q_i) = \{j : q_i \geq \bar{q}_j\}$. Since choice sets are determined by crossing screening thresholds, sorting firms by $\bar{q}_{(1)} \leq \cdots \leq \bar{q}_{(J)}$ yields nested choice sets indexed by contiguous intervals. The probability that worker $i$ falls in a given interval is:

$$\Pr(O(q_i)=O\mid v_i) = \Phi\!\left(\frac{\min_{k\notin O}\bar{q}_k - \gamma v_i - \lambda}{\sigma_e}\right) - \Phi\!\left(\frac{\max_{k\in O}\bar{q}_k - \gamma v_i - \lambda}{\sigma_e}\right)$$

where $\Phi$ is the standard normal CDF. This nested structure avoids enumerating $2^J$ subsets.

### Conditional choice probability (logit)

Conditional on having choice set $O$, preferences follow a logit:

$$U_{ij} = \delta_j - \tau\, d_{ij} + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim \text{Gumbel}$$

$$\Pr(j = \arg\max_{k\in O} U_{ik} \mid d_i) = \frac{\exp(\delta_j - \tau\, d_{ij})}{1 + \sum_{k\in O}\exp(\delta_k - \tau\, d_{ik})}$$

The denominator includes the outside option (unemployment), normalized to utility zero.

### Computational implementation

The code (`jax_model.py`) sorts firms by cutoff, computes interval probabilities via the normal CDF, and combines them with logit shares using a cumulative-sum trick that vectorizes across workers and firms in JAX.

## Second Term: Screening FOC Penalty

The screening first-order condition equates the wage to the marginal revenue product of the least-skilled accepted worker:

$$w_j = (1-\alpha)\,A_j\,H_j^{-\alpha}\,\bar{q}_j$$

where $H_j = L_j Q_j$ is efficiency units of labor. At the true parameters, violations of this condition should be zero.

### Eliminating $A_j$ via the control function

TFP $A_j$ is unobserved. Following the approach discussed in the proposal (adapting Ackerberg, Caves & Frazier 2015), substitute $\ln A_j = \ln R_j - (1-\alpha)\ln H_j$ using observed revenue $R_j$ to eliminate $A_j$ from the moment condition.

### Moment vector

Define $\rho_j \equiv (1-\alpha)\frac{R_j}{L_j\, w_j}$. The penalty uses four aggregate moment conditions $m(\theta) \in \mathbb{R}^4$:

$$m(\theta) = \begin{pmatrix} \sum_j \bigl[\ln R_j - (1-\alpha)\ln L_j - \tfrac{1-\alpha}{\sigma_e}\ln(Q_j - \lambda)\bigr] \\[6pt] \sum_j \bigl[(Q_j - \lambda) - \rho_j\,(\bar{q}_j - \lambda)\bigr] \\[6pt] \sum_j \tfrac{R_j}{L_j w_j}\,\bar{q}_j\,\bigl[(Q_j - \lambda) - \rho_j\,(\bar{q}_j - \lambda)\bigr] \\[6pt] \sum_j \tfrac{R_j}{L_j w_j}\,\bigl[(Q_j - \lambda) - \rho_j\,(\bar{q}_j - \lambda)\bigr] \end{pmatrix}$$

Here $L_j(\theta)$ and $Q_j(\theta)$ are the model-implied labor supply and average skill computed from the match probabilities, while $R_j$ and $w_j$ are observed. At the true parameter values, each component equals zero.

The penalty contribution to the objective is $\frac{1}{2} m(\theta)' W m(\theta)$, where $W$ is a $4 \times 4$ weight matrix (identity by default, optionally loaded from file).

## Reparameterization and Optimization

To enforce parameter constraints, the optimizer works in an unconstrained space $z \in \mathbb{R}^K$ mapped to $\theta$ via smooth bijections:

| Parameter | Transform $z \mapsto \theta$ |
|-----------|------------------------------|
| $\tau, \alpha$ | Logistic sigmoid scaled to $(\varepsilon, 1-\varepsilon)$ |
| $\gamma, \sigma_e$ | Softplus (ensures $> 0$) |
| $\bar{q}_j$ | Softplus (ensures $> 0$) |
| $\delta_j, \lambda$ | Identity (unconstrained) |

The objective $\Sigma(\theta(z))$ is minimized over $z$ using L-BFGS (via `jaxopt.LBFGS`), with exact gradients computed by JAX autodifferentiation through both the transform and the model.

## Standard Errors

After convergence, the code computes a sandwich covariance estimator via the Hessian $H$ of the objective and the outer product of gradients $\Omega$:

$$\widehat{\text{Var}}(\hat\theta) = H^{-1}\,\Omega\, H^{-1}$$

Standard errors are reported both in the raw parameterization $\theta$ and in a transformed parameterization $\tilde\theta$ that normalizes cutoffs by the skill scale: $\tilde{c}_j = (\bar{q}_j - \lambda)/\sigma_e$ and $\tilde\gamma = \gamma / \sigma_e$.

## Code-to-Notation Map

For quick reference when reading the code:

| Code (`run_mle_penalty_phi_sigma_jax.py`) | This document |
|-------------------------------------------|---------------|
| `gamma` | $\tau$ (commuting cost) |
| `beta` | $\alpha$ (DRS parameter) |
| `phi` | $\gamma$ (skill loading) |
| `sigma_skill` | $\sigma_e$ (skill scale) |
| `lambda_skill` | $\lambda$ (skill location) |
| `V_j` | $\delta_j$ (firm fixed effect) |
| `c_j` | $\bar{q}_j$ (screening threshold) |
| `x_skill` | $v_i$ (observed skill proxy) |
| `D_nat` | $d_{ij}$ (commuting distance) |
| `S_j` | $Q_j$ (average skill at firm $j$) |
| `Y_j` | $R_j$ (revenue) |
| `per_obs_nll` | $-\ln\sigma_{\mu(i)}$ |
| `m_total` | $m(\theta)$ |
| `weight_matrix` | $W$ |
