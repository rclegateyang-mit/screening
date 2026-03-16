# BLP Contraction Mapping for Delta Profiling

## Motivation

The baseline MLE jointly estimates all parameters:

```
theta = [tau, alpha, gamma, sigma_e, lambda_e, delta_1..J, qbar_1..J]
```

This gives K = 5 + 2J free parameters. With M = 50 markets and J_per = 100 firms per market, J = 5,000 and K = 10,005. At this scale the L-BFGS optimizer struggles to converge (grad_norm remains large even after thousands of iterations), because the optimizer must simultaneously navigate the high-dimensional delta surface while fitting the structural parameters.

The BLP contraction mapping addresses this by **profiling out the deltas**: instead of optimizing over them directly, we solve for the unique delta vector that rationalizes the observed market shares at each evaluation of the objective. This reduces the search space from 5 + 2J to 5 + J parameters, cutting the problem roughly in half.

This is the standard approach in the BLP demand estimation literature (Berry, Levinsohn, and Pakes, 1995).

## The contraction mapping

### Fixed-point equation

For a given market m with structural parameters (tau, qbar_m, gamma, sigma_e, mu_e), the mean utility vector delta_m is the unique solution to:

```
s_data(j) = s_model(j; delta_m)    for all firms j
```

where s_data are the observed empirical market shares and s_model are the predicted shares from the discrete-choice model.

The BLP contraction iterates:

```
delta^{t+1} = delta^{t} + log(s_data) - log(s_model(delta^{t}))
```

This mapping is a contraction under standard regularity conditions, so it converges to the unique fixed point from any starting value.

### Model shares

Predicted shares are computed from the choice probability model:

```
P(j | worker i) = choice_probabilities_from_cutoffs(tau, delta, qbar, X_i, aux)
s_model(j) = (1/N) * sum_i P(j | worker i)
```

where `choice_probabilities_from_cutoffs` is the core function from `jax_model.py` that evaluates the discrete-choice probabilities given the screening cutoffs and mean utilities.

### Convergence criterion

The iteration terminates when:

```
max_j |log(s_data(j)) - log(s_model(j; delta^t))| < tol
```

Default tolerance is 1e-12 with a maximum of 1,000 iterations. In practice, convergence typically occurs within 50-200 iterations.

## Parameter spaces

### Joint method (default)

```
theta = [tau, alpha, gamma, sigma_e, lambda_e, delta_1..J, qbar_1..J]
K = 5 + 2J
```

All parameters are optimized simultaneously by L-BFGS.

### Contraction method

```
theta = [tau, alpha, gamma, sigma_e, lambda_e, qbar_1..J]
K = 5 + J
```

At each objective evaluation, delta is solved via the contraction mapping rather than treated as a free parameter. The optimizer only searches over the structural parameters and the screening cutoffs.

### Parameter transforms

Both methods use the same reparameterization to map bounded parameters to unconstrained space:

| Parameter | Transform |
|-----------|-----------|
| tau       | sigmoid (clipped to (eps, 1-eps)) |
| alpha     | sigmoid (clipped to (eps, 1-eps)) |
| gamma     | softplus + floor |
| sigma_e   | softplus + floor |
| lambda_e  | identity (unbounded) |
| qbar      | softplus + floor |
| delta     | identity (joint only) |

## Implementation

### Files

| File | Role |
|------|------|
| `code/code/estimation/blp_contraction.py` | Contraction mapping functions |
| `code/code/estimation/jax_model.py` | Core choice probability model (exports `choice_probabilities_from_cutoffs`) |
| `code/code/estimation/run_mle_penalty_phi_sigma_jax.py` | Main estimation script with `--delta_method` flag |

### Key functions

**`solve_delta_contraction_single_market`** (`blp_contraction.py`)

Solves the contraction for one market. Uses `jax.lax.fori_loop` with an early-stopping flag so the computation is fully traceable by JAX's autodiff.

**`solve_delta_contraction_batched`** (`blp_contraction.py`)

Applies `jax.vmap` over M markets to solve all contractions in parallel on a single device call. All markets must have the same number of firms (J_per) and workers (N_per).

**`make_components_fn_batched_contraction`** (`run_mle_penalty_phi_sigma_jax.py`)

The JIT-compiled objective function for the contraction path. At each evaluation it:

1. Extracts (tau, alpha, gamma, sigma_e, lambda_e, qbar) from the reduced theta vector
2. Solves delta via `solve_delta_contraction_batched`
3. Evaluates the NLL and penalty moments using the solved deltas
4. Returns the same signature as the joint components function

### Gradient computation

The contraction output is wrapped in `jax.lax.stop_gradient`, so JAX does not differentiate through the contraction iterations. Instead, the solved delta is treated as a constant when computing gradients of the NLL.

This is valid because at the fixed point, the gradient of the objective with respect to the structural parameters flows correctly through the NLL evaluation itself. The contraction merely finds the delta that sets `s_model = s_data`; once found, the NLL depends on the structural parameters both directly and through delta, but the `stop_gradient` eliminates the (expensive) chain through the contraction. This is the standard NFP (nested fixed point) approach used in BLP estimation.

The alternative — implicit differentiation via `jax.custom_vjp` and the implicit function theorem — would give exact gradients including the indirect effect through delta. This is not yet implemented but could improve convergence if needed.

## Usage

### Basic invocation

```bash
python -m code.estimation.run_mle_penalty_phi_sigma_jax \
    --delta_method contraction \
    --skip_statistics \
    --maxiter 500
```

### With frozen parameters

```bash
python -m code.estimation.run_mle_penalty_phi_sigma_jax \
    --delta_method contraction \
    --freeze alpha,gamma,sigma_e \
    --maxiter 100 \
    --skip_statistics
```

Note: `--freeze delta` is a no-op with `--delta_method contraction` (a warning is printed). Deltas are solved internally and cannot be frozen.

### Contraction tuning

```bash
python -m code.estimation.run_mle_penalty_phi_sigma_jax \
    --delta_method contraction \
    --contraction_tol 1e-10 \
    --contraction_maxiter 500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--delta_method` | `joint` | `joint` or `contraction` |
| `--contraction_tol` | `1e-12` | Convergence tolerance |
| `--contraction_maxiter` | `1000` | Max contraction iterations per market |

## Output

The output JSON (`mle_*_estimates_jax.json`) always contains the **full** parameter vector in `theta_hat`, including the solved deltas. When using the contraction method:

- `theta_hat` has length 5 + 2J (deltas are reconstructed via a final contraction solve)
- `delta_method` field is set to `"contraction"`
- `theta0` has length 5 + J (the actual optimization starting point, without deltas)
- Standard errors are not computed (covariance estimation for the contraction path is not yet supported)

The run log (`mle_run_log.csv`) reports `K = 5 + J` for the contraction method, reflecting the actual number of optimized parameters.

## Limitations

1. **Multi-market only**: The contraction method currently requires multi-market data with equal J_per and N_per across markets. Single-market data raises an error.

2. **No standard errors**: Hessian-based covariance estimation is skipped in contraction mode. Implementing proper SEs would require implicit differentiation through the contraction fixed point.

3. **stop_gradient approximation**: The contraction uses `stop_gradient`, so the optimizer does not see the indirect effect of structural parameters on the objective through delta. This is the standard NFP approach, but implicit differentiation would give exact gradients.

4. **Warm starting**: The current implementation initializes delta at zero for each objective evaluation. A warm-start cache (reusing the previous solution) would reduce contraction iterations but adds implementation complexity.
