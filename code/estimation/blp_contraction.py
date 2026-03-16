"""BLP-style contraction mapping for solving market-level mean utilities (delta).

The contraction mapping profiles out deltas by solving the fixed-point equation:

    delta^{t+1} = delta^{t} + log(s_data) - log(s_model(delta^{t}))

where s_model are the predicted market shares given delta and the other structural
parameters (tau, qbar, gamma, sigma_e, mu_e).

This reduces the MLE parameter space from 5+2J to 5+J by eliminating delta from
the optimisation vector.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax

try:
    from .jax_model import choice_probabilities_from_cutoffs
except ImportError:  # pragma: no cover
    from jax_model import choice_probabilities_from_cutoffs  # type: ignore


def _model_shares(
    tau: jnp.ndarray,
    delta: jnp.ndarray,
    qbar: jnp.ndarray,
    X_m: jnp.ndarray,
    D_m: jnp.ndarray,
    gamma: jnp.ndarray,
    sigma_e: jnp.ndarray,
    mu_e: jnp.ndarray,
) -> jnp.ndarray:
    """Compute predicted market shares for a single market.

    Returns
    -------
    shares : (J_per,)
        Mean choice probabilities across workers for each firm (excluding outside).
    """
    aux = {
        "D_nat": D_m,
        "gamma": gamma,
        "sigma_e": sigma_e,
        "mu_e": mu_e,
    }
    P_nat = choice_probabilities_from_cutoffs(tau, delta, qbar, X_m, aux)
    return jnp.mean(P_nat[:, 1:], axis=0)


# Vectorized share computation across markets
_model_shares_vmapped = jax.vmap(
    _model_shares,
    in_axes=(None, 0, 0, 0, 0, None, None, None),
)


def solve_delta_contraction_single_market(
    tau: jnp.ndarray,
    qbar: jnp.ndarray,
    X_m: jnp.ndarray,
    D_m: jnp.ndarray,
    shares_emp: jnp.ndarray,
    delta_init: jnp.ndarray,
    gamma: jnp.ndarray,
    sigma_e: jnp.ndarray,
    mu_e: jnp.ndarray,
    tol: float = 1e-12,
    maxiter: int = 1000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve for delta via the BLP contraction mapping in a single market.

    Uses jax.lax.while_loop for JIT compatibility (e.g. for unit tests).
    """
    log_s_data = jnp.log(jnp.maximum(shares_emp, 1e-300))
    maxiter_arr = jnp.array(maxiter, dtype=jnp.int32)

    def cond_fn(carry):
        _, err, n_iters = carry
        return (err > tol) & (n_iters < maxiter_arr)

    def body_fn(carry):
        delta, _, n_iters = carry
        s_model = _model_shares(tau, delta, qbar, X_m, D_m, gamma, sigma_e, mu_e)
        log_s_model = jnp.log(jnp.maximum(s_model, 1e-300))
        diff = log_s_data - log_s_model
        return delta + diff, jnp.max(jnp.abs(diff)), n_iters + 1

    init_carry = (delta_init, jnp.array(1e10, dtype=delta_init.dtype), jnp.array(0, dtype=jnp.int32))
    delta_final, err_final, _ = lax.while_loop(cond_fn, body_fn, init_carry)
    converged = (err_final <= tol).astype(jnp.float64)
    return delta_final, converged


@jax.jit
def _contraction_step(
    delta_batch: jnp.ndarray,
    tau: jnp.ndarray,
    qbar_batch: jnp.ndarray,
    X_batch: jnp.ndarray,
    D_batch: jnp.ndarray,
    log_s_data_batch: jnp.ndarray,
    gamma: jnp.ndarray,
    sigma_e: jnp.ndarray,
    mu_e: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One contraction step across all M markets. Returns (delta_new, max_err)."""
    s_model_batch = _model_shares_vmapped(
        tau, delta_batch, qbar_batch, X_batch, D_batch,
        gamma, sigma_e, mu_e,
    )
    log_s_model = jnp.log(jnp.maximum(s_model_batch, 1e-300))
    diff = log_s_data_batch - log_s_model
    return delta_batch + diff, jnp.max(jnp.abs(diff))


def solve_delta_contraction_batched(
    delta_init_batch: jnp.ndarray,
    tau: jnp.ndarray,
    qbar_batch: jnp.ndarray,
    X_batch: jnp.ndarray,
    D_batch: jnp.ndarray,
    shares_emp_batch: jnp.ndarray,
    gamma: jnp.ndarray,
    sigma_e: jnp.ndarray,
    mu_e: jnp.ndarray,
    tol: float = 1e-12,
    maxiter: int = 1000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve BLP contraction across M markets using a Python-level loop.

    Uses a JIT-compiled per-step function with vmapped share computation.
    The Python loop avoids expensive XLA compilation of while_loop(vmap).

    NOTE: This function must NOT be called inside a JAX-traced context
    (jit/vmap/etc.) because it uses a Python for-loop with float().
    The caller must use jax.lax.stop_gradient on the result if the
    surrounding computation is being differentiated.

    Returns
    -------
    delta_batch : (M, J_per)
    converged_batch : (M,) float flags
    """
    log_s_data_batch = jnp.log(jnp.maximum(shares_emp_batch, 1e-300))

    delta = delta_init_batch
    for it in range(maxiter):
        delta, err = _contraction_step(
            delta, tau, qbar_batch, X_batch, D_batch,
            log_s_data_batch, gamma, sigma_e, mu_e,
        )
        if float(err) <= tol:
            break

    # Per-market convergence flags
    s_final = _model_shares_vmapped(
        tau, delta, qbar_batch, X_batch, D_batch,
        gamma, sigma_e, mu_e,
    )
    log_diff = jnp.abs(log_s_data_batch - jnp.log(jnp.maximum(s_final, 1e-300)))
    per_market_err = jnp.max(log_diff, axis=1)
    converged_batch = (per_market_err <= tol).astype(jnp.float64)

    return delta, converged_batch
