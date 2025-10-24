"""
Smoke-test Chamberlain instruments at a supplied theta vector.

- Uses helpers.chamberlain_instruments_numeric (approx_derivative-based).
- Uses your existing probability evaluator (same one used in estimation).
- Asserts shape compatibility and basic numerical sanity.
"""

import json, os, numpy as np, pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from code import get_data_dir
from code.estimation.helpers import (
    chamberlain_instruments_numeric,
    create_probability_evaluator,   # your existing factory
    read_parameters, read_firms_data, read_workers_data,
    compute_worker_firm_distances, compute_cutoffs,
)

# ---- 1) Theta from the user (γ first, then V in NATURAL firm order) ----
theta = np.array([
    0.05,
    3.052995413,
    4.481925254,
    0.614886381,
    4.061756074,
    4.816503231,
    5.920905335,
    3.982567338,
    5.240606644,
    5.023387359,
    5.194043518,
], dtype=float)

# ---- 2) Load data and build the SAME prob evaluator you use in estimation ----
data_dir = get_data_dir(create=True)
params = read_parameters(str(data_dir / "parameters_effective.csv"))
firm_ids, w, Y, A, xi, loc_firms, c = read_firms_data(str(data_dir / "equilibrium_firms.csv"))
x_skill, ell_x, ell_y, chosen_firm = read_workers_data(str(data_dir / "workers_dataset.csv"))

# Compute distances and cutoffs
D = compute_worker_firm_distances(ell_x, ell_y, loc_firms)     # (N,J)
c = compute_cutoffs(w, Y, A, params['beta'])                    # (J,)

# Create probability evaluator with correct arguments
V = params['alpha'] * np.log(w) + xi  # V = α*log w + ξ
prob_evaluator_base = create_probability_evaluator(
    V=V, distances=D, w=w, Y=Y, A=A, c=c, x_skill=x_skill,
    firm_ids=firm_ids, beta=params['beta'], phi=params.get('varphi', 1.0),
    mu_a=params['mu_a'], sigma_a=params['sigma_a']
)
# Wrap to accept BOTH signatures: θ (1D) or (γ, V_nat)
def prob_evaluator_theta_vec(theta_vec: np.ndarray) -> np.ndarray:
    theta_vec = np.asarray(theta_vec, float).ravel()
    gamma = float(theta_vec[0])
    V_nat = theta_vec[1:]
    try:
        # Many implementations already accept 1D θ
        return prob_evaluator_base(theta_vec)
    except TypeError:
        # Fallback to (gamma, V_nat) tuple
        return prob_evaluator_base((gamma, V_nat))

# ---- 3) Basic compatibility checks ----
# 3a) P at theta
P = prob_evaluator_theta_vec(theta)   # expect shape (N, J+1), col 0 = outside
assert P.ndim == 2 and P.shape[1] >= 2, f"Unexpected P shape: {P.shape}"
N, Jp1 = P.shape
J = Jp1 - 1
K = theta.size
assert K == 1 + J, f"theta length {K} != 1+J={1+J}. Check that V has one entry per firm."

# 3b) Probability sanity
row_sums = P.sum(axis=1)
max_dev = float(np.max(np.abs(row_sums - 1.0)))
assert max_dev <= 1e-10, f"Row sums deviate from 1 by {max_dev:.2e}"
assert np.all(P >= -1e-12), "Negative probabilities encountered (beyond tiny numerical noise)."

# ---- 4) Bounds for approx_derivative ----
lb = np.full(K, -np.inf); ub = np.full(K, np.inf)
lb[0] = 0.0;  ub[0] = 1.0     # γ bounds; V_j free
bounds = (lb, ub)

# ---- 5) Compute instruments G at theta ----
G = chamberlain_instruments_numeric(
    theta, prob_evaluator_theta_vec,
    bounds=bounds, rel_step=1e-6, abs_step=1e-5, method="3-point"
)  # shape (N, J, K)

# ---- 6) Report & persist lightweight diagnostics ----
msg = {
    "N": int(N), "J": int(J), "K": int(K),
    "G_shape": tuple(G.shape),
    "P_row_sum_max_deviation": max_dev,
    "G_finite": bool(np.isfinite(G).all()),
    "G_summary": {
        "mean_abs": float(np.mean(np.abs(G))),
        "max_abs": float(np.max(np.abs(G))),
        "pct99_abs": float(np.percentile(np.abs(G), 99.0)),
    },
}
print(json.dumps(msg, indent=2))

# Save a tiny sample for hand inspection
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

# Save theta as CSV
theta_df = pd.DataFrame({
    'parameter': ['gamma'] + [f'V_{j}' for j in range(1, J+1)],
    'value': theta
})
theta_df.to_csv(os.path.join(out_dir, "check_chamberlain_theta_sample.csv"), index=False)

# Save P_head as CSV (first 5 workers)
P_head_df = pd.DataFrame(P[:5], columns=['outside'] + [f'firm_{j}' for j in range(1, J+1)])
P_head_df.insert(0, 'worker', range(5))
P_head_df.to_csv(os.path.join(out_dir, "check_chamberlain_P_sample.csv"), index=False)

# Save G_head as CSV (first 2 workers, all firms, all parameters)
G_head_flat = []
for i in range(2):  # first 2 workers
    for j in range(J):  # all firms
        for k in range(K):  # all parameters
            G_head_flat.append({
                'worker': i,
                'firm': j+1,
                'parameter': 'gamma' if k == 0 else f'V_{k}',
                'derivative': G[i, j, k]
            })
G_head_df = pd.DataFrame(G_head_flat)
G_head_df.to_csv(os.path.join(out_dir, "check_chamberlain_G_sample.csv"), index=False)

print(f"Saved samples to CSV files in {out_dir}:")
print(f"  - check_chamberlain_theta_sample.csv")
print(f"  - check_chamberlain_P_sample.csv") 
print(f"  - check_chamberlain_G_sample.csv")
