# Project Layout

## Repository root (`/proj/screening/rcly/`)

```
code/               Python package, scripts, tests, and docs
data/               Data files (raw, clean, build stages)
output/             Generated outputs (equilibrium, estimation, etc.)
archive/            Archived/old files
```

## `code/`

```
code/code/          Main Python package (importable as `code`)
code/docs/          Project documentation
code/tests/         Test suite
code/requirements.txt
code/path_config.json       Data/output directory configuration
code/run_data_environment.sh
code/run_mle_per_market.sh
```

## `code/code/` (Python package)

```
__init__.py         Package root; path configuration (get_data_dir, get_output_dir, etc.)
data_environment/   Simulation: generates firms, workers, equilibrium
estimation/         MLE estimation code
LEHD/               LEHD data processing
```

### `code/code/estimation/`

Core estimation module:

| File | Purpose |
|------|---------|
| `helpers.py` | Data loading, distance computation, initial guesses |
| `jax_model.py` | JAX choice probabilities and penalty components |
| `blp_contraction.py` | BLP contraction mapping for delta profiling |
| `run_mle_penalty_phi_sigma_jax.py` | Main MLE script (single-process entry point) |
| `distributed_worker.py` | Per-market inner optimization + Schur complement extraction |
| `distributed_master.py` | Outer Newton/L-BFGS loop over 5 global parameters |
| `run_distributed_mle.py` | Distributed MLE entry point (multi-process) |
| `benchmark_scaling.py` | Performance benchmarks |

#### Distributed MLE architecture (MPI)

Profiles out per-market parameters `(δ_m, q̄_m)` via parallel MPI ranks.
Rank 0 optimises 5 global parameters `θ_G = (τ, α, γ, σ_e, λ_e)` using
aggregated gradient and Schur-complement Hessian from all ranks.

**Data persistence:** Rank 0 loads data and calls `comm.scatter()` to send
each rank only its partition of markets.  With M=1000 markets and 96 ranks,
each rank holds ~10 markets (~20 MB) — not the full 2 GB.  No data is
re-serialised during the outer loop; only `θ_G` (5 floats) is broadcast
per iteration.

**Communication per iteration:** `Bcast(θ_G)` + `Reduce(nll, grad, H)` =
~240 bytes via tree collectives.

**Naive initialization:** `compute_naive_init()` in `distributed_worker.py`
derives starting values from data alone (no true parameters needed):
- tau, delta: MNL logit of choices on distance + firm FE (scipy L-BFGS-B)
- qbar: screening FOC using observed wages, revenue, and average skill

Supports heterogeneous J across markets (no equal-J requirement).

**CPU allocation:** Total CPU usage = P (MPI ranks) × T (threads per rank).
The script defaults T=1 for all thread pools (OpenBLAS, XLA/Eigen, OpenMP)
because per-market matrices are small (~90×90) and MPI-level parallelism
is more efficient. Without pinning, each rank spawns up to 64 OpenBLAS
threads, causing severe oversubscription (e.g. 40 ranks × 64 = 2560 threads).
To override, set env vars before `mpirun`:
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 mpirun -np 10 ...` (10×4 = 40 CPUs).

```
# Typical: 40 ranks × 1 thread = 40 CPUs on a 48-core node
mpirun -np 40 python -m code.estimation.run_distributed_mle \
    --workers_path /path/to/workers_dataset.csv \
    --firms_path /path/to/equilibrium_firms.csv \
    --params_path /path/to/parameters_effective.csv \
    --inner_maxiter 200 --inner_tol 1e-6 \
    --outer_maxiter 50 --outer_tol 1e-5 \
    --freeze alpha,gamma,sigma_e,lambda_e \
    --M 100
```

### `code/code/data_environment/`

Simulation pipeline that generates synthetic data: parameters, firm locations, worker draws, and equilibrium solving.

## `data/`

```
data/raw/           Raw/input parameters (e.g. parameters_effective.csv)
data/clean/         Cleaned data (e.g. equilibrium_firms.csv)
data/build/         Built datasets (e.g. workers_dataset.csv)
```

Paths are configurable via `code/path_config.json` or environment variables `SCREENING_DATA_DIR` / `SCREENING_OUTPUT_DIR`.

## `data_scaling/`

Large-scale simulated dataset (M=1000 markets, N=2000 workers/market, J=76–96 firms/market after filtering). Used for distributed MLE scaling tests.

```
data_scaling/raw/               Parameters (parameters_effective.csv)
data_scaling/clean/             Equilibrium firms
data_scaling/build/             Combined workers_dataset.csv + per-market CSVs
data_scaling/scaling_results/   Estimation outputs by (M, P) combo
    scaling_summary.md          Consolidated results table
    comparison_table.md         Detailed comparison (pooled vs distributed)
    M{x}_P{y}/                  Per-run results (JSON estimates, inner estimates, run log)
```

## `output/`

```
output/equilibrium/ Equilibrium solver outputs
output/estimation/  MLE results (JSON estimates, plots, run logs)
output/workers/     Worker-level outputs
output/markdowns/   Generated reports
```

## `code/docs/`

| File | Topic |
|------|-------|
| `project_layout.md` | This file |
| `mle_objective.md` | MLE objective function and penalty structure |
| `blp_contraction_mapping.md` | BLP contraction mapping procedure |
| `LEHD/` | LEHD data documentation |
