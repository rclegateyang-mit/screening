# Project Layout

## Repository root (`/proj/screening/rcly/`)

```
CLAUDE.md               Project instructions for AI assistants
config/defaults.yaml    Canonical simulation parameters (single source of truth)
data/                   Input data (raw, clean, build stages)
code/                   Python package, scripts, tests
output/                 All generated outputs
docs/                   Documentation
archive/                Legacy material
```

## `data/`

```
data/raw/           Raw/input parameters (parameters_effective.csv, firms.csv)
data/clean/         Equilibrium outputs (equilibrium_firms.csv, per-market CSVs)
data/build/         Built datasets (workers_dataset.csv, per-market CSVs)
```

May contain scenario subfolders (e.g. `data/clean/test_reparam/`). Top-level files are the canonical dataset.

Paths configurable via `code/path_config.json` or env vars `SCREENING_DATA_DIR` / `SCREENING_OUTPUT_DIR`.

## `code/`

```
code/screening/             Main Python package (importable as `screening`)
code/code/                  Backward-compat shim (redirects to screening)
code/tests/                 Test suite
code/requirements.txt
code/path_config.json       Data/output directory configuration
code/run_data_environment.sh
code/run_mle_per_market.sh
```

Always run from `code/` with `PYTHONPATH=.`.

## `code/screening/` (Python package)

```
__init__.py              Package root; path configuration (get_data_dir, get_output_dir, etc.)
simulate/                Data generation (raw data)
  01_prep_data.py        Generate firms, support points, parameters
clean/                   Raw → clean (equilibrium solving)
  02_solve_equilibrium.py   Solve firm FOCs for wages and screening cutoffs
build/                   Clean → build (worker draws, markdowns)
  03_draw_workers.py     Draw worker characteristics, construct dataset
  04_compute_markdowns.py   Compute implied wage markdowns
analysis/                Estimation using build data
  lib/                   Shared model/estimation library
    helpers.py           Data loading, distance computation, initial guesses
    jax_model.py         JAX choice probabilities and penalty (structural parametrization)
    model_components.py  Spec M choice probs, average skill, GMM moments (standardized)
    blp_contraction.py   BLP contraction mapping for delta profiling
    naive_init.py        Pooled naive initialization (6-step data-driven procedure)
  mle/                   MLE estimators
    run_pooled.py        Single-process penalized MLE
    run_distributed.py   Distributed MLE entry point (MPI)
    distributed_master.py   Outer L-BFGS loop over globals
    distributed_worker.py   Per-market inner solve + Schur complement
  auglag/                Augmented Lagrangian hybrid MLE+GMM
    run_pooled.py        Single-process auglag solver
    run_distributed.py   Distributed auglag (MPI)
    master.py            Outer auglag loop orchestration
    worker.py            Per-market subproblem solve
  gmm/                   GMM estimation
    run_from_mle.py      Standalone GMM from MLE inner estimates
  scaling/               Benchmarking and scaling tests
    benchmark.py         MLE scaling in (N, J)
    run_nj_test.py       N/J recovery test
    run_two_step.py      Two-step scaling test (MLE + GMM)
LEHD/                    LEHD data processing (SAS scripts)
```

### Distributed MLE architecture (MPI)

Profiles out per-market parameters `(delta_m, qbar_m)` via parallel MPI ranks.
Rank 0 optimises global parameters `theta_G = (tau, tilde_gamma)` using
aggregated gradient and Schur-complement Hessian from all ranks.

```
# Typical: 40 ranks x 1 thread = 40 CPUs on a 48-core node
mpirun -np 40 python -m screening.analysis.mle.run_distributed \
    --workers_path /path/to/workers_dataset.csv \
    --firms_path /path/to/equilibrium_firms.csv \
    --params_path /path/to/parameters_effective.csv \
    --inner_maxiter 200 --inner_tol 1e-6 \
    --outer_maxiter 50 --outer_tol 1e-5 \
    --M 100
```

### DGP: firm fundamentals decomposition

```
logA_j = z_j^1 + nu_j^A     (TFP decomposition)
xi_j   = z_j^2 + nu_j^xi    (amenity decomposition)

z_j^1 ~ N(0, sigma_z1^2)    observed TFP shifter
z_j^2 ~ N(0, sigma_z2^2)    observed amenity shifter
```

### Per-market data files

**Firm data** (`equilibrium_firms_market_m.csv`): firm_id, w, qbar, L, Q, Y, A, xi, z1, z2, x, y

**Worker data** (`workers_dataset_market_m.csv`): worker_id, x_skill, ell_x, ell_y, chosen_firm

## `output/`

```
output/equilibrium/     Equilibrium solver outputs
output/estimation/      MLE/auglag/GMM results (JSON estimates, plots, run logs)
output/workers/         Worker-level outputs
output/markdowns/       Markdown reports
output/scaling/         Scaling benchmarks
  m_scaling/            M-scaling results
  nj_scaling/           N/J scaling results
  m1000_scaling_results/   M=1000 full scaling results
  m1000_estimation_test/   Estimation test outputs
```

## `docs/`

```
docs/methods/           Estimation methodology (tex, pdf, md)
  estimation.tex        Main estimation writeup
  mle_objective.md      MLE objective and penalty structure
  model_components.md   Standardized Spec M components
  blp_contraction_mapping.md   BLP contraction procedure
  naive_initialization.md      Naive init documentation
  project_layout.md     This file
docs/datastructure/     Data documentation
  LEHD/                 LEHD data variable descriptions
  LBD/                  LBD data documentation
docs/notes/             Meeting notes and research questions
```

## `archive/`

```
archive/old_project/         Original archive contents
archive/code_legacy/         Old estimation code (pre-restructure)
archive/code_old.zip         Archived code snapshot
archive/data_m100_baseline/  Old M=100 baseline dataset
archive/data_v2/             Old data_v2 dataset
archive/test_reparam/        Reparametrization test data
archive/output_scaling/      Old scaling outputs
archive/old_notes/           Old instruction text files
```
