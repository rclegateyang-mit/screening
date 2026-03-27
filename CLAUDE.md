# CLAUDE.md

## Project

Worker screening model: simulation, equilibrium solving, and penalized MLE estimation using JAX.

## Directory conventions

The project uses a strict 4-directory layout:

| Directory | Purpose | Rule |
|-----------|---------|------|
| `data/` | Input data (`raw/`, `clean/`, `build/`) | Never put results here |
| `code/` | All source code (`screening/` package, tests, scripts) | Run from `code/` with `PYTHONPATH=.` |
| `output/` | All estimation results, scaling benchmarks, plots | All outputs go here |
| `docs/` | Documentation (`methods/`, `datastructure/`, `notes/`) | |

- **Scenario subfolders**: `data/{raw,clean,build}/` and `output/` may contain one-off scenario subfolders (e.g. `data/clean/test_reparam/`, `output/estimation/nj_experiment/`). Top-level files in each directory are the canonical/default dataset.
- **`SCREENING_DATA_DIR`** / **`SCREENING_OUTPUT_DIR`** env vars override the entire data/output root for alternative runs.
- `archive/` holds all legacy material (old data, old code, old notes).

## Project layout

See [docs/methods/project_layout.md](docs/methods/project_layout.md) for a full directory map.

## Default parameters

See [config/defaults.yaml](config/defaults.yaml) for canonical simulation parameter values.
All data generation and estimation should use these unless explicitly overridden.

Key values: tau=0.4, alpha=0.2, gamma0=0.76, gamma1=0.94, eta=7.0, sigma_e=0.135, mu_v=12.0, sigma_v=0.25, J=100, N_workers=2000, M=1000.

## Python package

The package is `screening` (under `code/screening/`). Always run from `code/` with `PYTHONPATH=.`.

### Key entry points

- **MLE estimation (pooled)**: `python -m screening.analysis.mle.run_pooled`
- **MLE estimation (distributed)**: `mpirun -np P python -m screening.analysis.mle.run_distributed`
- **Augmented Lagrangian (distributed)**: `mpirun -np P python -m screening.analysis.auglag.run_distributed`
- **Data generation**: `bash run_data_environment.sh`

### Package structure

```
code/screening/
├── __init__.py              # Path config (get_data_dir, get_output_dir, etc.)
├── simulate/                # Data generation (raw data)
│   └── 01_prep_data.py
├── clean/                   # Raw → clean (equilibrium solving)
│   └── 02_solve_equilibrium.py
├── build/                   # Clean → build (worker draws, markdowns)
│   ├── 03_draw_workers.py
│   └── 04_compute_markdowns.py
└── analysis/                # Estimation using build data
    ├── lib/                 # Shared model/estimation library
    │   ├── helpers.py, jax_model.py, model_components.py
    │   ├── blp_contraction.py, naive_init.py
    ├── mle/                 # MLE estimators
    │   ├── run_pooled.py, run_distributed.py
    │   ├── distributed_master.py, distributed_worker.py
    ├── auglag/              # Augmented Lagrangian estimators
    │   ├── run_pooled.py, run_distributed.py
    │   ├── master.py, worker.py
    ├── gmm/                 # GMM from MLE
    │   └── run_from_mle.py
    └── scaling/             # Benchmarking/scaling tests
        ├── benchmark.py, run_nj_test.py, run_two_step.py
```

### Where to put new code

- New data processing scripts: `code/screening/{simulate,clean,build}/`
- New estimation code: `code/screening/analysis/`
- Shared model utilities: `code/screening/analysis/lib/`
- Outputs: always in `output/`, never in `data/`

## Conventions

- JAX with float64 (`jax_enable_x64`) throughout estimation code
- Parameter vector order: `[tau, alpha, gamma1, sigma_e, lambda_e, delta_1..J, ln_qbar_1..J]`
- Multi-market data requires equal J_per and N_per across markets (for `jax.vmap`)
- Transforms map bounded parameters to unconstrained space for L-BFGS optimization
- `config/defaults.yaml` is the single source of truth for parameters
