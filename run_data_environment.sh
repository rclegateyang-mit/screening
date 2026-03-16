#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# Customize these arrays to tweak CLI options.
PREP_ARGS=(
  --N_workers 1000
  --J 2
  --quad_n_x 25
  --quad_n_y 25
  --conduct_mode 1
  --tau 0.4
  --eta 10
  --seed 12345
  --firms_input_path data/raw/input_firms.csv
  --alpha 0.2
  --worker_loc_mode cartesian
  --rho_x_skill_ell_x 0.3
  --rho_x_skill_ell_y 0.3
  --mu_x_skill 12
  --sigma_x_skill 5
  --mu_e 0
  --sigma_e 0
)

SOLVE_ARGS=(
  --conduct_mode 2
  --plot_profits_fixed
  --profit_grid_n 100
  --profit_grid_log_span 0.2
)

NOSCREEN_ARGS=(
  --conduct_mode 2
  --plot_profits_fixed
  --profit_grid_n 100
  --profit_grid_log_span 0.2
  --noscreening
  --noscreening_mode heterogeneous_acceptall
)

WORKERS_ARGS=(
  # Example: --seed 123
)

MARKDOWN_ARGS=(
  --workers_noscreening_path "${repo_root}/data/build/workers_dataset_noscreening.csv"
)

run_module() {
  local module="$1"
  shift
  if (($#)); then
    PYTHONPATH=. python -m "$module" "$@"
  else
    PYTHONPATH=. python -m "$module"
  fi
}

run_module code.data_environment.01_prep_data "${PREP_ARGS[@]}"
run_module code.data_environment.02_solve_equilibrium "${SOLVE_ARGS[@]}"
run_module code.data_environment.02_solve_equilibrium "${NOSCREEN_ARGS[@]}"
run_module code.data_environment.03_draw_workers ${WORKERS_ARGS[@]+"${WORKERS_ARGS[@]}"}
run_module code.data_environment.04_compute_markdowns "${MARKDOWN_ARGS[@]}"
