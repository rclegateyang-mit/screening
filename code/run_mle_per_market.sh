#!/usr/bin/env bash
set -euo pipefail

# Run MLE estimation per market with frozen params (pure MLE).
# Usage: bash run_mle_per_market.sh [M] [maxiter]
#   M       = number of markets (default: read from data)
#   maxiter = max LBFGS iterations (default: 200)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

M="${1:-50}"
MAXITER="${2:-200}"

DATA_DIR="/proj/screening/rcly/data"
OUTPUT_DIR="/proj/screening/rcly/output"
MARKETS_BUILD="${DATA_DIR}/build/markets"
MARKETS_CLEAN="${DATA_DIR}/clean/markets"
PARAMS_PATH="${DATA_DIR}/raw/parameters_effective.csv"
MLE_RESULTS="${OUTPUT_DIR}/estimation/markets"

mkdir -p "$MLE_RESULTS"

echo "Per-market MLE benchmark: M=${M}, maxiter=${MAXITER}"
echo "  freeze: gamma,sigma_e,lambda_e"
echo "  penalty_weight: 0.0"
echo "  theta0: from helper (MNL + screening FOC)"
echo "========================================"

RESULTS_CSV="${MLE_RESULTS}/mle_scaling_results.csv"
echo "market_id,N,J,K,build_time,solve_time,total_time,objective,nll,penalty,nit,grad_norm,tau_error,alpha_error,delta_l2,qbar_l2,success" > "$RESULTS_CSV"

TOTAL_START=$(date +%s)

for m in $(seq 1 "$M"); do
    WORKERS="${MARKETS_BUILD}/workers_dataset_market_${m}.csv"
    FIRMS="${MARKETS_CLEAN}/equilibrium_firms_market_${m}.csv"
    OUT="${MLE_RESULTS}/market_${m}"

    if [[ ! -f "$WORKERS" ]] || [[ ! -f "$FIRMS" ]]; then
        echo "  Market ${m}: SKIP (files missing)"
        continue
    fi

    mkdir -p "$OUT"

    START=$(date +%s%N)
    PYTHONPATH=. python -m screening.analysis.mle.run_pooled \
        --workers_path "$WORKERS" \
        --firms_path "$FIRMS" \
        --params_path "$PARAMS_PATH" \
        --out_dir "$OUT" \
        --freeze gamma,sigma_e,lambda_e \
        --penalty_weight 0.0 \
        --theta0_from_helper \
        --maxiter "$MAXITER" \
        --skip_statistics \
        --skip_plot \
        > "${OUT}/stdout.log" 2>&1
    RC=$?
    END=$(date +%s%N)
    WALL_MS=$(( (END - START) / 1000000 ))

    if [[ $RC -eq 0 ]]; then
        # Extract metrics from JSON output
        JSON="${OUT}/mle_tau_alpha_gamma_sigma_e_lambda_e_delta_qbar_penalty_estimates_jax.json"
        if [[ -f "$JSON" ]]; then
            ROW=$(python3 -c "
import json, sys
with open('${JSON}') as f:
    d = json.load(f)
t = d['timings']
dm = d['distance_metrics']
J = (len(d['theta_hat']) - 5) // 2
N = int(d.get('market_shares', {}).get('empirical', [0]).__len__() and sum(1 for _ in open('${WORKERS}')) - 1)
print(f'${m},{N},{J},{5+2*J},{t[\"build_time_sec\"]:.2f},{t[\"solve_time_sec\"]:.2f},{t[\"total_time_sec\"]:.2f},{d[\"objective\"]:.4f},{d[\"objective_breakdown\"][\"neg_log_likelihood\"]:.4f},{d[\"objective_breakdown\"][\"penalty\"]:.4f},{d[\"nit\"]},{d[\"grad_norm\"]:.3e},{dm[\"tau_error\"]:.6f},{dm[\"alpha_error\"]:.6f},{dm[\"delta_l2\"]:.6f},{dm[\"qbar_l2\"]:.6f},True')
")
            echo "$ROW" >> "$RESULTS_CSV"
            echo "  Market ${m}: OK (${WALL_MS}ms wall)"
        else
            echo "  Market ${m}: OK but no JSON output"
        fi
    else
        echo "  Market ${m}: FAIL (rc=${RC})"
        echo "${m},,,,,,,,,,,,,,,False" >> "$RESULTS_CSV"
    fi
done

TOTAL_END=$(date +%s)
TOTAL_SEC=$(( TOTAL_END - TOTAL_START ))
echo ""
echo "========================================"
echo "Done: ${M} markets in ${TOTAL_SEC}s"
echo "Results: ${RESULTS_CSV}"
echo ""
echo "Summary:"
head -1 "$RESULTS_CSV"
tail -n +2 "$RESULTS_CSV" | head -5
echo "..."
echo ""
# Print timing stats
python3 -c "
import pandas as pd
df = pd.read_csv('${RESULTS_CSV}')
ok = df[df['success']==True]
print(f'Successful: {len(ok)}/{len(df)}')
if len(ok) > 0:
    print(f'solve_time: mean={ok[\"solve_time\"].mean():.1f}s, median={ok[\"solve_time\"].median():.1f}s, max={ok[\"solve_time\"].max():.1f}s')
    print(f'total_time: mean={ok[\"total_time\"].mean():.1f}s, median={ok[\"total_time\"].median():.1f}s, max={ok[\"total_time\"].max():.1f}s')
    print(f'tau_error:  mean={ok[\"tau_error\"].abs().mean():.4f}')
    print(f'delta_l2:   mean={ok[\"delta_l2\"].mean():.4f}')
    print(f'qbar_l2:    mean={ok[\"qbar_l2\"].mean():.4f}')
"
