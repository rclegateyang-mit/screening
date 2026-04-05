# Census RDC Job Harness Guide

Printable reference for running jobs on Census RDC servers.
Bring a hardcopy — there is no internet inside the RDC.

---

## Quick Start (5 commands)

```bash
# 1. Check what's available (first login)
cd /projects/<project_id>/programs
export PYTHONPATH=.
python rdc/env_check.py --save env_report.txt

# 2. Interactive debugging session
qsub -I -X -l select=1:ncpus=2:mem=8gb

# 3. Batch ML job (print script, inspect, then submit)
python -m screening.rdc.submit ml_prod --cmd "python my_gbt.py" --submit

# 4. Batch estimation job
python -m screening.rdc.submit est_prod --cmd "python my_estimation.py" --submit

# 5. Check job status
qstat -a
```

---

## How submit.py Works

`submit.py` generates a PBS bash script from a resource **profile** and a **command**.

**Default behavior: print only.** It shows the script so you can inspect it.
Add `--submit` to actually write the file and run `qsub`.

### Profiles

| Profile | CPUs | Memory | Walltime | Use case |
|---------|------|--------|----------|----------|
| `debug` | 2 | 8 GB | 2h | Interactive debugging |
| `dev` | 4 | 32 GB | 12h | Development batch runs |
| `ml_prod` | 8 | 64 GB | 48h | Production ML (150M obs GBT) |
| `est_prod` | 8 | 64 GB | 48h | Production estimation (100M params) |
| `interactive` | — | — | — | Prints `qsub -I` command to paste |
| `custom` | user | user | user | Specify everything manually |

Override any profile's defaults: `--ncpus 16 --mem 128gb --walltime 72:00:00`

### What the generated script contains

1. PBS resource directives (`#PBS -l select=...`)
2. Environment setup (PYTHONPATH, thread limiting, data/output dirs)
3. **Diagnostic block** — prints hostname, date, job ID, Python version to log
4. Your command
5. Exit code check with **common failure explanations** in the log

### Examples

```bash
# Preview a script (no submission)
python -m screening.rdc.submit ml_prod --cmd "python my_gbt.py"

# Submit with auto-estimated memory for 150M rows
python -m screening.rdc.submit ml_prod --cmd "python gbt.py" --data-rows 150000000

# Custom resources
python -m screening.rdc.submit custom --ncpus 16 --mem 128gb --walltime 72:00:00 \
    --cmd "python big_job.py" --submit

# Set data/output directories
python -m screening.rdc.submit dev --cmd "python test.py" \
    --data-dir /projects/X/data --output-dir /projects/X/data/results --submit
```

---

## Memory Estimation

### Formulas

**ML (sklearn GBT):**
```
raw = rows x cols x 8 bytes
working = raw x 3     (sorting, splits, tree construction)
overhead = 4 GB        (Python, pandas, sklearn internals)
total = working + overhead, round up to next 8 GB
```

**Estimation (scipy optimize):**
```
raw = rows x cols x 8 bytes
working = raw x 2     (gradients, Hessians, copies)
overhead = 4 GB
total = working + overhead, round up to next 8 GB
```

### Worked examples

| Workload | Rows | Cols | Raw | Working | Total | Request |
|----------|------|------|-----|---------|-------|---------|
| GBT | 150M | 10 | 12 GB | 36 GB | 40 GB | **48 GB** |
| GBT | 150M | 20 | 24 GB | 72 GB | 76 GB | **80 GB** |
| GBT | 10M | 10 | 0.8 GB | 2.4 GB | 6.4 GB | **8 GB** |
| Estimation | 100M | 5 | 4 GB | 8 GB | 12 GB | **16 GB** |

### Auto-estimate

```bash
python -m screening.rdc.submit ml_prod --cmd "python x.py" --data-rows 150000000 --data-cols 12
```

This prints the calculation and uses the estimate as the `--mem` default.

### Using float32 to halve memory

If your features don't need float64 precision (most ML tasks):

```python
df = pd.read_csv("data.csv", dtype={col: "float32" for col in feature_cols})
# Halves memory: 150M x 10 cols = 6 GB instead of 12 GB
```

---

## PBS Cheat Sheet

| Command | What it does |
|---------|-------------|
| `qsub script.bash` | Submit a batch job |
| `qsub -I -X -l select=1:ncpus=2:mem=8gb` | Start interactive session |
| `qstat` | Show your jobs |
| `qstat -a` | Show all jobs (all users) |
| `qstat -f JOBID` | Full details of a job |
| `qdel JOBID` | Kill a job |
| `pbsnodes -a` | See available nodes and resources |

**Log files** land in the `--logdir` directory (default `./logs/`):
- `jobname.oJOBID` — combined stdout + stderr (when `#PBS -j oe` is set)

**Interactive sessions timeout after 10-12 hours.** Use batch for overnight runs.

---

## Data Conversion (SAS to CSV)

All Census data arrives in SAS format. Convert before using in Python.

### Quick conversion

```python
from screening.rdc.helpers import sas_to_csv
sas_to_csv("/projects/X/transfer/lehd_file.sas7bdat", "/projects/X/data/lehd.csv")
```

### Read SAS directly in pandas (no conversion)

```python
import pandas as pd
df = pd.read_sas("/projects/X/transfer/lehd_file.sas7bdat")
```

### Chunked reading for large files

```python
from screening.rdc.helpers import read_sas_chunked
for chunk in read_sas_chunked("/projects/X/transfer/big_file.sas7bdat"):
    process(chunk)
```

---

## Disclosure Output

All results leaving the RDC must pass disclosure review. These helpers automate compliance.

### Round to 4 significant digits (Census requirement)

```python
from screening.rdc.helpers import round_to_sig_digits
round_to_sig_digits(3.14159)    # -> 3.142
round_to_sig_digits(0.001234)   # -> 0.001234
round_to_sig_digits(98765.0)    # -> 98760.0
```

### Prepare a full disclosure-ready CSV

```python
from screening.rdc.helpers import prepare_disclosure_csv

# Rounds all numeric columns, flags cells with N < 15, saves both versions
prepare_disclosure_csv(
    df=results_df,
    output_path="disclosure/20260401/results/estimates.csv",
    counts_path="disclosure/20260401/support/unrounded.csv",
    count_col="n_obs",
    min_cell=15,
)
```

### Pseudo-quantiles (instead of true medians/percentiles)

True quantiles reveal actual data values and **cannot be released**.

```python
from screening.rdc.helpers import pseudo_quantile
import numpy as np

wages = np.array(...)  # your data
median = pseudo_quantile(wages, 0.50)    # mean of 11 obs around median
p25 = pseudo_quantile(wages, 0.25)
p75 = pseudo_quantile(wages, 0.75)
```

### Disclosure directory structure

```
/projects/<project_id>/disclosure/YYYYMMDD/
    results/         <- rounded results (CSV only)
    support/         <- unrounded counts, cell sizes (for reviewer)
    programs/        <- code that produced the results
    memo/            <- clearance request memo (TXT)
```

---

## Available Packages

### Definitely available (standard Anaconda)

numpy, scipy, pandas, scikit-learn, matplotlib, statsmodels

### Almost certainly NOT available (must request from RDCA)

JAX, jaxlib, mpi4py, xgboost, lightgbm, tensorflow, pytorch

### Confirm on first login

```bash
python rdc/env_check.py
```

### HistGradientBoosting (critical for 150M obs)

sklearn >= 1.0 includes `HistGradientBoostingRegressor` — a histogram-based GBT
that bins features into 256 buckets. This is **orders of magnitude faster** than
the regular `GradientBoostingRegressor` for large datasets.

```python
# FAST (use this for 150M rows)
from sklearn.ensemble import HistGradientBoostingRegressor
model = HistGradientBoostingRegressor(max_iter=500, max_depth=8)

# SLOW (fallback if sklearn < 1.0 — may take days on 150M rows)
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=500, max_depth=8, subsample=0.1)
# Note: subsample=0.1 uses 10% of data per tree — much faster but noisier
```

If `HistGradientBoostingRegressor` is not available, options:
1. Request sklearn >= 1.0 from your RDCA (best)
2. Use `GradientBoostingRegressor` with `subsample=0.1` (slower, noisier)
3. Request xgboost or lightgbm installation (both handle 150M rows well)

---

## Estimation Without JAX

If JAX is not installed, use scipy for optimization with numerical gradients.

### scipy L-BFGS-B with numerical Jacobian

```python
from scipy.optimize import minimize

result = minimize(
    fun=objective,           # your loss function (numpy, not jax)
    x0=initial_params,
    method="L-BFGS-B",
    jac="3-point",           # numerical gradient via finite differences
    bounds=bounds,
    options={"maxiter": 10000, "ftol": 1e-10, "disp": True},
)
```

### Replace jax.numpy with numpy

```python
# JAX version:                    # numpy fallback:
import jax.numpy as jnp           import numpy as np
from jax.scipy.special import     from scipy.stats import norm
    ndtr                           # ndtr(x) -> norm.cdf(x)
```

### Replace MPI with multiprocessing

```python
from multiprocessing import Pool

def estimate_market(market_data):
    result = minimize(objective, x0, args=(market_data,),
                      method="L-BFGS-B", jac="3-point")
    return result

with Pool(processes=ncpus) as pool:
    results = pool.map(estimate_market, market_data_list)
```

### Replace jax.vmap with loops

```python
# JAX: results = jax.vmap(fn)(batched_args)
# numpy:
results = np.array([fn(args_i) for args_i in batched_args])
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'screening'"

```bash
cd /projects/<project_id>/programs    # must be in the code directory
export PYTHONPATH=.                    # must include current directory
python -c "import screening"          # test it
```

### Job killed with no error message (exit code 137)

Out of memory. The job was killed by the OS.
- Check how much memory you requested: `qstat -f JOBID`
- Request more: `--mem 128gb`
- Use float32 instead of float64 to halve memory
- Process data in chunks instead of loading all at once

### Job stuck in queue ("Q" status)

Not enough resources available. Options:
- `pbsnodes -a` to see what's free
- Request fewer CPUs or less memory
- Wait for other jobs to finish
- Ask your RDCA if there's a queue priority system

### Interactive session died

Hit the 10-12 hour timeout. Use batch jobs for long runs:
```bash
python -m screening.rdc.submit dev --cmd "python my_script.py" --submit
```

### Wrong Python version

```bash
which python          # check which python is active
python --version      # check version
conda activate base   # try activating the base Anaconda environment
```

### "ImportError: cannot import name 'HistGradientBoostingRegressor'"

sklearn version is too old (< 1.0). See the HistGradientBoosting section above.

### Path errors (file not found)

- Census data is in `/data/economic/`, `/data/decennial/`, `/data/demographic/`
- Your project data is in `/projects/<project_id>/transfer/` (placed by Census staff)
- Your working files go in `/projects/<project_id>/data/`
- Use `ls` to explore — file names may differ from documentation

### Disclosure output rejected

- Must be CSV or TXT (no Excel, pickle, HDF5)
- All numbers rounded to 4 significant digits
- Cells with N < 15 must be suppressed
- No true quantiles/percentiles/medians — use `pseudo_quantile()`
- Include unrounded support files for reviewer verification

---

## helpers.py Quick Reference

```python
from screening.rdc.helpers import (
    log, log_section, log_error,       # timestamped logging
    time_block,                         # context manager for timing
    get_peak_rss_mb,                    # current process peak memory
    estimate_pbs_memory,                # memory recommendation
    round_to_sig_digits,                # Census 4-sig-digit rounding
    prepare_disclosure_csv,             # disclosure-ready output
    pseudo_quantile,                    # Census-compliant quantile
    read_sas_chunked,                   # chunked SAS reader
    sas_to_csv,                         # SAS to CSV converter
    set_single_threaded,                # prevent thread explosion
)
```
