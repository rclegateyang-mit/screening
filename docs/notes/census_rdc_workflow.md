# Census RDC Workflow Guide (Project-Specific)

Practical workflow for running the screening model estimation on Census RDC servers.
Read alongside: `census_rdc_computing_environment.md` and `census_rdc_disclosure_rules.md`.

---

## Getting Code Into the RDC

1. **Email code + Program Upload Form** to your RDCA at least **7 days before your visit**.
2. Code must NOT contain data.
3. RDCA reviews and uploads to `/projects/<project_id>/programs/`.
4. Print code on paper as reference (can bring hardcopies).
5. No laptops, USB drives, or portable storage allowed.

### Code preparation checklist (before submitting)

- [ ] Use **relative paths only** (no `/home/rcly/...` or full machine paths).
- [ ] Remove any hardcoded data values or confidential information.
- [ ] Remove user IDs from comments.
- [ ] Comments don't reveal data characteristics.
- [ ] All dependencies documented (which packages are needed).
- [ ] Config file specifies all parameters (no magic numbers in code).

---

## First Login Checklist

Run these immediately on first login to characterize the environment:

```bash
# System info
uname -a
cat /etc/redhat-release

# Python environment
python --version
python3 --version
which python
conda --version
conda list                    # full package list
conda list numpy scipy pandas matplotlib  # key packages
pip list 2>/dev/null          # may not have pip

# Check for JAX (probably not available)
python -c "import jax; print(jax.__version__)" 2>&1
python -c "import jaxlib; print(jaxlib.__version__)" 2>&1

# Check for MPI
which mpirun 2>/dev/null
python -c "import mpi4py; print(mpi4py.__version__)" 2>&1

# Check multiprocessing
python -c "import multiprocessing; print(multiprocessing.cpu_count())"

# Check for useful packages
python -c "import scipy; print(scipy.__version__)"
python -c "import pandas; print(pandas.__version__)"
python -c "import statsmodels; print(statsmodels.__version__)" 2>&1
python -c "import sklearn; print(sklearn.__version__)" 2>&1

# PBS Pro
which qsub
pbsnodes -a    # see available compute nodes and resources

# Project directory structure
ls -la /projects/
ls -la /projects/<project_id>/
ls -la /projects/<project_id>/transfer/  # check if data has been placed

# Data files
ls -la /data/economic/     # or wherever LEHD/LBD data lives

# Shared code library
ls /data/support/researcher/codelib/

# Shared Stata .ado files
ls /apps/shared/stata/
```

---

## Data Preparation Workflow

Census data arrives in SAS format. Convert before using in Python.

### Option A: SAS export to CSV

```sas
/* Run as a batch job via qsas */
libname mydata '/projects/<project_id>/transfer/';

proc export data=mydata.lehd_file
    outfile="/projects/<project_id>/data/lehd_data.csv"
    dbms=csv replace;
run;
```

### Option B: Read SAS directly in Python

```python
import pandas as pd

# pandas can read SAS7BDAT files directly
df = pd.read_sas('/projects/<project_id>/transfer/lehd_file.sas7bdat')

# For large files, read in chunks
reader = pd.read_sas('/projects/<project_id>/transfer/lehd_file.sas7bdat',
                      chunksize=100_000)
for chunk in reader:
    process(chunk)
```

### Option C: Use Stat/Transfer

```bash
# Convert SAS to Stata (then read .dta in pandas)
# Check Stat/Transfer docs on RDC intranet: http://rdcdoc.cods.census.gov
```

---

## Running Estimation Jobs

### Interactive debugging (small data subset)

```bash
# Get an interactive compute node
qsub -IX

# Activate Python (required on compute nodes — not on PATH by default)
source /apps/anaconda/bin/activate py3cf

# Navigate to project
cd /projects/<project_id>/programs

# Run on small subset
python run_estimation.py --markets 5 --debug
```

### Batch production run

```bash
#!/bin/bash
#PBS -N screening_mle
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o /projects/<project_id>/logs/
#PBS -e /projects/<project_id>/logs/

cd /projects/<project_id>/programs

# Activate Python and set environment
source /apps/anaconda/bin/activate py3cf
export PYTHONPATH=.

# Run estimation
python -m screening.analysis.mle.run_pooled \
    --data_dir /projects/<project_id>/data \
    --output_dir /projects/<project_id>/data/results
```

Submit: `qsub run_mle.bash &`

### Multi-core PBS job

```bash
#!/bin/bash
#PBS -N screening_multicore
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=48:00:00

cd /projects/<project_id>/programs
source /apps/anaconda/bin/activate py3cf
export PYTHONPATH=.

# Use Python multiprocessing (if MPI unavailable)
python run_estimation_multiprocess.py --workers 8
```

### MPI job (if mpi4py is available)

```bash
#!/bin/bash
#PBS -N screening_distributed
#PBS -l select=1:ncpus=20:mem=64gb
#PBS -l walltime=48:00:00

cd /projects/<project_id>/programs
source /apps/anaconda/bin/activate py3cf
export PYTHONPATH=.

mpirun -np 20 python -m screening.analysis.mle.run_distributed
```

---

## Project-Specific: JAX Fallback Plan

If JAX is not available, the code needs these adaptations:

### 1. Replace JAX arrays with numpy

```python
# Instead of:
import jax.numpy as jnp
x = jnp.array(data)

# Use:
import numpy as np
x = np.array(data)
```

### 2. Replace JAX autodiff with numerical gradients

```python
from scipy.optimize import approx_fprime

def objective(params, data):
    """Pure numpy objective function."""
    # ... same math, numpy instead of jnp ...
    return nll

def grad_objective(params, data):
    """Numerical gradient via finite differences."""
    return approx_fprime(params, objective, 1e-7, data)
```

Or use `scipy.optimize.minimize` with `jac='2-point'` or `jac='3-point'`:

```python
from scipy.optimize import minimize

result = minimize(
    objective, x0, args=(data,),
    method='L-BFGS-B',
    jac='3-point',           # numerical Jacobian
    bounds=bounds,
    options={'maxiter': 10000, 'ftol': 1e-10}
)
```

### 3. Replace distributed MPI with multiprocessing

```python
from multiprocessing import Pool

def estimate_market(market_data):
    """Single-market estimation (no JAX, no MPI)."""
    result = minimize(objective, x0, args=(market_data,),
                      method='L-BFGS-B', jac='3-point')
    return result

# Parallel across markets within a single PBS node
with Pool(processes=ncpus) as pool:
    results = pool.map(estimate_market, market_data_list)
```

### 4. Replace jax.vmap with explicit loops or vectorized numpy

```python
# Instead of jax.vmap(fn)(batched_args):
results = np.array([fn(args_i) for args_i in batched_args])

# Or vectorize with numpy broadcasting where possible
```

---

## Preparing Output for Disclosure Review

### Directory setup

```bash
mkdir -p /projects/<project_id>/disclosure/YYYYMMDD
chmod -R g+rwx /projects/<project_id>/disclosure/YYYYMMDD
```

### What to place in the disclosure directory

```
disclosure/YYYYMMDD/
    results/
        estimation_results.csv          # main results (rounded to 4 sig digits)
        regression_table_1.csv          # formatted for paper
    support/
        sample_counts.csv               # unweighted N per sample/subsample
        cell_counts.csv                 # unweighted counts per cell
        rounding_verification.csv       # both rounded and unrounded values
        concentration_ratios.csv        # for economic data
    programs/
        run_estimation.py               # code that produced the output
        prepare_output.py               # code for disclosure stats
    memo/
        clearance_request_memo.txt      # describes what's being released and why
```

### Auto-generate disclosure statistics

Build this into your estimation code:

```python
import numpy as np

def round_to_sig_digits(x, n=4):
    """Round to n significant digits (Census requirement)."""
    if x == 0:
        return 0.0
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

def prepare_disclosure_output(results_df, sample_df):
    """Prepare estimation results for disclosure review."""

    # 1. Round all estimates to 4 significant digits
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    rounded = results_df.copy()
    for col in numeric_cols:
        rounded[col] = rounded[col].apply(lambda x: round_to_sig_digits(x, 4))

    # 2. Compute unweighted sample counts
    sample_counts = sample_df.groupby('subsample').size()
    flagged = sample_counts[sample_counts < 15]
    if len(flagged) > 0:
        print(f"WARNING: {len(flagged)} subsamples have N < 15 — must suppress")

    # 3. Save both rounded and unrounded
    results_df.to_csv('support/unrounded_results.csv', index=False)
    rounded.to_csv('results/estimation_results.csv', index=False)
    sample_counts.to_csv('support/sample_counts.csv')

    return rounded, sample_counts
```

### Pseudo-quantile computation

```python
def pseudo_quantile(data, q, window=5):
    """Compute Census-compliant pseudo-quantile.

    Takes mean of observation at quantile q plus `window` observations
    on each side (minimum 2*window + 1 = 11 observations total).
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    idx = int(q * n)
    lo = max(0, idx - window)
    hi = min(n, idx + window + 1)
    # Ensure at least 11 observations
    assert hi - lo >= 2 * window + 1, f"Not enough observations for pseudo-quantile"
    return np.mean(sorted_data[lo:hi])

# Example: pseudo-median and pseudo-IQR
p25 = pseudo_quantile(wages, 0.25, window=5)
p50 = pseudo_quantile(wages, 0.50, window=5)
p75 = pseudo_quantile(wages, 0.75, window=5)
```

---

## Typical Session Workflow

```
1. Badge in, sit at thin client, connect via NX
2. Open Konsole terminal
3. ssh to login node (if not already there)
4. qsub -I -X -l select=1:ncpus=2:mem=8gb   # interactive compute node
5. cd /projects/<project_id>/programs
6. Debug on small data subset
7. Once working: write PBS batch script
8. exit  # leave interactive session
9. qsub run_estimation.bash &
10. qstat  # monitor
11. Check logs in /projects/<project_id>/logs/
12. Prepare output for disclosure in /projects/<project_id>/disclosure/YYYYMMDD/
13. Notify RDCA when ready for review
```

---

## Common Pitfalls

- **Forgetting to use compute nodes**: Don't run analyses on login nodes. Always `qsub`.
- **SAS data format**: Budget time for data conversion on first visit.
- **No internet for debugging**: Print error messages / Stack Overflow answers beforehand.
- **Package version mismatch**: Test code locally with the same Python/numpy/scipy versions
  as the RDC before uploading.
- **Interactive session timeout**: 10-12 hours max. Use batch for long runs.
- **Disclosure output format**: Must be `.csv` or `.txt`. No Excel, no pickle, no HDF5.
- **Path hardcoding**: Use relative paths or config variables, not absolute paths.
- **Overnight runs**: Must be batch (PBS), not interactive.
- **Forgetting disclosure stats**: Build automatic disclosure statistic generation into code
  from the start -- don't try to reconstruct after the fact.
