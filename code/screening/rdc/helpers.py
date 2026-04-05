"""RDC job harness utilities: logging, memory, disclosure, data conversion.

All functions are stateless. No logging module, no classes (except one dataclass).
Designed to be readable top-to-bottom with no internet access for reference.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Logging — replaces Python logging module with plain print + file append
# ---------------------------------------------------------------------------


def log(msg: str, logfile: str | None = None) -> None:
    """Print timestamped message. Optionally append to logfile."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")


def log_section(title: str, logfile: str | None = None) -> None:
    """Print a section header."""
    log(f"==== {title} ====", logfile)


def log_error(msg: str, logfile: str | None = None) -> None:
    """Print error with [ERROR] prefix."""
    log(f"[ERROR] {msg}", logfile)


# ---------------------------------------------------------------------------
# 2. Timing — context manager + peak RSS from /proc
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    label: str
    elapsed_sec: float
    peak_rss_mb: float


@contextmanager
def time_block(label: str, logfile: str | None = None):
    """Context manager that prints elapsed time and peak RSS on exit.

    Usage:
        with time_block("Loading data"):
            df = pd.read_csv(...)
    """
    rss_before = get_peak_rss_mb()
    t0 = time.perf_counter()
    result = TimingResult(label=label, elapsed_sec=0.0, peak_rss_mb=0.0)
    try:
        yield result
    finally:
        result.elapsed_sec = time.perf_counter() - t0
        result.peak_rss_mb = get_peak_rss_mb()
        rss_delta = result.peak_rss_mb - rss_before
        log(f"[{label}] {result.elapsed_sec:.1f}s, peak RSS {result.peak_rss_mb:.0f} MB "
            f"(+{rss_delta:.0f} MB)", logfile)


def get_peak_rss_mb() -> float:
    """Read current process peak RSS (VmHWM) from /proc/self/status.

    Returns 0.0 if /proc is unavailable (e.g. macOS).
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# 3. Memory estimation — help decide PBS --mem value
# ---------------------------------------------------------------------------


def estimate_pbs_memory(n_rows: int, n_cols: int, workload: str = "ml") -> str:
    """Estimate PBS memory request for a workload. Prints calculation.

    workload='ml':         sklearn GBT needs ~3x raw for sorting/splits + 4 GB overhead.
    workload='estimation': scipy optimize needs ~2x raw for gradients + 4 GB overhead.

    Returns string like '48gb' suitable for PBS -l mem=.
    """
    raw_gb = n_rows * n_cols * 8 / 1e9  # float64

    if workload == "ml":
        multiplier, label = 3.0, "sklearn GBT (sorting + splits)"
    elif workload == "estimation":
        multiplier, label = 2.0, "scipy optimize (gradients + workspace)"
    else:
        multiplier, label = 2.0, "general"

    working_gb = raw_gb * multiplier
    overhead_gb = 4.0
    total_gb = working_gb + overhead_gb
    # Round up to next 8 GB
    requested_gb = int(math.ceil(total_gb / 8.0)) * 8

    print(f"Memory estimate for {workload} workload:")
    print(f"  Raw data:    {n_rows:>15,} rows x {n_cols} cols x 8 bytes = {raw_gb:.1f} GB")
    print(f"  Working set: {raw_gb:.1f} GB x {multiplier:.0f} ({label}) = {working_gb:.1f} GB")
    print(f"  Overhead:    {overhead_gb:.0f} GB (Python, pandas, internals)")
    print(f"  Total:       {total_gb:.1f} GB -> request {requested_gb} GB")

    return f"{requested_gb}gb"


# ---------------------------------------------------------------------------
# 4. Disclosure output — Census 4-significant-digit rounding + cell checks
# ---------------------------------------------------------------------------


def round_to_sig_digits(x: float, n: int = 4) -> float:
    """Round a number to n significant digits. Census requires 4.

    >>> round_to_sig_digits(3.14159)
    3.142
    >>> round_to_sig_digits(0.001234)
    0.001234
    >>> round_to_sig_digits(98765.0)
    98760.0
    """
    if x == 0 or not math.isfinite(x):
        return x
    magnitude = int(math.floor(math.log10(abs(x))))
    return round(x, -magnitude + (n - 1))


def prepare_disclosure_csv(
    df: pd.DataFrame,
    output_path: str | Path,
    counts_path: str | Path | None = None,
    count_col: str | None = None,
    min_cell: int = 15,
) -> pd.DataFrame:
    """Round all numeric columns to 4 sig digits and save to CSV.

    If count_col is given, flag rows where count < min_cell.
    If counts_path is given, save unweighted counts alongside (for reviewer).
    Returns the rounded DataFrame.
    """
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        rounded[col] = rounded[col].apply(lambda v: round_to_sig_digits(v, 4))

    # Flag small cells
    if count_col and count_col in rounded.columns:
        small = rounded[rounded[count_col] < min_cell]
        if len(small) > 0:
            print(f"WARNING: {len(small)} rows have {count_col} < {min_cell} — must suppress")
            for idx in small.index:
                print(f"  Row {idx}: {count_col} = {df.loc[idx, count_col]}")

    rounded.to_csv(output_path, index=False)
    print(f"Saved disclosure-ready output: {output_path}")

    if counts_path is not None:
        # Save unrounded for reviewer verification
        df.to_csv(counts_path, index=False)
        print(f"Saved unrounded support file:  {counts_path}")

    return rounded


def pseudo_quantile(data: np.ndarray, q: float, window: int = 5) -> float:
    """Census-compliant pseudo-quantile.

    Instead of the true quantile (which reveals a real data value), compute
    the mean of the target observation and `window` observations on each side.
    Minimum 2*window + 1 = 11 observations required.

    Args:
        data: 1-D array of values.
        q: quantile in [0, 1] (e.g. 0.5 for median).
        window: observations on each side of the target (default 5 -> 11 total).

    Returns:
        Pseudo-quantile value (mean of 11 observations around the target).
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    min_obs = 2 * window + 1

    if n < min_obs:
        raise ValueError(
            f"Need at least {min_obs} observations for pseudo-quantile, got {n}. "
            f"Try a larger sample or reduce window."
        )

    idx = int(q * (n - 1))
    lo = max(0, idx - window)
    hi = min(n, idx + window + 1)

    # Ensure we always have enough observations
    if hi - lo < min_obs:
        if lo == 0:
            hi = min(n, min_obs)
        else:
            lo = max(0, hi - min_obs)

    return float(np.mean(sorted_data[lo:hi]))


# ---------------------------------------------------------------------------
# 5. Data conversion — SAS to CSV for large files
# ---------------------------------------------------------------------------


def read_sas_chunked(sas_path: str | Path, chunksize: int = 500_000):
    """Read a .sas7bdat file in chunks. Yields DataFrames with progress.

    Usage:
        for chunk in read_sas_chunked('data.sas7bdat'):
            process(chunk)
    """
    total_rows = 0
    reader = pd.read_sas(str(sas_path), chunksize=chunksize)
    for chunk in reader:
        total_rows += len(chunk)
        print(f"  Read {total_rows:,} rows...", end="\r", flush=True)
        yield chunk
    print(f"  Read {total_rows:,} rows total.")


def sas_to_csv(
    sas_path: str | Path,
    csv_path: str | Path,
    chunksize: int = 500_000,
) -> int:
    """Convert SAS file to CSV, handling large files via chunked read/append.

    Returns total row count.
    """
    sas_path, csv_path = Path(sas_path), Path(csv_path)
    print(f"Converting {sas_path.name} -> {csv_path.name}")

    t0 = time.perf_counter()
    total_rows = 0
    first_chunk = True

    for chunk in read_sas_chunked(sas_path, chunksize):
        chunk.to_csv(csv_path, index=False, mode="a" if not first_chunk else "w",
                     header=first_chunk)
        total_rows += len(chunk)
        first_chunk = False

    elapsed = time.perf_counter() - t0
    size_mb = csv_path.stat().st_size / 1e6
    print(f"Done: {total_rows:,} rows, {size_mb:.0f} MB, {elapsed:.1f}s")
    return total_rows


# ---------------------------------------------------------------------------
# 6. Environment setup — prevent numpy/scipy thread explosion on shared nodes
# ---------------------------------------------------------------------------


def set_single_threaded() -> None:
    """Set env vars to prevent numpy/scipy from spawning threads.

    MUST be called before importing numpy. On PBS compute nodes, each job
    gets a fixed CPU allocation — extra threads just cause contention.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = os.environ.get(var, "1")
    # XLA (JAX) thread limiting — harmless if JAX not installed
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
