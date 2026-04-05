#!/usr/bin/env python3
"""Census RDC environment check. Run on first login to see what's available.

Usage:
    python rdc/env_check.py              # print to stdout
    python rdc/env_check.py --save report.txt   # also save to file

No dependencies beyond stdlib (so it runs even if numpy is broken).
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run(cmd: list[str], timeout: int = 10) -> str:
    """Run a command and return stdout, or error message."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip() if result.returncode == 0 else f"(exit {result.returncode})"
    except FileNotFoundError:
        return "(not found)"
    except subprocess.TimeoutExpired:
        return "(timed out)"
    except Exception as e:
        return f"(error: {e})"


def _try_import(name: str) -> str:
    """Try to import a package. Return version string or error."""
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "installed (no version attr)")
    except ImportError:
        return "NOT FOUND"
    except Exception as e:
        return f"ERROR: {e}"


def _read_meminfo() -> str:
    """Read total RAM from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024 / 1024:.1f} GB"
    except (FileNotFoundError, ValueError):
        pass
    return "(unknown)"


def _read_cpu_count() -> str:
    """Count CPUs from /proc/cpuinfo or os.cpu_count()."""
    count = os.cpu_count()
    return str(count) if count else "(unknown)"


def _parse_pbsnodes(output: str) -> str:
    """Parse pbsnodes -a output to count nodes and resources."""
    if output.startswith("("):
        return output  # error message
    lines = output.strip().split("\n")
    node_count = sum(1 for line in lines if line and not line.startswith(" "))
    return f"{node_count} nodes found"


# ---------------------------------------------------------------------------


def run_checks(save_path: str | None = None) -> None:
    lines: list[str] = []

    def out(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    out(f"Census RDC Environment Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"{'=' * 70}")

    # ---- SYSTEM ----
    out(f"\n[1] SYSTEM")
    out(f"  Hostname:   {platform.node()}")
    out(f"  OS:         {platform.platform()}")
    out(f"  RAM:        {_read_meminfo()}")
    out(f"  CPUs:       {_read_cpu_count()}")

    # Try to read /etc/redhat-release
    try:
        with open("/etc/redhat-release") as f:
            out(f"  RedHat:     {f.read().strip()}")
    except FileNotFoundError:
        out(f"  RedHat:     (not a RedHat system)")

    # ---- PYTHON ----
    out(f"\n[2] PYTHON")
    out(f"  Version:    {sys.version}")
    out(f"  Executable: {sys.executable}")
    out(f"  conda:      {_run(['conda', '--version'])}")
    out(f"  pip:        {_run([sys.executable, '-m', 'pip', '--version'])}")

    # ---- REQUIRED PACKAGES ----
    out(f"\n[3] REQUIRED PACKAGES")
    required = ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "statsmodels"]
    for pkg in required:
        display = "scikit-learn" if pkg == "sklearn" else pkg
        ver = _try_import(pkg)
        status = "OK" if ver != "NOT FOUND" else "MISSING"
        out(f"  {display:20s} {ver:30s} [{status}]")

    # ---- OPTIONAL PACKAGES ----
    out(f"\n[4] OPTIONAL PACKAGES (may not be installed)")
    optional = {
        "jax": "JAX (autodiff) — use scipy fallback if missing",
        "jaxlib": "JAX backend",
        "mpi4py": "MPI — use multiprocessing if missing",
        "jaxopt": "JAX optimizers",
        "xgboost": "XGBoost (alternative to sklearn GBT)",
        "lightgbm": "LightGBM (alternative to sklearn GBT)",
    }
    for pkg, desc in optional.items():
        ver = _try_import(pkg)
        out(f"  {pkg:20s} {ver:30s} — {desc}")

    # ---- sklearn detail ----
    out(f"\n[5] SCIKIT-LEARN DETAIL")
    try:
        import sklearn
        ver = sklearn.__version__
        major, minor = int(ver.split(".")[0]), int(ver.split(".")[1])
        out(f"  Version: {ver}")
        if major >= 1 or (major == 0 and minor >= 24):
            out(f"  HistGradientBoosting: AVAILABLE (fast histogram-based GBT for large data)")
        else:
            out(f"  HistGradientBoosting: NOT AVAILABLE (sklearn < 0.24)")
            out(f"  -> GradientBoostingRegressor works but is MUCH SLOWER on 150M rows")
            out(f"  -> Consider requesting sklearn >= 1.0 from your RDCA")
    except Exception:
        out(f"  (sklearn not importable — see section 3)")

    # ---- PBS PRO ----
    out(f"\n[6] PBS PRO (job scheduler)")
    out(f"  qsub:       {shutil.which('qsub') or '(not found)'}")
    out(f"  qstat:      {shutil.which('qstat') or '(not found)'}")
    pbsnodes_out = _run(["pbsnodes", "-a"], timeout=15)
    out(f"  pbsnodes:   {_parse_pbsnodes(pbsnodes_out)}")

    # ---- PROJECT PATHS ----
    out(f"\n[7] PROJECT PATHS")
    data_dir = os.environ.get("SCREENING_DATA_DIR", "(not set)")
    output_dir = os.environ.get("SCREENING_OUTPUT_DIR", "(not set)")
    out(f"  SCREENING_DATA_DIR:   {data_dir}")
    out(f"  SCREENING_OUTPUT_DIR: {output_dir}")
    out(f"  Working directory:    {os.getcwd()}")

    # Check if screening package is importable
    try:
        import screening
        out(f"  screening package:    {screening.PACKAGE_ROOT}")
        out(f"  data_dir resolved:    {screening.get_data_dir()}")
        out(f"  output_dir resolved:  {screening.get_output_dir()}")
    except ImportError:
        out(f"  screening package:    NOT IMPORTABLE")
        out(f"  -> Make sure PYTHONPATH includes the code/ directory")
        out(f"  -> Example: cd /projects/<project_id>/programs && export PYTHONPATH=.")

    # Check for /projects/ directory (RDC-specific)
    projects = Path("/projects")
    if projects.exists():
        subdirs = [d.name for d in projects.iterdir() if d.is_dir()]
        out(f"  /projects/ dirs:      {', '.join(subdirs[:5])}")
    else:
        out(f"  /projects/:           (not found — expected on RDC servers)")

    # ---- RECOMMENDATIONS ----
    out(f"\n[8] RECOMMENDATIONS")
    out(f"{'=' * 70}")

    numpy_ver = _try_import("numpy")
    if numpy_ver == "NOT FOUND":
        out(f"  CRITICAL: numpy not found. Anaconda may not be activated.")
        out(f"  -> Try: source activate base")
    else:
        out(f"  numpy {numpy_ver} — OK")

    jax_ver = _try_import("jax")
    if jax_ver == "NOT FOUND":
        out(f"  JAX not found — estimation must use scipy fallback (expected)")
    else:
        out(f"  JAX {jax_ver} — available, can use JIT optimization")

    mpi_ver = _try_import("mpi4py")
    if mpi_ver == "NOT FOUND":
        out(f"  MPI not found — use multiprocessing within a single PBS node (expected)")
    else:
        out(f"  mpi4py {mpi_ver} — available, can use distributed estimation")

    sklearn_ver = _try_import("sklearn")
    if sklearn_ver != "NOT FOUND":
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
            out(f"  HistGradientBoostingRegressor — AVAILABLE (use this for 150M obs)")
        except ImportError:
            out(f"  HistGradientBoostingRegressor — NOT AVAILABLE")
            out(f"  -> For 150M obs, GradientBoostingRegressor will be very slow")
            out(f"  -> Consider subsampling or requesting sklearn >= 1.0")

    out(f"\n{'=' * 70}")
    out(f"Save this output for reference: python rdc/env_check.py --save report.txt")

    if save_path:
        with open(save_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nReport saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check Census RDC environment: packages, PBS, paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save", type=str, default=None,
                        help="Save report to this file path")
    args = parser.parse_args()
    run_checks(save_path=args.save)


if __name__ == "__main__":
    main()
