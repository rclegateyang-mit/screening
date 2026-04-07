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
    required = ["numpy", "scipy", "pandas", "sklearn", "xgboost"]
    for pkg in required:
        display = "scikit-learn" if pkg == "sklearn" else pkg
        ver = _try_import(pkg)
        status = "OK" if ver != "NOT FOUND" else "MISSING"
        out(f"  {display:20s} {ver:30s} [{status}]")

    # ---- OPTIONAL PACKAGES ----
    out(f"\n[4] OPTIONAL PACKAGES (may not be installed)")
    optional = {
        "matplotlib": "plotting",
        "statsmodels": "statistical models",
        "lightgbm": "alternative GBT (not currently used)",
    }
    for pkg, desc in optional.items():
        ver = _try_import(pkg)
        out(f"  {pkg:20s} {ver:30s} — {desc}")

    # ---- XGBOOST detail ----
    out(f"\n[5] XGBOOST DETAIL")
    try:
        import xgboost as xgb
        out(f"  Version: {xgb.__version__}")
        try:
            m = xgb.XGBRegressor(n_estimators=2, tree_method="hist")
            out(f"  XGBRegressor with tree_method=hist: AVAILABLE")
        except Exception as e:
            out(f"  XGBRegressor instantiation failed: {e}")
    except ImportError:
        out(f"  xgboost NOT IMPORTABLE — see section 3")
        out(f"  -> Try a different conda env (e.g. py3cf): conda env list")
        out(f"  -> Or request xgboost installation from your RDCA")

    # ---- PBS PRO ----
    out(f"\n[6] PBS PRO (job scheduler)")
    out(f"  qsub:       {shutil.which('qsub') or '(not found)'}")
    out(f"  qstat:      {shutil.which('qstat') or '(not found)'}")
    pbsnodes_out = _run(["pbsnodes", "-a"], timeout=15)
    out(f"  pbsnodes:   {_parse_pbsnodes(pbsnodes_out)}")

    # ---- PROJECT PATHS ----
    out(f"\n[7] PROJECT PATHS")
    out(f"  Working directory:    {os.getcwd()}")
    out(f"  PYTHONPATH:           {os.environ.get('PYTHONPATH', '(not set)')}")

    # Check if screening.rdc package is importable (the deployed package)
    try:
        from screening.rdc import helpers  # noqa: F401
        out(f"  screening.rdc:        IMPORTABLE")
    except ImportError as e:
        out(f"  screening.rdc:        NOT IMPORTABLE ({e})")
        out(f"  -> cd /projects/<project_id>/programs && export PYTHONPATH=.")
        out(f"  -> screening/rdc/ folder must be in this directory")

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
        out(f"  -> source /apps/anaconda/bin/activate py3cf")
    else:
        out(f"  numpy {numpy_ver} — OK")

    xgb_ver = _try_import("xgboost")
    if xgb_ver == "NOT FOUND":
        out(f"  CRITICAL: xgboost not found — required for education imputation.")
        out(f"  -> Try a different conda env: conda env list")
        out(f"  -> Or request from your RDCA")
    else:
        out(f"  xgboost {xgb_ver} — OK")

    sklearn_ver = _try_import("sklearn")
    if sklearn_ver == "NOT FOUND":
        out(f"  CRITICAL: scikit-learn not found — required for StratifiedKFold.")
    else:
        out(f"  scikit-learn {sklearn_ver} — OK")

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
