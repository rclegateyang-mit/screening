"""Backward-compat shim: redirects ``from code.X`` to ``from screening.X``."""

import importlib
import warnings

# Re-export everything from screening so ``from code import get_data_dir`` works.
from screening import *  # noqa: F401,F403
from screening import (
    PATHS, DATA_RAW, DATA_CLEAN, DATA_BUILD,
    OUTPUT_EQUILIBRIUM, OUTPUT_WORKERS, OUTPUT_MARKDOWNS, OUTPUT_ESTIMATION,
    get_paths, get_data_dir, get_output_dir,
    get_data_subdir, get_output_subdir, configure_paths,
)

# Map old subpackage names → new ones for ``from code.estimation.X`` etc.
_SUBMODULE_MAP = {
    "data_environment": "screening.simulate",
    "estimation": "screening.analysis",
}


def __getattr__(name):
    if name in _SUBMODULE_MAP:
        warnings.warn(
            f"Import 'code.{name}' is deprecated; use '{_SUBMODULE_MAP[name]}'",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.import_module(_SUBMODULE_MAP[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
