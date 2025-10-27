"""Core estimation package for worker screening simulations."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = {
    "helpers",
    "jax_model",
    "run_mle_penalty_phi_sigma_jax",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_SUBMODULES))


__all__ = sorted(list(_SUBMODULES))
