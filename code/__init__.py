"""Worker screening simulation package with configurable storage paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
_CONFIG_FILE = PROJECT_ROOT / "path_config.json"
_ENV_DATA = "SCREENING_DATA_DIR"
_ENV_OUTPUT = "SCREENING_OUTPUT_DIR"

_DEFAULTS: Dict[str, str] = {
    "data_dir": str(PROJECT_ROOT / "data"),
    "output_dir": str(PROJECT_ROOT / "output"),
}


def _normalize(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _load_paths() -> Dict[str, str]:
    paths = dict(_DEFAULTS)

    if _CONFIG_FILE.exists():
        try:
            loaded = json.loads(_CONFIG_FILE.read_text())
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            for key in ("data_dir", "output_dir"):
                value = loaded.get(key)
                if value:
                    paths[key] = str(value)

    env_data = os.getenv(_ENV_DATA)
    if env_data:
        paths["data_dir"] = env_data

    env_output = os.getenv(_ENV_OUTPUT)
    if env_output:
        paths["output_dir"] = env_output

    return {key: _normalize(value) for key, value in paths.items()}


PATHS: Dict[str, str] = _load_paths()


def get_paths(*, create: bool = False) -> Dict[str, Path]:
    """Return a copy of the configured paths."""
    resolved = {key: Path(value) for key, value in PATHS.items()}
    if create:
        for directory in resolved.values():
            directory.mkdir(parents=True, exist_ok=True)
    return resolved


def get_data_dir(*, create: bool = False) -> Path:
    """Return the configured data directory."""
    path = Path(PATHS["data_dir"])
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(*, create: bool = False) -> Path:
    """Return the configured output directory."""
    path = Path(PATHS["output_dir"])
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def configure_paths(
    *,
    data_dir: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    persist: bool = True,
    create: bool = True,
) -> Dict[str, Path]:
    """Configure (and optionally persist) custom data/output directories."""
    updated = dict(PATHS)

    if data_dir is not None:
        updated["data_dir"] = _normalize(str(data_dir))
    if output_dir is not None:
        updated["output_dir"] = _normalize(str(output_dir))

    if create:
        Path(updated["data_dir"]).mkdir(parents=True, exist_ok=True)
        Path(updated["output_dir"]).mkdir(parents=True, exist_ok=True)

    if persist:
        _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(json.dumps(updated, indent=2))

    PATHS.clear()
    PATHS.update(updated)
    return get_paths(create=False)


import importlib


def __getattr__(name: str):
    if name in {"data_environment", "estimation"}:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        list(globals().keys())
        + ["data_environment", "estimation"]
        + ["configure_paths", "get_data_dir", "get_output_dir", "get_paths", "PATHS"]
    )


__all__ = ["configure_paths", "get_data_dir", "get_output_dir", "get_paths", "PATHS"]
