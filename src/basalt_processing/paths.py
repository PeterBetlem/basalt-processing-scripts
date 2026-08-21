from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a TOML config file, returning an empty dict when no path is supplied."""
    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    data["_config_dir"] = config_path.parent
    return data


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a path, optionally relative to a config directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if base_dir is None:
        return path.resolve()
    return (base_dir / path).resolve()


def ensure_parent(path: str | Path) -> Path:
    """Create the parent folder for an output path and return the path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path
