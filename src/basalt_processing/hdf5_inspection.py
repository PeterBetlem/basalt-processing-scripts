from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import h5py


def is_pickled_amira_header(attribute_name: str) -> bool:
    """Return whether an HDF5 attribute name identifies a pickled Amira header."""
    normalized_name = str(attribute_name).lower()
    return all(token in normalized_name for token in ("amira", "header", "pickled"))


def describe_hdf5(path: str | Path) -> None:
    """Print a concise, recursive, read-only summary of an HDF5 file."""
    hdf5_path = Path(path)

    def describe_object(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Group):
            print(f"{indent}Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}Dataset: {name}")
            print(f"{indent}  shape: {obj.shape}")
            print(f"{indent}  dtype: {obj.dtype}")
            for attribute_name, value in obj.attrs.items():
                if is_pickled_amira_header(attribute_name):
                    print(f"{indent}  attribute: {attribute_name} (pickled Amira header omitted)")
                else:
                    print(f"{indent}  attribute: {attribute_name} = {value}")

    with h5py.File(hdf5_path, "r") as h5:
        print(f"HDF5 file: {hdf5_path}")
        print(f"Top-level entries: {', '.join(h5.keys())}")
        h5.visititems(describe_object)


def raw_hdf5_attribute(path: str | Path, object_path: str, attribute_name: str) -> Any:
    """Return a raw HDF5 attribute without deserializing it."""
    with h5py.File(Path(path), "r") as h5:
        return h5[object_path].attrs[attribute_name]


def find_pickled_amira_headers(path: str | Path) -> list[tuple[str, str]]:
    """Return dataset paths and attribute names for pickled Amira headers."""
    headers: list[tuple[str, str]] = []

    def find_headers(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            for attribute_name in obj.attrs:
                if is_pickled_amira_header(attribute_name):
                    headers.append((name, str(attribute_name)))

    with h5py.File(Path(path), "r") as h5:
        h5.visititems(find_headers)
    return headers


def unpickle_amira_header(path: str | Path, object_path: str, attribute_name: str) -> Any:
    """Deserialize a pickled Amira header from a trusted HDF5 file."""
    raw_value = raw_hdf5_attribute(path, object_path, attribute_name)
    return pickle.loads(raw_value)


def render_amira_header(header: Any) -> str:
    """Decode a pickled Avizo/Amira header byte array as text."""
    payload = header[0] if isinstance(header, (list, tuple)) and len(header) == 1 else header
    if hasattr(payload, "tobytes"):
        return payload.tobytes().decode("utf-8", errors="replace")
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return str(payload)
