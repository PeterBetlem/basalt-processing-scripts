from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def parse_scan_datetime(folder_name: str) -> pd.Timestamp:
    for date_format in ("%Y%m%d-%H%M%S", "%Y%m%d_%H%M%S"):
        try:
            return pd.to_datetime(folder_name, format=date_format)
        except ValueError:
            continue
    raise ValueError(f"Could not parse a scan datetime from {folder_name!r}")


def xray_kv_from_xtekct(path: str | Path) -> float:
    path = Path(path)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*XraykV\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$", line, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    raise ValueError(f"No XraykV value found in {path}")


def scan_kv_from_raw_folder(raw_scan_folder: str | Path) -> float | None:
    raw_scan_folder = Path(raw_scan_folder)
    metadata_paths = sorted(raw_scan_folder.rglob("*.xtekct"))
    if not metadata_paths:
        return None
    primary_path = next((path for path in metadata_paths if not path.stem.endswith("_01")), metadata_paths[0])
    return xray_kv_from_xtekct(primary_path)


def build_ct_datetime_table(processed_root: str | Path, raw_root: str | Path) -> pd.DataFrame:
    processed_root = Path(processed_root)
    raw_root = Path(raw_root)
    rows = []
    for processed_folder in sorted(path for path in processed_root.iterdir() if path.is_dir()):
        try:
            scan_datetime = parse_scan_datetime(processed_folder.name)
        except ValueError:
            continue
        rows.append(
            {
                "Folder_Name": processed_folder.name,
                "folder_name": processed_folder.name,
                "datetime": scan_datetime,
                "kV": scan_kv_from_raw_folder(raw_root / processed_folder.name),
            }
        )
    return pd.DataFrame(rows, columns=["Folder_Name", "folder_name", "datetime", "kV"])
