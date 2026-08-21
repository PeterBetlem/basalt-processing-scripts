from __future__ import annotations

import argparse
import re
from math import pi
from pathlib import Path

import pandas as pd

from basalt_processing.paths import ensure_parent


def load_resistance_file(
    path: str | Path,
    delimiter: str | None = None,
    decimal: str = ",",
) -> pd.DataFrame:
    path = Path(path)
    read_options: dict[str, str] = {"decimal": decimal}
    if delimiter is None:
        read_options.update({"sep": None, "engine": "python"})
    else:
        read_options["sep"] = delimiter
    df = pd.read_csv(path, **read_options)
    df["folder"] = path.parent.as_posix()
    df["filename"] = path.name
    if {"Date", "Time"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
            dayfirst=True,
        )
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "Timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    else:
        raise ValueError(f"No Date/Time, datetime, or Timestamp column found in {path}")
    return df


def load_resistance_folder(folder: str | Path, pattern: str = "*Hz_avg_combined.csv") -> pd.DataFrame:
    folder = Path(folder)
    frames = [load_resistance_file(path) for path in sorted(folder.rglob(pattern))]
    if not frames:
        raise FileNotFoundError(f"No resistance files matching {pattern!r} under {folder}")
    return pd.concat(frames, ignore_index=True).sort_values("datetime")


def frequency_hz_from_filename(path: str | Path) -> float:
    match = re.search(r"(\d+(?:[,.]\d+)?)\s*Hz", Path(path).name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not determine measurement frequency from {path}")
    return float(match.group(1).replace(",", "."))


def calculate_resistivity(
    frame: pd.DataFrame,
    *,
    reference_datetime: str | pd.Timestamp | None = None,
    min_time_hours: float | None = 72,
    length_m: float = 0.04,
    diameter_m: float = 0.0252,
) -> pd.DataFrame:
    if "R_Ch1" not in frame:
        raise ValueError("Resistance data must contain an R_Ch1 column.")

    cleaned = frame.copy()
    area_m2 = pi * (diameter_m / 2) ** 2
    cleaned["rho_Ch1"] = cleaned["R_Ch1"] * area_m2 / length_m

    if reference_datetime is None:
        return cleaned
    if "datetime" not in cleaned:
        raise ValueError("Resistance data must contain a datetime column.")

    reference = pd.Timestamp(reference_datetime)
    cleaned["time"] = (cleaned["datetime"] - reference) / pd.Timedelta(hours=1)
    if min_time_hours is not None:
        cleaned = cleaned.loc[cleaned["time"] >= min_time_hours].copy()
    return cleaned


def clean_resistance_folder(
    folder: str | Path,
    *,
    reference_datetime: str | pd.Timestamp,
    min_time_hours: float | None = 72,
    pattern: str = "*Hz_avg_combined.csv",
) -> dict[float, pd.DataFrame]:
    folder = Path(folder)
    frames_by_frequency: dict[float, list[pd.DataFrame]] = {}
    for path in sorted(folder.rglob(pattern)):
        frequency_hz = frequency_hz_from_filename(path)
        cleaned = calculate_resistivity(
            load_resistance_file(path),
            reference_datetime=reference_datetime,
            min_time_hours=min_time_hours,
        )
        cleaned["frequency_hz"] = frequency_hz
        frames_by_frequency.setdefault(frequency_hz, []).append(cleaned)

    if not frames_by_frequency:
        raise FileNotFoundError(f"No resistance files matching {pattern!r} under {folder}")
    return {
        frequency_hz: pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
        for frequency_hz, frames in frames_by_frequency.items()
    }


def write_resistivity_pickles(
    frames_by_frequency: dict[float, pd.DataFrame], output_root: str | Path
) -> dict[str, Path]:
    output_root = Path(output_root)
    outputs: dict[str, Path] = {}
    ordered_frames = []
    for frequency_hz in sorted(frames_by_frequency):
        label = f"{frequency_hz:g}hz"
        path = output_root / f"resistivity_{label}.pkl"
        ensure_parent(path)
        frame = frames_by_frequency[frequency_hz]
        frame.to_pickle(path)
        outputs[label] = path
        ordered_frames.append(frame)

    combined_path = output_root / "resistivity.pkl"
    pd.concat(ordered_frames, ignore_index=True).sort_values("datetime").reset_index(drop=True).to_pickle(combined_path)
    outputs["combined"] = combined_path
    return outputs


def write_resistance_outputs(df: pd.DataFrame, output_stem: str | Path) -> None:
    stem = Path(output_stem)
    ensure_parent(stem.with_suffix(".csv"))
    df.to_csv(stem.with_suffix(".csv"), index=False)
    df.to_excel(stem.with_suffix(".xlsx"), index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean and merge resistance exports.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pattern", default="*Hz_avg_combined.csv")
    parser.add_argument("--output", required=True, type=Path, help="Output stem without extension.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = load_resistance_folder(args.folder, pattern=args.pattern)
    write_resistance_outputs(df, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
