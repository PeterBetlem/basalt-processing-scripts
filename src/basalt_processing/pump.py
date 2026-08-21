from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from basalt_processing.paths import ensure_parent


def load_pump_file(path: str | Path, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    df = pd.read_csv(path, header=0, skiprows=[1, 2, 3], delimiter="\t", encoding=encoding)
    time_col = "Time" if "Time" in df.columns else df.columns[0]
    df["datetime"] = pd.to_datetime(df[time_col], format="%d:%m:%Y %H:%M:%S", errors="coerce")
    if "Time2" in df.columns:
        df["datetime2"] = pd.to_datetime(df["Time2"], format="%d:%m:%Y %H:%M:%S", errors="coerce")
    return df


def write_pump_outputs(df: pd.DataFrame, output_stem: str | Path) -> None:
    stem = Path(output_stem)
    ensure_parent(stem.with_suffix(".csv"))
    df.to_csv(stem.with_suffix(".csv"), index=False)
    df.to_excel(stem.with_suffix(".xlsx"), index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean MODLab/Autolab pump export data.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output stem without extension.")
    parser.add_argument("--encoding", default="ISO-8859-1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frames = [load_pump_file(path, encoding=args.encoding) for path in args.inputs]
    df = pd.concat(frames, ignore_index=True).sort_values("datetime")
    write_pump_outputs(df, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
