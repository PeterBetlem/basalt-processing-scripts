from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from basalt_processing.paths import ensure_parent


def calculate_permeability(
    flow_rate_m3_s: float,
    viscosity_pa_s: float,
    diameter_m: float,
    length_m: float,
    pressure_upstream_pa: float,
    pressure_downstream_pa: float,
) -> float:
    """Calculate cylindrical-sample permeability from Darcy's law in m²."""
    pressure_drop_pa = pressure_upstream_pa - pressure_downstream_pa
    cross_section_m2 = 3.141592653589793 * (diameter_m / 2) ** 2
    return (flow_rate_m3_s * viscosity_pa_s * length_m) / (cross_section_m2 * pressure_drop_pa)


def add_flow_rate_columns(
    df: pd.DataFrame,
    volume_columns: list[str],
    window: int = 10,
) -> pd.DataFrame:
    out = df.copy()
    if "datetime" not in out.columns:
        raise ValueError("Input dataframe must contain a datetime column")
    elapsed_s = out["datetime"].diff().dt.total_seconds()
    for column in volume_columns:
        if column not in out.columns:
            raise KeyError(f"Missing volume column: {column}")
        rate_name = f"{column}_rate"
        smooth_name = f"{rate_name}_smooth"
        out[rate_name] = out[column].diff() / elapsed_s
        out[smooth_name] = out[rate_name].rolling(window=window, min_periods=1, center=True).mean()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate pump-derived flow-rate columns for permeability review.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--datetime-column", default="datetime")
    parser.add_argument("--volume-columns", nargs="+", required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = pd.read_excel(args.input) if args.input.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(args.input)
    df["datetime"] = pd.to_datetime(df[args.datetime_column], errors="coerce")
    out = add_flow_rate_columns(df, args.volume_columns, window=args.window)
    ensure_parent(args.output)
    out.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
