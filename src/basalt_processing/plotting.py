from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from basalt_processing.paths import ensure_parent


def _read_table(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def plot_timeline(
    pump: pd.DataFrame | None,
    resistance: pd.DataFrame | None,
    output: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    if pump is not None and "datetime" in pump.columns:
        y_col = next((c for c in pump.columns if "Pressure" in c and c.endswith("_Eng")), None)
        if y_col is not None:
            ax.plot(pd.to_datetime(pump["datetime"]), pump[y_col], label=y_col)
    if resistance is not None and "datetime" in resistance.columns:
        y_col = next((c for c in resistance.columns if c.startswith("R_")), None)
        if y_col is not None:
            ax2 = ax.twinx()
            ax2.plot(pd.to_datetime(resistance["datetime"]), resistance[y_col], color="tab:red", label=y_col)
            ax2.set_ylabel(y_col)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Pump record")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    out = ensure_parent(output)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot basalt experiment timeline data.")
    parser.add_argument("--pump", type=Path)
    parser.add_argument("--resistance", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_timeline(_read_table(args.pump), _read_table(args.resistance), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
