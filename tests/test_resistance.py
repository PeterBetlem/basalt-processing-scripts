from pathlib import Path

import pandas as pd

from basalt_processing.resistance import (
    clean_resistance_folder,
    load_resistance_file,
    write_resistivity_pickles,
)


def test_load_resistance_file_builds_datetime_from_date_time(tmp_path: Path):
    path = tmp_path / "t2636_5000_avg.csv"
    path.write_text("Date;Time;R_Ch1\n31.01.2023;17:17:00;12,5\n", encoding="utf-8")
    df = load_resistance_file(path)
    assert df.loc[0, "R_Ch1"] == 12.5
    assert str(df.loc[0, "datetime"]) == "2023-01-31 17:17:00"
    assert df.loc[0, "filename"] == "t2636_5000_avg.csv"


def test_load_resistance_file_reads_current_combined_export_format(tmp_path: Path):
    path = tmp_path / "1000,0Hz_avg_combined.csv"
    path.write_text(
        "Timestamp,R_Ch0,R_Ch1,U_Ch0\n"
        "2023-02-03 14:03:00,997.0,610854.5,0.01629664\n",
        encoding="utf-8",
    )

    df = load_resistance_file(path)

    assert df.loc[0, "R_Ch1"] == 610854.5
    assert str(df.loc[0, "datetime"]) == "2023-02-03 14:03:00"


def test_clean_resistance_folder_adds_resistivity_and_elapsed_time(tmp_path: Path):
    for frequency in (1000, 5000):
        path = tmp_path / f"{frequency},0Hz_avg_combined.csv"
        path.write_text(
            "Timestamp,R_Ch1\n"
            "2023-02-03 13:00:00,100.0\n"
            "2023-02-03 14:00:00,200.0\n",
            encoding="utf-8",
        )

    cleaned = clean_resistance_folder(
        tmp_path,
        reference_datetime="2023-01-31 14:00:00",
        min_time_hours=72,
    )

    assert set(cleaned) == {1000.0, 5000.0}
    frame = cleaned[1000.0]
    assert frame["time"].tolist() == [72.0]
    assert frame["rho_Ch1"].iloc[0] > 0

    outputs = write_resistivity_pickles(cleaned, tmp_path / "output")
    assert set(outputs) == {"1000hz", "5000hz", "combined"}
    assert pd.read_pickle(outputs["combined"])["frequency_hz"].nunique() == 2
