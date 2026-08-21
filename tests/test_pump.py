from pathlib import Path

from basalt_processing.pump import load_pump_file


def test_load_pump_file_skips_modlab_header_rows(tmp_path: Path):
    path = tmp_path / "T2636.txt"
    path.write_text(
        "Time\tCell Pressure_Eng\n"
        "sensor\tB1\n"
        "formula\t-\n"
        "unit\tMPa\n"
        "31:01:2023 17:17:00\t1.2\n",
        encoding="ISO-8859-1",
    )
    df = load_pump_file(path)
    assert list(df.columns[:2]) == ["Time", "Cell Pressure_Eng"]
    assert df.loc[0, "Cell Pressure_Eng"] == 1.2
    assert str(df.loc[0, "datetime"]) == "2023-01-31 17:17:00"
