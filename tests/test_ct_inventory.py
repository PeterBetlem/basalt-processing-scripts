from pathlib import Path

from basalt_processing.ct_inventory import build_ct_datetime_table


def test_build_ct_datetime_table_parses_processed_folder_dates_and_voltage(tmp_path: Path):
    processed_root = tmp_path / "processed"
    raw_root = tmp_path / "raw"
    scan_folder = processed_root / "20230209-111213"
    scan_folder.mkdir(parents=True)
    metadata_path = raw_root / "20230209-111213" / "data" / "sample.xtekct"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text("[Xrays]\nXraykV=140\n", encoding="utf-8")

    table = build_ct_datetime_table(processed_root, raw_root)

    assert table.loc[0, "Folder_Name"] == "20230209-111213"
    assert str(table.loc[0, "datetime"]) == "2023-02-09 11:12:13"
    assert table.loc[0, "kV"] == 140
