from __future__ import annotations

import json
from pathlib import Path
import tomllib


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "reference"
EXPECTED_NOTEBOOKS = {
    "01_ct_conversion_and_scan_inventory.ipynb",
    "02_ct_cylinder_crop_examples.ipynb",
    "03_ct_porosity_analysis_and_visualisation.ipynb",
    "04_ct_global_registration.ipynb",
    "05_ct_combined_volume_and_difference_visualisation.ipynb",
    "06_pump_resistance_permeability_processing.ipynb",
    "07_timeline_and_key_tables.ipynb",
}


def notebook_text(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def notebook_all_text(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))


def notebook_has_rendered_output(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if output.get("output_type") in {"display_data", "execute_result"} and (
                "image/png" in data or "image/jpeg" in data or "text/html" in data
            ):
                return True
    return False


def test_reference_folder_contains_only_expected_notebooks():
    names = {path.name for path in NOTEBOOK_DIR.iterdir()}
    assert names == EXPECTED_NOTEBOOKS


def test_readme_documents_renumbered_notebook_workflow():
    readme = (NOTEBOOK_DIR.parents[1] / "README.md").read_text(encoding="utf-8")

    for name in sorted(EXPECTED_NOTEBOOKS):
        assert f"`{name}`" in readme
    assert "CT Combined volume and difference visualisation" in readme
    assert "Notebook 6 produces the timeline inputs consumed by notebook 7" in readme


def test_notebooks_and_readme_do_not_reference_stale_numbered_filenames():
    text = (NOTEBOOK_DIR.parents[1] / "README.md").read_text(encoding="utf-8")
    text += "\n".join(
        notebook_all_text(path) for path in NOTEBOOK_DIR.glob("*.ipynb")
    )

    for stale_name in (
        "02_ct_global_registration.ipynb",
        "03_ct_visual_checks.ipynb",
        "04_pump_resistance_permeability_processing.ipynb",
        "05_timeline_and_key_tables.ipynb",
        "06_ct_cylinder_crop_examples.ipynb",
        "07_ct_porosity_analysis_and_visualisation.ipynb",
    ):
        assert stale_name not in text


def test_notebooks_parse_as_json():
    for path in NOTEBOOK_DIR.glob("*.ipynb"):
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] >= 4
        assert isinstance(nb.get("cells"), list)


def test_no_active_p_drive_paths_in_code_cells():
    offenders = []
    for path in NOTEBOOK_DIR.glob("*.ipynb"):
        text = notebook_text(path)
        if "P:\\" in text or "P:/" in text:
            offenders.append(path.name)
    assert offenders == []


def test_no_ome_zarr_or_dual_energy_in_code_cells():
    forbidden = ("registration_ome_zarr", "ome_zarr", "ome-zarr", "zarr", "DualEnergy", "paths_de")
    offenders = {}
    for path in NOTEBOOK_DIR.glob("*.ipynb"):
        text = notebook_text(path)
        hits = [term for term in forbidden if term in text]
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


def test_visual_and_timeline_notebooks_keep_rendered_outputs():
    required = [
        "05_ct_combined_volume_and_difference_visualisation.ipynb",
        "06_pump_resistance_permeability_processing.ipynb",
        "07_timeline_and_key_tables.ipynb",
    ]
    missing = [name for name in required if not notebook_has_rendered_output(NOTEBOOK_DIR / name)]
    assert missing == []


def test_registration_notebook_only_transforms_existing_registered_outputs():
    path = NOTEBOOK_DIR / "04_ct_global_registration.ipynb"
    text = notebook_text(path)

    assert "existing_transform_pairs" in text
    assert "candidate.is_file()" in text
    assert "RUN_DATASET_TRANSFORM = False" in text
    assert "aligned_out_h5=aligned_out_h5" in text


def test_registration_notebook_validates_shared_global_coordinates_without_zscore():
    registration_text = notebook_text(NOTEBOOK_DIR / "04_ct_global_registration.ipynb")
    visual_checks_text = notebook_all_text(
        NOTEBOOK_DIR / "05_ct_combined_volume_and_difference_visualisation.ipynb"
    )

    assert "global_origin_xyz_mm" in registration_text
    assert "global_spacing_xyz_mm" in registration_text
    assert "global_size_xyz" in registration_text
    assert "np.allclose" in registration_text
    assert "slice_positions_mm" in registration_text
    assert "z-score difference" not in registration_text
    assert "## Normalized difference checks" in visual_checks_text
    assert "load_zscore_volume" in visual_checks_text


def test_registration_notebook_uses_pregenerated_140kv_global_files():
    text = notebook_text(NOTEBOOK_DIR / "04_ct_global_registration.ipynb")

    assert 'paths_config["ct_global_validation"]' in text
    assert "path.relative_to(ct_processed_root)" in text
    assert '"*._global*.h5"' in text
    assert '"*._global*.hdf5"' in text
    assert 'np.zeros(3, dtype=float)' in text
    assert "grid_groups" in text
    assert "largest_compatible_group" in text
    assert "VALIDATION_CT_FOLDERS" in text
    assert "selected_global_folders" in text
    assert "REGISTRATION_CT_FOLDERS" in text
    assert '"20230208-160256"' in text


def test_visual_checks_notebook_uses_shared_config_and_documents_recipes():
    path = NOTEBOOK_DIR / "05_ct_combined_volume_and_difference_visualisation.ipynb"
    code_text = notebook_text(path)
    all_text = notebook_all_text(path)

    assert "project_root" in code_text
    assert "load_config(CONFIG_PATH)" in code_text
    assert 'paths_config["ct_global_validation"]' in code_text
    assert "VISUAL_CT_FOLDERS" in code_text
    assert "SELECTED_VISUAL_CT_FOLDERS" in code_text
    assert 'global_visual_root / ct_folder' in code_text
    assert '"copy" not in path.name.casefold()' in code_text
    assert "global_candidates_for_folder" in code_text
    assert "legacy_prefix" in code_text
    assert '"20230208-160256"' in code_text
    assert "from basalt_processing.visual_checks import plot_combined_orthogonal_comparison" in code_text
    assert "plot_combined_orthogonal_comparison(" in code_text
    assert "from basalt_processing.visual_checks import contourf_slices_3d" in code_text
    assert "contourf_slices_3d(" in code_text
    assert "from basalt_processing.zscore_functions import zscore_volume_using_matrix_stats" in code_text
    assert 'handle["threshold_mask_global"]' in code_text
    assert "zscore_volume_using_matrix_stats(volume, pore_mask)" in code_text
    assert "bagit_subpath" not in code_text
    assert "## Load normalized comparison volumes" in all_text
    assert "2D/3D comparison" in all_text
    assert "## 3D contour-slice inspection" in all_text
    assert "## Overlay inspection" in all_text
    assert "# CT Combined volume and difference visualisation" in all_text


def test_pump_and_timeline_notebooks_use_the_shared_configured_roots():
    pump_text = notebook_text(NOTEBOOK_DIR / "06_pump_resistance_permeability_processing.ipynb")
    timeline_text = notebook_text(NOTEBOOK_DIR / "07_timeline_and_key_tables.ipynb")

    for text in (pump_text, timeline_text):
        assert "project_root" in text
        assert "load_config(CONFIG_PATH)" in text
        assert 'paths_config["data_root"]' in text
        assert "bagit_subpath" not in text
        assert "get_bagit_root" not in text

    assert 'paths_config["pump"]' in pump_text
    assert 'paths_config["resistance"]' in pump_text
    assert 'paths_config["output"]' in pump_text
    assert 'paths_config["fluid_state"]' in timeline_text
    assert 'paths_config["ct_processed"]' in timeline_text
    assert 'paths_config["ct_pretest_processed"]' in timeline_text
    assert 'paths_config["ct_pretest_raw"]' in timeline_text


def test_timeline_notebook_initial_cells_load_notebook_06_exports():
    path = NOTEBOOK_DIR / "07_timeline_and_key_tables.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    initial_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"][:8]
        if cell.get("cell_type") == "code"
    )

    assert "project_root" in initial_code
    assert "load_config(CONFIG_PATH)" in initial_code
    assert 'paths_config["fluid_state"]' in initial_code
    assert 'paths_config["output"]' in initial_code
    assert 'output_root / "mechanical.pkl"' in initial_code
    assert 'output_root / "resistivity.pkl"' in initial_code
    assert 'output_root / "resistivity_1000hz.pkl"' in initial_code
    assert 'output_root / "resistivity_5000hz.pkl"' in initial_code
    assert 'output_root / "resistivity_10000hz.pkl"' in initial_code
    assert 'output_root / "CT_datetimes.pkl"' in initial_code
    assert 'resistivity_1000hz_data = pd.read_pickle' in initial_code
    assert 'resistivity_5000hz_data = pd.read_pickle' in initial_code
    assert 'resistivity_10000hz_data = pd.read_pickle' in initial_code
    assert "RUN_TIMELINE_PICKLE_EXPORT = True" in initial_code


def test_timeline_table_builder_uses_processed_folder_names_without_cross_root_relative_paths():
    text = notebook_text(NOTEBOOK_DIR / "07_timeline_and_key_tables.ipynb")

    assert "processed_root = ct_processed_root" in text
    assert '"folder": folder_name' in text
    assert "folder_path.relative_to(data_root)" not in text
    assert 'ct_datetime = pd.Timestamp(info["datetime"])' in text
    assert 'PRETEST_ACQUISITION = "20200275-CYL11159231-1500-100kV-200uA-025mmAg_with_foam"' in text
    assert "PRETEST_FOLDER = ct_pretest_processed_root" in text
    assert '"scan_role": "Pre-test scan"' in text


def test_config_does_not_retain_legacy_bagit_paths():
    config_text = (NOTEBOOK_DIR.parents[1] / "config" / "basalt.example.toml").read_text(encoding="utf-8")

    for key in (
        "bagit_root",
        "raw_ct",
        "processed_ct",
        "pump_data",
        "resistance_data",
        "ct_sample_only_raw",
        "ct_sample_only_processed",
    ):
        assert f"{key} =" not in config_text
    assert 'ct_global_validation = "output/CT/global_registered"' in config_text
    assert 'ct_pretest_processed = "CT/pre-test/processed"' in config_text


def test_pump_notebook_explains_smoothed_flow_rates_and_labels_review_plots():
    path = NOTEBOOK_DIR / "06_pump_resistance_permeability_processing.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    all_text = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    code_text = notebook_text(path)

    assert "centred 10-sample moving average" in all_text
    assert "cumulative-volume difference divided by elapsed seconds" in all_text
    assert "mean of the absolute smoothed rates" in all_text
    assert "ax.set_xlabel" in code_text
    assert "ax.set_ylabel" in code_text


def test_pump_notebook_includes_resistance_cleaning_and_review():
    path = NOTEBOOK_DIR / "06_pump_resistance_permeability_processing.ipynb"
    code_text = notebook_text(path)
    all_text = notebook_all_text(path)

    assert "from basalt_processing.resistance import load_resistance_folder, write_resistance_outputs" in code_text
    assert 'RESISTANCE_PATTERN = "*Hz_avg_combined.csv"' in code_text
    assert "RUN_RESISTANCE_EXPORT = False" in code_text
    assert "clean_resistance_folder(" in code_text
    assert "RUN_TIMELINE_PICKLE_EXPORT" in code_text
    assert 'output_root / "mechanical.pkl"' in code_text
    assert "write_resistivity_pickles(resistivity_by_frequency, output_root)" in code_text
    assert "build_ct_datetime_table(ct_processed_root, ct_raw_root)" in code_text
    assert 'output_root / "CT_datetimes.pkl"' in code_text
    assert "## Load, format, and review resistance data" in all_text
    assert "Resistance channels" in all_text


def test_conversion_notebook_round_trips_exported_amira_to_distinct_hdf5():
    path = NOTEBOOK_DIR / "01_ct_conversion_and_scan_inventory.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]

    export_index = next(
        index for index, text in enumerate(code_cells) if "hdf5_to_amira(" in text
    )
    import_index = next(
        index for index, text in enumerate(code_cells) if "am_to_hdf5(" in text
    )
    import_cell = code_cells[import_index]

    assert import_index == export_index + 1
    assert "RUN_AMIRA_IMPORT = False" in import_cell
    assert "exported_am is None or not exported_am.is_file()" in import_cell
    assert "roundtrip_h5 = (" in import_cell
    assert 'data_root / "output" / "amira-roundtrip"' in import_cell
    assert 'f"{selected_hdf5.stem}.roundtrip.hdf5"' in import_cell
    assert "roundtrip_h5.resolve() == selected_hdf5.resolve()" in import_cell
    assert "am_to_hdf5(exported_am, roundtrip_h5)" in import_cell


def test_ct_crop_and_porosity_notebooks_have_scoped_workflows():
    crop = notebook_all_text(NOTEBOOK_DIR / "02_ct_cylinder_crop_examples.ipynb")
    porosity = notebook_text(NOTEBOOK_DIR / "03_ct_porosity_analysis_and_visualisation.ipynb")

    assert "Automatic Hough-circle workflow" in crop
    assert "Manual centreline and radius workflow" in crop
    assert "write_cylinder_crop(" in crop
    assert "calculate_porosity(" in porosity
    assert "pyvista.Plotter" in porosity
    assert 'h5py.File(PORESPACE_H5, "r+")' not in porosity


def test_ct_crop_notebook_uses_inline_paths_and_shared_source():
    config_path = NOTEBOOK_DIR.parents[1] / "config" / "basalt.example.toml"
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    assert "cylinder_crop" not in config

    code = notebook_text(NOTEBOOK_DIR / "02_ct_cylinder_crop_examples.ipynb")
    for expression in (
        "SOURCE_H5_DATA = resolve_path(",
        "SOURCE_H5_BINARY = resolve_path(",
        '"CT/pre-test/processed/"',
        '"20200275-CYL11159231-1500-100kV-200uA-025mmAg_with_foam.hdf5"',
        '"20200275-CYL11159231-1500-100kV-200uA-025mmAg_with_foam(2).hdf5"',
        'OUTPUT_H5 = resolve_path("output/data_cylinder_crop.hdf5", data_root)',
        'DATA_DATASET = "data"',
        'THRESHOLD_DATASET = "threshold_mask"',
        'MASK_DATASET = "cylinder_mask"',
        'MASKED_DATASET = "data_masked"',
        'with h5py.File(SOURCE_H5_BINARY, "r") as source_file:',
        "data_volume = source_file[DATA_DATASET]",
        "threshold_volume = source_file[THRESHOLD_DATASET]",
        "threshold_volume[z_index]",
        "write_cylinder_crop(",
        "SOURCE_H5_BINARY, OUTPUT_H5",
        "SOURCE_H5_BINARY, MANUAL_OUTPUT_H5",
        'h5py.File(SOURCE_H5_DATA, "r") as data_file',
        'h5py.File(SOURCE_H5_BINARY, "r") as binary_file',
        "data_file[DATA_DATASET][z + 350, 500:1500, 500:1500]",
        "binary_file[THRESHOLD_DATASET][z]",
        "data_slice.shape == threshold_slice.shape == mask_slice.shape",
        "masked_slice = np.where(mask_slice, data_slice, np.nan)",
        "fig, axes = plt.subplots(1, 4, figsize=(20, 5))",
    ):
        assert expression in code

    for obsolete in ("crop_config", "DATA_AM", "THRESHOLD_AM", "am_to_hdf5("):
        assert obsolete not in code


def test_ct_porosity_notebook_uses_shared_configured_paths():
    config_path = NOTEBOOK_DIR.parents[1] / "config" / "basalt.example.toml"
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    porosity_config = config["porosity"]
    assert porosity_config == {
        "porespace_h5": (
            "CT/pre-test/processed/"
            "20200275-CYL11159231-1500-100kV-200uA-025mmAg_with_foam(2)."
            "_data_masked_porespace.hdf5"
        ),
        "cylinder_crop_h5": "CT/pre-test/processed/data_cylinder_crop.hdf5",
        "porespace_dataset": "data",
        "mask_dataset": "cylinder_mask",
    }
    code = notebook_text(NOTEBOOK_DIR / "03_ct_porosity_analysis_and_visualisation.ipynb")
    for expression in (
        "project_root",
        "load_config(CONFIG_PATH)",
        'paths_config["data_root"]',
        'porosity_config = config["porosity"]',
        'resolve_path(porosity_config["porespace_h5"], data_root)',
        'resolve_path(porosity_config["cylinder_crop_h5"], data_root)',
        'porosity_config["porespace_dataset"]',
        'porosity_config["mask_dataset"]',
        "RUN_PYVISTA = False",
    ):
        assert expression in code


def test_ct_crop_notebook_validates_matching_3d_inputs_before_sampling_threshold_slices():
    path = NOTEBOOK_DIR / "02_ct_cylinder_crop_examples.ipynb"
    code = notebook_text(path)
    all_text = notebook_all_text(path)

    assert 'if data_volume.ndim != 3 or threshold_volume.ndim != 3:' in code
    assert 'if threshold_volume.shape != data_shape:' in code
    assert code.index('if threshold_volume.shape != data_shape:') < code.index('threshold_volume[z_index]')
    assert 'sample_z = np.linspace(0, data_shape[0] - 1, num=min(12, data_shape[0]), dtype=int)' in code
    assert "OpenCV Hough-circle detection" in all_text
    assert "Z-Y-X" in all_text
    assert "X-Y" in all_text
    assert "`data`, `cylinder_mask`, and `data_masked`" in all_text
