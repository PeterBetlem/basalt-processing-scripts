# Basalt Processing Scripts

Processing tools and reproducible reference notebooks for the Basalt core-flooding CT, pump, resistance, permeability, and timeline workflows.

## Install and run

Use the project environment through `uv`:

```powershell
uv sync --extra dev
uv run basalt-process-pump --help
```

The notebooks locate the project source directory themselves. Select the project `uv` environment as the VS Code notebook kernel.

## Configuration

[`config/basalt.example.toml`](config/basalt.example.toml) describes the processing layout. `data_root` is the base path; all other paths are resolved relative to it unless absolute.

| Key | Purpose |
|---|---|
| `ct_raw`, `ct_processed` | Raw BagIt scan folders and processed CT volumes |
| `ct_pretest_raw`, `ct_pretest_processed` | Pre-test CT acquisition and processed volume |
| `ct_global_validation` | Global CT volumes written and validated by notebook 04, then visualized in notebook 05 |
| `fluid_state`, `resistance`, `pump` | Experiment data roots |
| `output` | Derived tables, pickle exports, figures, and other outputs |
| `[porosity]` | Notebook 03 porespace and cylinder-crop inputs and dataset names |

Relative paths in the workflow-specific tables are also resolved beneath `data_root`.

## Command-line scripts

| Command | Purpose | Notebook counterpart |
|---|---|---|
| `basalt-convert-amira` | Convert an Amira/Avizo `.am` volume to HDF5. | 01 |
| `basalt-convert-hdf5-to-amira` | Export an HDF5 volume as an Amira `.am` file. | 01 |
| `basalt-register-global` | Register CT volumes on a shared global grid. | 04 |
| `basalt-process-pump` | Clean and merge MODLab/Autolab pump exports. | 06 |
| `basalt-process-resistance` | Clean and merge resistance exports. | 06 |
| `basalt-process-permeability` | Add pump-derived flow-rate columns for permeability review. | 06 |
| `basalt-plot-timelines` | Create a simple pump/resistance timeline plot. | 07 provides the publication-focused timeline and key table. |

Example invocations:

```powershell
uv run basalt-convert-amira input.am output.hdf5 --dset data --gzip 4
uv run basalt-convert-hdf5-to-amira output.hdf5 output.am --dataset data_masked
uv run basalt-register-global --inputs scan1.hdf5 scan2.hdf5 --outputs scan1._global.hdf5 scan2._global.hdf5
uv run basalt-process-pump pump-1.txt pump-2.txt --output output/pump
uv run basalt-process-resistance resistance --pattern "*Hz_avg_combined.csv" --output output/resistance
uv run basalt-process-permeability output/pump.csv --volume-columns "GDS Perm Volume A_Eng" --output output/permeability.csv
uv run basalt-plot-timelines --pump output/pump.csv --resistance output/resistance.csv --output output/timeline.png
```

## Reference notebooks

The notebooks are ordered by workflow, with focused CT preparation and
analysis first, followed by registration/visual comparison and the
experiment-wide pump, resistance, permeability, and timeline workflows.

| Notebook | Scope |
|---|---|
| `01_ct_conversion_and_scan_inventory.ipynb` | Inventories processed CT scans; inspects HDF5 datasets and stored Amira headers; exports HDF5 data to Amira; and demonstrates an Amira-to-HDF5 round trip while preserving header metadata. |
| `02_ct_cylinder_crop_examples.ipynb` | Documents the pre-test crop coordinates; validates aligned source datasets; detects a cylinder with sampled Hough circles; writes automatic or reviewed manual cylinder crops; and compares original intensity, threshold mask, cylinder mask, and masked intensity slices. |
| `03_ct_porosity_analysis_and_visualisation.ipynb` | Calculates read-only per-Z porespace fractions inside the cylinder mask, plots selected porespace/mask slices, and optionally displays the volume with PyVista. |
| `04_ct_global_registration.ipynb` | Defines the registration run order; supports manual and stored-transform registration; transforms derived datasets; and validates that registered CT volumes share compatible global coordinates. |
| `05_ct_combined_volume_and_difference_visualisation.ipynb` | **CT Combined volume and difference visualisation:** loads configured registered volumes, standardizes comparison ranges, computes normalized differences, and provides middle-slice, overlay, 3-D contour-slice, and combined 2-D/3-D checks. |
| `06_pump_resistance_permeability_processing.ipynb` | Cleans and reviews pump and resistance exports; derives smoothed flow and pressure series; evaluates permeability windows and calculations; and optionally writes the pickle inputs used for the publication timeline. |
| `07_timeline_and_key_tables.ipynb` | Loads the prepared pump, resistance, and CT timing exports; builds the publication-focused experiment timeline; and produces CT date/time and scan-key tables. |

Notebook 6 produces the timeline inputs consumed by notebook 7 when
`RUN_TIMELINE_PICKLE_EXPORT` is enabled: `mechanical.pkl`, combined and
frequency-specific resistivity pickles, and `CT_datetimes.pkl`.

## Library-only helpers

`hdf5_inspection`, `ct_inventory`, `visual_checks`, and `zscore_functions` support the reference notebooks but do not currently expose command-line entry points.

## License

This repository is licensed under the [European Union Public Licence 1.2](LICENSE).
It includes adapted Amira-reading components whose original EUPL-1.2 and
Apache-2.0 terms and attributions are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
