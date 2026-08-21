# Third-party notices

This repository is distributed under the European Union Public Licence 1.2
(EUPL-1.2). It includes or adapts the following third-party software. The
original notices and licence terms remain applicable to those components.

## Biomedisa Amira support

The following files are derived from the Biomedisa project:

- `src/basalt_processing/amira_to_np/amira_helper.py`
- `src/basalt_processing/amira_to_np/mesh.py`

Biomedisa is copyright Philipp Lösel and the Biomedisa contributors and is
distributed under EUPL-1.2. These files were incorporated into the
`basalt_processing.amira_to_np` namespace on 2026-07-22.

Upstream source: <https://github.com/biomedisa/biomedisa>

## ahds Amira parser

The following files contain code derived from the `ahds` Amira header and data
stream parser, obtained through Biomedisa's bundled Amira support:

- `src/basalt_processing/amira_to_np/amira_data_stream.py`
- `src/basalt_processing/amira_to_np/amira_grammar.py`
- `src/basalt_processing/amira_to_np/amira_header.py`

`ahds` is copyright Paul K. Korir and contributors and is distributed under
the Apache License 2.0. The Apache licence text is included at
`LICENSES/Apache-2.0.txt`. These files were incorporated into the
`basalt_processing.amira_to_np` namespace on 2026-07-22 and are distributed as
part of this EUPL-1.2 project; the original Apache-2.0 terms and attribution
remain applicable to the upstream material.

Upstream source: <https://github.com/emdb-empiar/ahds>

## Modification record

- 2026-07-22: the files listed above were incorporated into the
  `basalt_processing.amira_to_np` namespace for Amira reading, writing, and
  voxel-spacing support.
- 2026-07-24: `amira_data_stream.py` was changed to use
  `numpy.ndarray.tobytes()` instead of the deprecated
  `numpy.ndarray.tostring()` when writing Amira data.
- 2026-08-21: SPDX identifiers and third-party provenance notices were added;
  no functional third-party code changes were made on this date.
