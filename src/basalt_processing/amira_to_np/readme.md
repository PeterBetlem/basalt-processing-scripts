# amira_helper.py Reference

This document provides a summary of the functionality found in [`amira_helper.py`](https://github.com/biomedisa/biomedisa/blob/master/biomedisa/features/amira_to_np/amira_helper.py).

## Overview

The `amira_helper.py` module contains helper functions for reading and processing Amira Mesh files, typically used in biomedical image analysis.

## Key Functions

- **read_amira(filename)**  
    Reads Amira Mesh files and extracts data arrays.

- **parse_amira_header(header_lines)**  
    Parses the header of Amira files to extract metadata.

- **get_data_type(header_lines)**  
    Determines the data type of the mesh data.

- **read_data_block(file, dtype, shape)**  
    Reads the binary data block from the file.

## Usage Example

```python
from amira_helper import read_amira

data, metadata = read_amira('example.am')
print(data.shape)
print(metadata)
```

## Resources

- [biomedisa/biomedisa GitHub Repository](https://github.com/biomedisa/biomedisa)
- [amira_helper.py Source](https://github.com/biomedisa/biomedisa/blob/master/biomedisa/features/amira_to_np/amira_helper.py)
