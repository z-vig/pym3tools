"""
Welcome to `pym3tools`!
---

Pure python code for downloading, georeferencing and processing data from\
NASA's Moon Mineralogy Mapper (M\u00b3). Downloads access NASA's planetary
data system (PDS) to get indivudal M\u00b3 datasets.

#### Available Subpackages:
- `data_retrieval`: Downloading and data structure
- `geo_ops`: Geographic operations such as georeferencing
- `m3catalog`: Data structure for M\u00b3 metadata
- `rdn2rfl`: Radiance (L1) to Reflectance (L2) Pipeline
- `save_models`: Data structures for saving processed data
"""

from .data_retrieval.data_directory import M3DataPaths

__all__ = ["M3DataPaths"]
