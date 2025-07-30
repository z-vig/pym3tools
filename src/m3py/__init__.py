"""
# m3py

Data utilities for processing Moon Mineralogy Mapper Data from the NASA
Planetary Data System. Available modules include:

- **io** // Reading and writing utilities. Files can be read from the native
        NASA PDS format (i.e. a disk image with the suffix *.img). Files are
        written to HDF5.
- **selenography** // Spatial processing utilities including projecting data
                  into geographic or projected coorindate systems, cropping
                  larger data swaths and precise georeferencing.
- **level2pipeline** // Converts level 1 radiance data from the PDS to level 2
                    reflectance data using various configuration options. The
                    pipeline is configured using *.yaml files.
"""

from . import constants
from . import io
from . import level2pipeline
from . import selenography
from .PDSretrieval import M3FileConfig, get_m3_id

__all__ = [
    "constants",
    "io",
    "level2pipeline",
    "selenography",
    "M3FileConfig",
    "get_m3_id"
]
