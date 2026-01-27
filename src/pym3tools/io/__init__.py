"""
### I/O

Module containing code for reading M3 binary image data and writing this data
to hdf5 or geotiff formats.
"""

from .read_m3_binary import read_m3, Window, get_wavelengths
from .read_m3_georef import read_m3_georef
from .read_m3_rasterio import read_m3_rio
from .write_raster import write_to_raster

__all__ = [
    "read_m3",
    "read_m3_georef",
    "read_m3_rio",
    "get_wavelengths",
    "Window",
    "write_to_raster",
]
