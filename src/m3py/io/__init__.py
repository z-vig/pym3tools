"""
### I/O

Module containing code for reading M3 binary image data and writing this data
to hdf5 or geotiff formats.
"""

from .read_m3 import read_m3, Window, get_wavelengths

__all__ = [
    "read_m3",
    "get_wavelengths",
    "Window"
]
