"""
### I/O

Module containing code for reading M3 binary image data and writing this data
to hdf5 or geotiff formats.
"""

from .m3_data_class import M3Data
from .read_m3 import read_m3, Window, get_wavelengths
from .m3_data_window import show_lazy_image

__all__ = [
    "M3Data",
    "read_m3",
    "get_wavelengths",
    "Window",
    "show_lazy_image"
]
