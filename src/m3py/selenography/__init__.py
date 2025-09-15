"""
### Selenography

Module contains specialized geospatial functions for visualizing data on the
Moon. This module relies on the use of the python standard library `subprocess`
and a user-installed `conda` environment with `gdal` installed. To create
a conda environment, you can use the "gdal_env.yml" that is shipped with the
package (conda env create -f gdal_env.yml) or use the following series of
commands to install it mannually:

~~~
> conda create -n gdal
> conda activate gdal
> conda install gdal
~~~
"""

from .crop import polar_crop, regional_crop
from .gcp_utils import apply_gcps
from .gcp_loaders import load_gcps, read_gcps_header, read_gcps
from .basic_pixel_alignment import align_pixels
from .mosaic import mosaic_arrays
from .numpy_to_gtiff import numpy_to_gtiff

__all__ = [
    "polar_crop",
    "regional_crop",
    "apply_gcps",
    "load_gcps",
    "read_gcps",
    "read_gcps_header",
    "align_pixels",
    "mosaic_arrays",
    "numpy_to_gtiff",
]
