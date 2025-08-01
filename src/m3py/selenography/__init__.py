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

from .crop import polar_crop
from .gcp_utils import apply_gcps
from .gcp_loaders import load_gcps, read_gcps_header

__all__ = [
    "polar_crop",
    "apply_gcps",
    "load_gcps",
    "read_gcps_header"
]
