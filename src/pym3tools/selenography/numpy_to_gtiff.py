# Standard Libraries
import os
from pathlib import Path
from typing import Optional
import tempfile as tf

# Dependencies
import numpy as np
import rasterio as rio  # type: ignore
from rasterio.crs import CRS  # type: ignore

PathLike = Optional[str | os.PathLike | Path]


def numpy_to_gtiff(
    arr: np.ndarray,
    crs: CRS,
    gtrans: Optional[tuple[float, ...]] = None,
    band_names: Optional[list[str | float]] = None,
    dst_path: Optional[PathLike] = None,
) -> Path:
    """
    Saves a numpy array to a file with a dummy geotransform [0,1,0,0,0,1,0] and
    a specified Cooridinate Reference System.

    Parameters
    ----------
    arr: np.ndarray
        Array to save as GeoTiff.
    crs: CRS
        A rasterio CRS type. The array will be saved to this type.
    gtrans: tuple of floats, optional
        Geotransform to be applied. If None (default), a default geotransform
        of (0, 1, 0, 0, 0, 1) will be used.
    band_names: list of numbers or strings, optional
        This list will be used to name the bands for reading by ArcGIS, etc...
    dst_path: PathLike, optional
        Destination path. If None (default), the file will be saved as a temp
        file, and the name of the temp file will be returned. If the tempfile
        is returned, it will exist indefinitely until deleted by another
        function.

    Returns
    -------
    Path
    """
    if gtrans is None:
        gtrans = (0, 1, 0, 0, 0, 1, 0)

    profile = {
        "driver": "GTiff",
        "dtype": arr.dtype,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "count": arr.shape[2],
        "crs": crs,
        "transform": gtrans,
        "nodata": -999,
    }

    if dst_path is None:
        tempfile = tf.NamedTemporaryFile(suffix=".tif", delete=False)
        tempfile.close()
        dst_path = Path(tempfile.name)
    else:
        dst_path = Path(dst_path)

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    if arr.ndim not in [2, 3]:
        raise ValueError(
            f"Array shape of {arr.shape} is invalid. Wrong number"
            "of dimensions. It must be either 2 or 3."
        )

    with rio.open(dst_path, "w", **profile) as dst:
        for i in range(1, arr.shape[2] + 1):
            dst.write(arr[:, :, i - 1], i)
            if band_names is not None:
                print(i - 1)
                dst.set_band_description(i, band_names[i - 1])

    return dst_path
