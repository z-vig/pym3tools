# Standard Libraries
from tempfile import NamedTemporaryFile
from typing import Optional

# Dependencies
import rasterio as rio  # type: ignore
from rasterio.merge import merge  # type: ignore
from rasterio.crs import CRS  # type: ignore
import numpy as np

# Relative Imports
from .numpy_to_gtiff import numpy_to_gtiff

# Top-Level Imports
from pym3tools.types import PathLike
from pym3tools.io.write_raster import write_to_raster


def mosaic_arrays(
    arr_list: list[np.ndarray],
    gtrans_list: list[tuple[float, ...]],
    crs: CRS,
    save_path: PathLike,
    band_lbls: Optional[list] = None,
    wavelength_field: bool = False,
    bbl: Optional[list[bool]] = None,
):
    """
    Writes a list of images to disk as a single composite image.

    Parameters
    ----------
    arr_list: list of np.ndarray
        List of image arrays to be mosaicked.
    gtrans_list: list of tuples
        List of geotransform tuples corresponding to each array in `arr_list`.
    crs: CRS
        The coordinate reference system object for the mosaic.
    save_path: PathLike
        File path to save the mosaic.
    band_lbls:
    """
    temp_file_list: list[str] = []
    for arr, gtrans in zip(arr_list, gtrans_list):
        temp = NamedTemporaryFile(suffix=".tif")
        temp.close()
        temp_file_list.append(temp.name)

        arr[np.isnan(arr)] = -999

        numpy_to_gtiff(arr, crs, gtrans, dst_path=temp.name)

    mosaic_list: list[rio.DatasetReader] = []
    for i in temp_file_list:
        src = rio.open(i)
        mosaic_list.append(src)

    mosaic, mosaic_transform = merge(mosaic_list, method="max")
    profile = mosaic_list[0].profile.copy()

    print(f"Writing mosaic of size: {mosaic.shape} to {save_path}")

    mosaic = np.transpose(mosaic, (1, 2, 0))
    write_to_raster(
        mosaic,
        profile["crs"],
        mosaic_transform,
        band_lbls=band_lbls,
        dst_path=save_path,
        save_mode="ENVI",
        wavelength_field=wavelength_field,
        bbl=bbl,
    )
