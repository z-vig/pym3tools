# Standard Libraries
from tempfile import NamedTemporaryFile
from typing import Optional
from pathlib import Path

# Dependencies
import rasterio as rio  # type: ignore
from rasterio.merge import merge  # type: ignore
from rasterio.crs import CRS  # type: ignore
import numpy as np
import numpy.typing as npt

# Relative Imports
from .numpy_to_gtiff import numpy_to_gtiff

# Top-Level Imports
from pym3tools.types import PathLike


def mosaic_arrays(
    arr_list: list[np.ndarray],
    gtrans_list: list[tuple[float, ...]],
    crs: CRS,
    save_path: PathLike,
    band_lbls: Optional[npt.NDArray | list] = None,
    wavelength_field: bool = False,
):
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
    profile.update(
        {
            "driver": "ENVI",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform,
            "nodata": -999,
        }
    )

    with rio.open(Path(save_path).with_suffix(".bsq"), "w", **profile) as dst:
        for band in range(1, mosaic.shape[0] + 1):
            dst.write(mosaic[band - 1, :, :], band)
            if band_lbls is not None:
                if not wavelength_field:
                    dst.set_band_description(band, band_lbls[band - 1])

    if (band_lbls is not None) and wavelength_field:
        hdr_lines = [
            "wavelength units = nm",
            "wavelength = {" + ", ".join(map(str, band_lbls)) + "}",
        ]
        with open(Path(save_path).with_suffix(".hdr"), "a") as f:
            f.write("\n".join(hdr_lines))

    for i in mosaic_list:
        if not isinstance(i, str):
            i.close()
