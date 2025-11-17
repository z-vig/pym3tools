# Standard Libraries
from typing import Optional, Literal
import tempfile as tf
import re

# Dependencies
import numpy as np
import rasterio as rio  # type: ignore
from rasterio.crs import CRS  # type: ignore

# Top-Level Imports
from pym3tools.types import PathLike, Path

type SaveModeType = Literal["ENVI", "GTiff"]


class BandLabelError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def write_to_raster(
    arr: np.ndarray,
    crs: CRS,
    gtrans: Optional[tuple[float, ...]] = None,
    band_lbls: Optional[list[str | float]] = None,
    dst_path: Optional[PathLike] = None,
    save_mode: SaveModeType = "ENVI",
    wavelength_field: bool = False,
) -> Path:
    """
    Saves a numpy array to a georefenced array file in one of two modes.

    Parameters
    ----------
    arr: np.ndarray
        Array to save as GeoTiff.
    crs: CRS
        A rasterio CRS type. The array will be saved to this type.
    gtrans: tuple of floats, optional
        Geotransform to be applied. If None (default), a default geotransform
        of (0, 1, 0, 0, 0, 1) will be used.
    band_lbls: list of numbers or strings, optional
        This list will be used to name the bands for reading by ArcGIS, etc...
    dst_path: PathLike, optional
        Destination path. If None (default), the file will be saved as a temp
        file, and the name of the temp file will be returned. If the tempfile
        is returned, it will exist indefinitely until deleted by another
        function.
    save_mode: "ENVI" or "GTiff"
        Chooses the driver that is used to save the raster data.
    wavelength_field: bool
        If "ENVI" is the chosen save mode, use this to specify whether the
        band_lbls are a wavelength field (True) or some other type of label
        (False). Default is False.

    Returns
    -------
    Path
    """
    if gtrans is None:
        gtrans = (0, 1, 0, 0, 0, 1, 0)

    # Adjusting for dimensionality of the dataset (2D or 3D).
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    if arr.ndim not in [2, 3]:
        raise ValueError(
            f"Array shape of {arr.shape} is invalid. Wrong number"
            "of dimensions. It must be either 2 or 3."
        )

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
    # Saves to a temp file if dst_path is not specified.
    if dst_path is None:
        tempfile = tf.NamedTemporaryFile(suffix=".tif", delete=False)
        tempfile.close()
        dst_path = Path(tempfile.name)
    else:
        dst_path = Path(dst_path).with_suffix(".tif")

    if save_mode == "GTiff":
        with rio.open(dst_path, "w", **profile) as dst:
            for i in range(1, arr.shape[2] + 1):
                dst.write(arr[:, :, i - 1], i)

    if save_mode == "ENVI":
        if band_lbls is not None:
            if len(band_lbls) != arr.shape[2]:
                raise BandLabelError(
                    f"{len(band_lbls)} were provided for an image cube of "
                    f"{arr.shape[2]} bands."
                )
        profile["driver"] = "ENVI"
        with rio.open(dst_path.with_suffix(".bsq"), "w", **profile) as dst:
            for i in range(1, arr.shape[2] + 1):
                dst.write(arr[:, :, i - 1], i)

        if (band_lbls is not None) and wavelength_field:
            hdr_lines = [
                "wavelength units = nm",
                "wavelength = {" + ", ".join(map(str, band_lbls)) + "}",
            ]
            with open(Path(dst_path).with_suffix(".hdr"), "a") as f:
                f.write("\n".join(hdr_lines))

        if (band_lbls is not None) and not wavelength_field:
            patt = re.compile(r"band names = {[\d\D]+}\n")

            new_band_names = "band names = {\n"
            for lbl in band_lbls:
                new_band_names += f"{lbl},\n"
            new_band_names += "}\n"

            with open(Path(dst_path).with_suffix(".hdr"), "r") as f:
                fread = f.read()

            search = re.search(patt, fread)
            if search is not None:
                match_slice = slice(*search.span(), 1)

                new_fread = fread.replace(fread[match_slice], new_band_names)

                with open(Path(dst_path).with_suffix(".hdr"), "w") as f:
                    f.write(new_fread)
            else:
                raise ValueError(".hdr file is in an incompatible format.")

    return dst_path
