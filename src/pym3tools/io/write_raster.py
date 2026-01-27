# Standard Libraries
from typing import Optional, Literal, Any
import tempfile as tf
from warnings import warn

# Dependencies
import numpy as np
import rasterio as rio  # type: ignore
from rasterio.crs import CRS  # type: ignore
from affine import Affine  # type: ignore

# Relative Imports
from . import envi_header_writers as hdr_writer

# Top-Level Imports
from pym3tools.types import PathLike, Path

type SaveModeType = Literal["ENVI", "GTiff"]


class BandLabelError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def _validate_gtrans(gtrans: Optional[Affine]) -> Affine:
    "Returns a null transform if the transform is None."
    if gtrans is None:
        return Affine(0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
    return gtrans


def _validate_array_size(arr: np.ndarray) -> np.ndarray:
    """
    If the array has only 2 dimenions, adds at dummy axis at axis=2.

    Raises
    ------
    ValueError
        If the number of array dimenions is not 2 or 3.
    """
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]

    if arr.ndim not in [2, 3]:
        raise ValueError(
            f"Array shape of {arr.shape} is invalid. Wrong number"
            "of dimensions. It must be either 2 or 3."
        )

    return arr


def _validate_band_lbls(
    arr: np.ndarray, band_lbls: Optional[list[float] | list[str]]
) -> None:
    """
    Ensures band labels are the same length as the axis=2 of the array.

        Raises
    ------
    BandLabelError
        If band labels are not the same length as axis=2
    """
    if band_lbls is not None:
        if len(band_lbls) != arr.shape[2]:
            raise BandLabelError(
                f"{len(band_lbls)} were provided for an image cube of "
                f"{arr.shape[2]} bands."
            )


def _write_geotiff(
    arr: np.ndarray,
    profile: dict[str, Any],
    dst_path: str | Path,
) -> None:
    """
    Writes array to a geotiff file.

    Parameters
    ----------
    arr: np.ndarray
        The array to write to file. Bands should be at axis=2.
    profile: dict
        The rasterio profile to use for writing the file.
    dst_path: str | Path
        The destination filepath.
    """
    profile["driver"] = "GTiff"
    profile["count"] = arr.shape[2]
    with rio.open(dst_path, "w", **profile) as dst:
        for i in range(1, arr.shape[2] + 1):
            dst.write(arr[:, :, i - 1], i)


def _write_bil(
    arr: np.ndarray, profile: dict[str, Any], dst_path: Path
) -> None:
    """
    Writes a band interleaved by line file in ENVI-compatible format.
    """
    profile["driver"] = "ENVI"
    profile["interleave"] = "bil"
    profile["count"] = arr.shape[2]
    with rio.open(dst_path.with_suffix(".bil"), "w", **profile) as dst:
        for i in range(1, arr.shape[2] + 1):
            dst.write(arr[:, :, i - 1], i)


def _get_dst_path(dst_path: Optional[str | PathLike]) -> Path:
    """Creates a temporary path if dst_path is None."""
    if dst_path is None:
        tempfile = tf.NamedTemporaryFile(suffix=".tif", delete=False)
        tempfile.close()
        return Path(tempfile.name)
    return Path(dst_path).with_suffix(".tif")


def _add_bad_bands(arr: np.ndarray, bbl: list[bool]) -> np.ndarray:
    """Adds bad bands according to bad band list."""
    if arr.shape[2] == len(bbl):
        warn("Array matches bbl length. Bad Bands not added.")
        return arr
    elif arr.shape[2] < len(bbl):
        # Good band size check
        if arr.shape[2] != len([v for v in bbl if v]):
            raise BandLabelError(
                "Number of good bands in bbl does not match array size."
            )
        arr_with_bad_bands = np.empty((*arr.shape[:2], len(bbl)))
        ngoodbands = 0
        for n, b in enumerate(bbl):
            if b:
                arr_with_bad_bands[:, :, n] = arr[:, :, ngoodbands]
                ngoodbands += 1
            else:
                arr_with_bad_bands[:, :, n] = -999 * np.ones(arr.shape[:2])
        return arr_with_bad_bands
    else:
        raise BandLabelError("Array is larger than bbl.")


def write_to_raster(
    arr: np.ndarray,
    crs: CRS,
    gtrans: Optional[Affine] = None,
    band_lbls: Optional[list[str] | list[float]] = None,
    dst_path: Optional[PathLike] = None,
    save_mode: SaveModeType = "ENVI",
    wavelength_field: bool = False,
    bbl: Optional[list[bool]] = None,
) -> Path:
    """
    Saves a numpy array to a georefenced array file in one of two modes.

    Parameters
    ----------
    arr: np.ndarray
        Array to save as GeoTiff. Bands should be at axis=2.
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
    bbl: list[bool], optional
        List of bad bands where False is indicative of a bad band.
    add_bad_bands: bool
        Choose to add in blank bands where the bad band list is False.

    Returns
    -------
    Path
    """
    # Validating inputs
    gtrans = _validate_gtrans(gtrans)
    arr = _validate_array_size(arr)

    # Defining standard ENVI Profile
    profile: dict[str, Any] = {
        "dtype": arr.dtype,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "crs": crs,
        "transform": gtrans,
        "nodata": -999,
    }

    # Temporary path if dst_path is None
    dst_path = _get_dst_path(dst_path)

    # Writing files
    if save_mode == "GTiff":
        _write_geotiff(arr, profile, dst_path)

    if save_mode == "ENVI":
        if bbl is not None:
            arr = _add_bad_bands(arr, bbl)
        _validate_band_lbls(arr, band_lbls)
        _write_bil(arr, profile, dst_path)

        # Writing HDR File
        if (band_lbls is not None) and wavelength_field:
            hdr_writer.add_wavelength_field(band_lbls, dst_path)
        if (band_lbls is not None) and not wavelength_field:
            hdr_writer.replace_band_labels_field(band_lbls, dst_path)
        if bbl is not None:
            hdr_writer.add_bbl_field(bbl, dst_path)

    return dst_path
