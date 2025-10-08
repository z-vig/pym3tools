# Standard Libraries
import os
from pathlib import Path
from importlib.resources import files
from typing import Optional, List, Tuple
import tempfile as tf
import subprocess

# Dependencies
import numpy as np

# import rasterio as rio
from rasterio.control import GroundControlPoint  # type: ignore
from rasterio.crs import CRS  # type: ignore

# Relative Imports
from .gcp_loaders import load_gcps, read_gcps_header
from .numpy_to_gtiff import numpy_to_gtiff

# Top-Level Imports

PathLike = str | os.PathLike | Path


def _apply_gcps_from_file(
    arr: np.ndarray,
    gcp_file_path: PathLike,
    dst_path: PathLike,
    input_array_offsets: Tuple[int, int],
    verbose: bool = False,
) -> None:
    """
    Loads Ground Control Points from a *.gcps file and applies them to a numpy
    array. The GCPs in this file are assumed to be in the CGS_MOON_2000
    coordinate reference system. The loaded GCPs are in reference to a region
    of the whole M3 stripe that is recorded by a row/column offset and a width/
    height. These points are transformed to the region outline by the input
    array using:

    `new_gcp` = (`loaded_gcp` + `loaded_offset`) - `input_offset`

    Parameters
    ----------
    arr: np.ndarray
        The first-order GCP georeference will be applied to this array.
    gcp_file_path: PathLike
        File path to the *.gcps Ground Control Point save file. To see how to
        create a *.gcps file see the `georeferencer` subpackage.
    dst_path: PathLike
        File path to save destination.
    input_array_offsets: Tuple of 2 integers
        (row offset, column offset) for the input array from the full
        M<sup>3</sup> strip.
    """
    prj_file = files("m3py.selenography.data").joinpath("cgs_moon_2000.prj")
    with prj_file.open() as f:
        crs = CRS.from_wkt(f.read())

    gcps = load_gcps(gcp_file_path)
    gcp_meta = read_gcps_header(gcp_file_path)

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim > 3:
        raise ValueError(f"{arr.ndim} dimensions is too big. Must be 2 or 3.")

    row_off, col_off = input_array_offsets
    new_gcps = []
    for i in gcps:
        if i.row is None:
            continue
        if i.col is None:
            continue

        new_row = (i.row + gcp_meta.row_offset) - row_off
        new_col = (i.col + gcp_meta.col_offset) - col_off

        if (new_row > arr.shape[0]) or (new_row < 0):
            continue

        if (new_col > arr.shape[1]) or (new_col < 0):
            continue

        new_gcps.append(
            GroundControlPoint(row=new_row, col=new_col, x=i.x, y=i.y)
        )

    tempfile = numpy_to_gtiff(arr, crs)
    warp_to_gcps(tempfile, new_gcps, dst_path=dst_path)
    os.remove(tempfile)


def _apply_gcps_from_geolocation_array(
    arr: np.ndarray, loc: np.ndarray, dst_path: PathLike, verbose: bool = False
) -> None:
    """
    Collects Ground Control Points from a Geolocation Array backplane and
    applies them to a numpy array. For M<sup>3</sup>, this is provided by the
    NASA PDS. It is assumed that the geolocation array is in the CGS_MOON_2000
    coordinate reference system.

    Parameters
    ----------
    arr: np.ndarray
        The first-order GCP georeference will be applied to this array.
    loc: np.ndarray
        The geolocation backplane array from the NASA PDS. This must be the
        same size as `arr`. It must be 3D with the third axis labels being:
        [longitude, latitude, elevation] in that order.
    dst_path: PathLike
        File path to save destination.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If arr and loc are not the same shape.

    Notes
    -----
    This georeferencing technique has a known offset associated with it, due to
    inaccuracies in the Geolocation Array, so any regional M<sup>3</sup>
    analyses will need hand-picked Ground Control Points to be accurately
    georeferenced.
    """
    if arr.shape != loc.shape:
        raise ValueError(
            f"arr of size {arr.shape} is not compatible with"
            f"loc of size {loc.shape}. They must be the same"
            "shape"
        )

    prj_file = files("m3py.selenography.data").joinpath("cgs_moon_2000.prj")
    with prj_file.open() as f:
        crs = CRS.from_wkt(f.read())

    gcps = []
    for n, i in enumerate(range(0, loc.shape[0])):
        gcps.append(
            GroundControlPoint(row=i, col=0, x=loc[i, 0, 0], y=loc[i, 0, 1])
        )
        gcps.append(
            GroundControlPoint(
                row=i, col=loc.shape[1] - 1, x=loc[i, -1, 0], y=loc[i, -1, 1]
            )
        )

    tempfile = numpy_to_gtiff(arr, crs)

    warp_to_gcps(tempfile, gcps, dst_path=dst_path)

    os.remove(tempfile)


def apply_gcps(
    arr: np.ndarray,
    gcp_pointer: np.ndarray | PathLike,
    dst_path: PathLike,
    input_array_offsets: Tuple[int, int],
    verbose: bool = False,
) -> None:
    """
    Applies Ground Control Points to a numpy array and saves it to dst_path as
    a GeoTiff.

    Parameters
    ----------
    arr: np.ndarray
        Array to be georeferenced.

    gcp_pointer: np.ndarray | PathLike
        If an array is passed, it is assumed to be a geolocation array and
        `apply_gcps_from_geolocation_array` will be applied. If a file path
        is passed, this is assumed to be a path to a *.gcps file and
        `apply_gcps_from_file` will be used.

    dst_path: PathLike
        Save file path. Saved file will be in the GeoTiff format.

    input_array_offsets: Tuple of 2 integers
        (row offset, column offset) for the input array from the full
        M<sup>3</sup> strip.
    """
    if isinstance(gcp_pointer, np.ndarray):
        _apply_gcps_from_geolocation_array(arr, gcp_pointer, dst_path, verbose)
    elif isinstance(gcp_pointer, PathLike):
        _apply_gcps_from_file(
            arr, gcp_pointer, dst_path, input_array_offsets, verbose
        )
    else:
        raise TypeError(
            f"{type(gcp_pointer)} is an invalid type for a gcp" "pointer."
        )


def warp_to_gcps(
    src_path: PathLike,
    gcps: List[GroundControlPoint],
    dst_path: Optional[PathLike] = None,
    gdal_conda_env_name: str = "gdal",
    verbose: bool = False,
):
    """
    Uses GDALs warp functionality to warp an image to a set of Ground Control
    Points using the thin plate spline algorithm.

    Parameters
    ----------
    src_path: PathLike
        Path to source image.
    gcps: List of GroundControlPoint
        GCP List.
    dst_path: PathLike
        Path to save destination. If None (default), the destination path will
        be the same as the src_path, but with the file ending in *_gcp.tif.
    gdal_conda_env_name: str
        Name of the conda environment with gdal installed. This will be called
        via the user's command line using Python's `subprocess` library.
    """
    # Getting Source File Path
    src_path = Path(src_path)

    # Getting Destination File Path
    if dst_path is None:
        dst_path = src_path.with_name(f"{src_path.stem}_gcp.tif")
    else:
        dst_path = Path(dst_path)

    # Getting GCS Moon 2000 projection, which is the default coordinate
    # System for M3 backplanes.
    prj_file = files("m3py.selenography.data").joinpath("cgs_moon_2000.prj")

    tempfile = tf.NamedTemporaryFile(suffix=".tif")
    tempfile.close()
    tempfile_path = Path(tempfile.name)

    gcp_list_as_strings = [f"{i.col} {i.row} {i.x} {i.y}" for i in gcps]
    gdal_translate = (
        "gdal_translate -gcp "
        f"{' -gcp '.join(gcp_list_as_strings)} "
        f"{src_path} {tempfile_path}"
    )

    gdal_warp = (
        "gdalwarp -r near -tps -t_srs "
        f"{prj_file} {tempfile_path} {dst_path}"
    )

    def run_command_in_conda_env(env_name: str, command_str: str):
        out = subprocess.run(
            f"conda run -n {env_name} {command_str}".split(),
            shell=True,
            capture_output=True,
        )
        if verbose:
            print(command_str)
            print(f"----STDOUT----\n{out.stdout.decode("utf-8")}")
            print(f"----STDERR----\n{out.stderr.decode("utf-8")}")

    run_command_in_conda_env(gdal_conda_env_name, gdal_translate)
    run_command_in_conda_env(gdal_conda_env_name, gdal_warp)
