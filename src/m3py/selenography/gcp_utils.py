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
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS

# Relative Imports
from .gcp_loaders import load_gcps, read_gcps_header
from .numpy_to_gtiff import numpy_to_gtiff

# Top-Level Imports

PathLike = str | os.PathLike | Path


def apply_gcps_from_file(
    arr: np.ndarray,
    gcp_file_path: PathLike,
    dst_path: PathLike,
    arr_cropping: str = 'none',
    input_array_offsets: Tuple[int, int] = (0, 0)
) -> None:
    """
    Loads Ground Control Points from a *.gcps file and applies them to a numpy
    array. The GCPs in this file are assumed to be in the CGS_MOON_2000
    coordinate reference system. Rather than applying GCPs and saving the full
    extent of the input array as a GeoTiff, if any part of the input array
    falls outside of the extent for which the gcps in the *.gcps file are valid
    (as specified by the row/column offsets, width and height), the saved
    GeoTiff will be a cropped version of the input array.

    Parameters
    ----------
    arr: np.ndarray
        The first-order GCP georeference will be applied to this array.
    gcp_file_path: PathLike
        File path to the *.gcps Ground Control Point save file. To see how to
        create a *.gcps file see the `georeferencer` subpackage.
    dst_path: PathLike
        File path to save destination.
    arr_cropping: str, optional
        The pixel row/column coordinates in a *.gcps file are typically
        obtained using a cropped version of a full M<sup>3</sup> strip. This
        flag indicates the relationship between the input array and the array
        used to perform the georeferencing. Options are:

        - `'none'` (default): Assumes that arr is a full M<sup>3</sup> strip
        and will add the row and column offsets to the pixel row/column coords.
        - `'same'`: Assumes that arr is already cropped to the same extent as
        the regional image used for georeferencing (i.e. where GCPs are valid)
        and the full extent of the input array will be saved as a GeoTiff.
        - `'different'`: Assumes that arr is cropped but to a different region
        than the valid region of the GCPs. The resulting GeoTiff will be the
        intersection of the input array and the valid GCP region. If this
        option is chosen, `input_array_offsets` must be given.

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

    if arr_cropping == 'none':
        arr = arr[
            gcp_meta.row_offset:gcp_meta.row_offset+gcp_meta.height,
            gcp_meta.col_offset:gcp_meta.col_offset+gcp_meta.width,
            :
        ]
    elif arr_cropping == 'same':
        pass
    elif arr_cropping == 'different':
        raise NotImplementedError("Input array cannot be of a different"
                                  "cropped region at this time.")

    tempfile = numpy_to_gtiff(arr, crs)
    warp_to_gcps(tempfile, gcps, dst_path=dst_path)
    os.remove(tempfile)


def apply_gcps_from_geolocation_array(
    arr: np.ndarray,
    loc: np.ndarray,
    dst_path: PathLike
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
        raise ValueError(f"arr of size {arr.shape} is not compatible with"
                         f"loc of size {loc.shape}. They must be the same"
                         "shape")

    prj_file = files("m3py.selenography.data").joinpath("cgs_moon_2000.prj")
    with prj_file.open() as f:
        crs = CRS.from_wkt(f.read())

    gcps = []
    for n, i in enumerate(range(0, loc.shape[0])):
        gcps.append(GroundControlPoint(
            row=i, col=0, x=loc[i, 0, 0], y=loc[i, 0, 1]
        ))
        gcps.append(GroundControlPoint(
            row=i, col=loc.shape[1]-1, x=loc[i, -1, 0], y=loc[i, -1, 1]
        ))

    tempfile = numpy_to_gtiff(arr, crs)

    warp_to_gcps(tempfile, gcps, dst_path=dst_path)

    os.remove(tempfile)


def apply_gcps(
    arr: np.ndarray,
    gcp_pointer: np.ndarray | PathLike,
    dst_path: PathLike
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
    """
    if isinstance(gcp_pointer, np.ndarray):
        apply_gcps_from_geolocation_array(arr, gcp_pointer, dst_path)
    elif isinstance(gcp_pointer, PathLike):
        apply_gcps_from_file(arr, gcp_pointer, dst_path)
    else:
        raise TypeError(f"{type(gcp_pointer)} is an invalid type for a gcp"
                        "pointer.")


def warp_to_gcps(
    src_path: PathLike,
    gcps: List[GroundControlPoint],
    dst_path: Optional[PathLike] = None,
    gdal_conda_env_name: str = "gdal"
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
    tempfile = Path(tempfile.name)

    gcp_list_as_strings = [f"{i.col} {i.row} {i.x} {i.y}" for i in gcps]
    gdal_translate = "gdal_translate -gcp " \
                     f"{' -gcp '.join(gcp_list_as_strings)} "\
                     f"{src_path} {tempfile}"

    gdal_warp = "gdalwarp -r near -tps -t_srs " \
                f"{prj_file} {tempfile} {dst_path}"

    def run_command_in_conda_env(env_name: str, command_str: str):
        out = subprocess.run(
            f"conda run -n {env_name} {command_str}".split(),
            shell=True, capture_output=True
        )
        print(command_str)
        print(f"----STDOUT----\n{out.stdout.decode("utf-8")}")
        print(f"----STDERR----\n{out.stderr.decode("utf-8")}")

    run_command_in_conda_env(gdal_conda_env_name, gdal_translate)
    run_command_in_conda_env(gdal_conda_env_name, gdal_warp)
