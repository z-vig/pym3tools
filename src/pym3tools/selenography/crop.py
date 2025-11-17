# Standarad Libraries
from typing import Tuple, Optional, Literal

# Dependencies
import numpy as np
import rasterio as rio  # type: ignore
from rasterio.coords import BoundingBox  # type: ignore
from rasterio.crs import CRS  # type: ignore
from affine import Affine  # type: ignore

# Top-Level Imports
from pym3tools.types import PathLike, Path


def find_furthest_idx(array: np.ndarray, value: np.intp) -> int:
    """Finds fursthest index away from value."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmax()
    return int(array[idx])


def polar_crop(
    target_arr: np.ndarray,
    loc_arr: np.ndarray,
    cropping_type: str = "equitorial",
) -> np.ndarray:
    """
    Either crops an image arr to exclude the poles or include only the north
    or south pole as defined by +/-70 degrees latitude.
    """
    no_poles = np.where((loc_arr[:, :, 1] > 70) | (loc_arr[:, :, 1] < -70))

    bool_image = np.zeros((loc_arr.shape[:2]))
    bool_image[*no_poles] = 1

    profile = np.max(bool_image, axis=1)
    diff = np.abs(np.diff(profile))
    if np.count_nonzero(diff != 0) == 1:
        idx1 = np.argmax(diff)
        idx2 = find_furthest_idx(np.array([0, loc_arr.shape[0]]), idx1)
    else:
        idx1, idx2 = np.argsort(diff)[:-2]

    equitorial_idx = np.arange(
        min(int(idx1), int(idx2)), max(int(idx1), int(idx2))
    )
    if min(int(idx1), int(idx2)) != 0:
        north_polar_idx = np.arange(0, min(int(idx1), int(idx2)))
    else:
        north_polar_idx = None

    if max(int(idx1), int(idx2)) != loc_arr.shape[0]:
        south_polar_idx = np.arange(
            max(int(idx1), int(idx2)), loc_arr.shape[0]
        )
    else:
        south_polar_idx = None

    if cropping_type == "equitorial":
        return target_arr[equitorial_idx, ...]
    elif cropping_type == "north":
        if north_polar_idx is not None:
            return target_arr[north_polar_idx, ...]
        else:
            raise IndexError("This image does not contain North Polar data.")
    elif cropping_type == "south":
        if south_polar_idx is not None:
            return target_arr[south_polar_idx, ...]
        else:
            raise IndexError("This image does not contain South Polar Data")
    else:
        raise ValueError(f"{cropping_type} is not a valid cropping type.")


def regional_crop(
    arr: np.ndarray, loc_arr: np.ndarray, bbox: BoundingBox
) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Gets the offsets, width and height associated with a bounding box given
    a geolocation backplane. This window is then applied to the input array.
    This function cannot yet be used for polar regions.

    Parameters
    ----------
    arr: np.ndarray
        Array to be cropped.
    loc_arr: np.ndarray
        Geolocation array in the format [0:X, 0:Y, {Longitude, Latitude,
        Elevation}]
    bbox: BoundingBox
        Rasterio BoundingBox object.

    Returns
    -------
    cropped_arr: np.ndarray
        Inout array cropped to the BoundingBox.
    row_offset: int
        Starting row of the dataset in relation to whole M3 stripe.
    col_offset: int
        Starting column of the dataset in relation to whole M3 stripe.
    height: int
        Height of the cropped_arr.
    width:
        Width of the cropped_arr.
    """

    longitude_condition = (loc_arr[:, :, 0] - 360 > bbox.left) & (
        loc_arr[:, :, 0] - 360 < bbox.right
    )

    latitude_condition = (loc_arr[:, :, 1] > bbox.bottom) & (
        loc_arr[:, :, 1] < bbox.top
    )

    bool_loc = np.zeros(loc_arr.shape[:2])
    bool_loc[longitude_condition & latitude_condition] = 1

    rows, cols = np.where(bool_loc == 1)

    row_offset = rows.min()
    height = (rows.max() - rows.min()) + 1

    col_offset = cols.min()
    width = (cols.max() - cols.min()) + 1

    cropped_arr = arr[
        row_offset : row_offset + height,  # noqa
        col_offset : col_offset + width,  # noqa
        ...,
    ]

    return cropped_arr, row_offset, col_offset, height, width


def general_crop(
    data_fp: PathLike,
    bbox: BoundingBox,
    save_name: str,
    save_mode: Literal["ENVI"] | Literal["GTIFF"],
    arr: Optional[np.ndarray] = None,
    geotransform: Optional[
        tuple[float, float, float, float, float, float]
    ] = None,
    crs: Optional[CRS] = None,
):
    if arr is not None:
        raise NotImplementedError(
            "Cropping of a numpy array is not implemented yet. Please provide"
            " a path to a georeferenced file."
        )

    if save_mode == "ENVI":
        driver = "ENVI"
        save_path = Path(Path(data_fp).parent, save_name).with_suffix(".bsq")
    elif save_mode == "GTIFF":
        driver = "GTiff"
        save_path = Path(Path(data_fp).parent, save_name).with_suffix(".tif")
    with rio.open(data_fp, "r") as src:
        w = src.window(*bbox)
        print(src.shape)
        print(w.col_off)

        _old = src.transform
        left = _old.c + _old.a * w.col_off
        top = _old.f + _old.e * w.row_off
        new_transform = Affine(_old.a, _old.b, left, _old.d, _old.e, top)

        arr = src.read(window=w)
        if arr is None:
            return
        prfl = src.profile
        # Update the profile so the written file has correct size and
        # georeference
        prfl.update(
            {
                "driver": driver,
                "height": arr.shape[1],
                "width": arr.shape[2],
                "transform": new_transform,
            }
        )

        with rio.open(save_path, "w", **prfl) as dst:
            dst.write(arr * 1000)

    return arr
