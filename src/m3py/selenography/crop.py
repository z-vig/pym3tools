# Standarad Libraries

# Dependencies
import numpy as np
from rasterio.coords import BoundingBox


def find_furthest_idx(
    array: np.ndarray,
    value: np.intp
) -> int:
    """Finds fursthest index away from value."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmax()
    return int(array[idx])


def polar_crop(
    target_arr: np.ndarray,
    loc_arr: np.ndarray,
    cropping_type: str = "equitorial"
) -> np.ndarray:
    """
    Either crops an image arr to exclude the poles or include only the north
    or south pole as defined by +/-70 degrees latitude.
    """
    no_poles = np.where(
        (loc_arr[:, :, 1] > 70) | (loc_arr[:, :, 1] < -70)
    )

    bool_image = np.zeros((loc_arr.shape[:2]))
    bool_image[*no_poles] = 1

    profile = np.max(bool_image, axis=1)
    diff = np.abs(np.diff(profile))
    if np.count_nonzero(diff != 0) == 1:
        idx1 = np.argmax(diff)
        idx2 = find_furthest_idx(np.array([0, loc_arr.shape[0]]), idx1)
    else:
        idx1, idx2 = np.argsort(diff)[:-2]

    equitorial_idx = np.arange(min(idx1, idx2), max(idx1, idx2))
    if min(idx1, idx2) != 0:
        north_polar_idx = np.arange(0, min(idx1, idx2))
    else:
        north_polar_idx = None

    if max(idx1, idx2) != loc_arr.shape[0]:
        south_polar_idx = np.arange(max(idx1, idx2), loc_arr.shape[0])
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
    target_arr: np.ndarray,
    loc_arr: np.ndarray,
    bbox: BoundingBox
) -> np.ndarray:
    longitude_profile = np.mean(loc_arr[:, :, 0], axis=0)
    latitude_profile = np.mean(loc_arr[:, :, 1], axis=1)

    rows_in_bbox = np.where(
        (longitude_profile > bbox.left) and longitude_profile < bbox.right
    )[0]

    cols_in_bbox = np.where(
        (latitude_profile > bbox.bottom) and (latitude_profile < bbox.top)
    )[0]

    print(cols_in_bbox)

    return rows_in_bbox
