"""
Utilities for converting targeted mode M3 data to global resolution.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

# Global resampling slices. See Green et al., 2011, paragraph 21.
# A key change was made to process from L1, instead of L0. The first four
# bands were removed, so the values in Green et al., 2011 were offset by four.
GLOBAL_RESAMPLING: dict[slice, int] = {
    slice(0, 28): 4,
    slice(28, 112): 2,
    slice(112, 256): 4,
}

# Number of global M3 channels in L1B data.
N_GLOBAL_CHANNELS: int = 85


def dynamic_slice(a: np.ndarray, axis: int, slc: slice) -> np.ndarray:
    """
    Fast, dynamic slicing of numpy arrays along a specified axis.
    See: https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis
    """
    return a[(slice(None),) * (axis % a.ndim) + (slc,)]


def dynamic_index(
    a: np.ndarray, axis: int, idx: int
) -> tuple[slice | int, ...]:
    """Used for dynamic axis array indexing."""
    slc: list[slice | int] = [slice(None)] * a.ndim
    slc[axis] = idx
    return tuple(slc)


def filter_cube_slice(
    cube: np.ndarray,
    z_axis: int,
    z_slice: slice,
    filter_width: int,
    add_offset: bool = True,
) -> np.ndarray:
    """
    Filters one slice of a cube at a specific filter width.

    Parameters
    ----------
    cube: np.ndarray
        Cube to be filtered.
    z_axis: int
        The location of the z axis in the cube.
    z_slice: slice
        Slice over which to apply the uniform z-axis filter
    add_offset: bool, optional
        Toggles adding an rightward offset to the filter. Having this offset
        ensures that only real bands are being average. Default is True.

    Returns
    -------
    np.ndarray
        Filtered numpy array.
    """
    piece = dynamic_slice(cube, z_axis, z_slice)  # fast slicing of cube
    filter_offset = -filter_width // 2
    if not add_offset:
        filter_offset = 0
    filtered_arr = uniform_filter1d(
        piece,
        filter_width,
        axis=z_axis,
        mode="nearest",
        origin=filter_offset,
    )
    return filtered_arr


def downsample_cube(cube: np.ndarray, z_axis: int, resamp_factor: int):
    """Downsamples a cube by a resampling factor."""
    resamp_slice = slice(0, cube.shape[z_axis], resamp_factor)
    return dynamic_slice(cube, z_axis, resamp_slice)


def slice_target_to_global(
    targeted_slice: slice, resamp_factor: int, global_position: int
) -> tuple[slice, int]:
    """
    Maps a targeted z-axis slice to a global z-axis slice.

    Parameters
    ----------
    targeted_slice: slice
        Slice on the targeted z-axis.
    resamp_factor: int
        Factor that is used to downsample the cube slice.
    global_position: int
        The starting index on the global cube.

    Returns
    -------
    slice
        Targeted z-axis slice.
    int
        Length of the targeted slice. Used to advance global index position.
    """
    length = (targeted_slice.stop - targeted_slice.start) // resamp_factor
    global_slice = slice(global_position, global_position + length)
    return global_slice, length


def targeted_to_global_spectral(
    cube: np.ndarray,
    v_axis: int = 0,
    h_axis: int = 1,
    z_axis: int = 2,
    replace_last_targeted_band: bool = True,
) -> np.ndarray:
    """
    Resamples a targeted M3 cube to global spectral resolution. The spatial
    resolution remains the same.

    Parameters
    ----------
    cube: np.ndarray
        Targeted reflectance cube.
    v_axis: int
        Position of the vertical data axis.
    h_axis: int
        Position of the horizontal data axis.
    z_axis: int
        Position of the spectral data axis.
    replace_last_targeted_band: bool, optional
        Toggles the replacement of the last band in targeted mode, which is a
        bad band will affect the resulting global mode data.
    """

    # ==== Initializing empty global array. ====
    global_array = np.empty(
        (cube.shape[v_axis], cube.shape[h_axis], N_GLOBAL_CHANNELS),
        dtype=np.float32,
    )

    global_position = 0  # Tracking z-axis position

    if replace_last_targeted_band:
        last_band = dynamic_index(cube, z_axis, -1)
        second_to_last_band = dynamic_index(cube, z_axis, -2)
        cube[last_band] = cube[second_to_last_band]

    # ==== Looping over global resampling slices ====
    for slc, resamp in GLOBAL_RESAMPLING.items():
        # Filtering the raw cube
        filtered_arr = filter_cube_slice(cube, z_axis, slc, resamp)

        # Downsampling filtered cube to global resolution
        downsampled_arr = downsample_cube(filtered_arr, z_axis, resamp)

        # Finding the position of the data in the global cube
        glb_slc, lngth = slice_target_to_global(slc, resamp, global_position)
        global_array[:, :, glb_slc] = downsampled_arr

        # Advancing global index
        global_position += lngth

    return global_array
