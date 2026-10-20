import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from pym3tools.rdn2rfl.pipeline_state import (
    CompletedFlag,
    PipelineState,
    get_standard_dset_attrs,
)
from pym3tools.save_models.pipeline_cache_schema import PipelineCache
from pym3tools.types import DestripingMethod


def column_mean_correction(arr: np.ndarray | xr.DataArray) -> np.ndarray:
    """
    Subtracts correction factors from columns so that the means of all the
    columns are equal. Array must be in BIP format.
    """

    col_means = np.nanmean(arr, axis=(0, 2))
    correction_offsets = col_means - col_means.mean()

    corr_arr = arr - correction_offsets[None, :, None]

    return corr_arr


def gaussian_column_filter(arr: np.ndarray | xr.DataArray) -> np.ndarray:
    """Applies a gaussian filter along the column axis."""
    return gaussian_filter1d(arr, 1, axis=1)


def modify_state(
    state: PipelineState, method: DestripingMethod
) -> PipelineState:
    if method == "UniformColumnMean":
        state.data = column_mean_correction(state.data)
    if method == "GaussianFilter":
        state.data = gaussian_column_filter(state.data)
    state.flags |= CompletedFlag.DESTRIPED
    return state


def write_to_cache(
    timestamp: str,
    output: PipelineState,
    cache: PipelineCache,
    method: DestripingMethod,
) -> None:
    cube_attrs = get_standard_dset_attrs(timestamp, output)

    # ==== Saving Cube ====
    cache.destriped.cube.write(np.array(output.data, dtype=np.float32))
    cache.destriped.cube.set_attrs(cube_attrs)

    # ==== Setting Step Attributes ====
    cache.destriped.set_attrs(
        {"timestamp": timestamp, "flags": int(output.flags), "method": method}
    )
