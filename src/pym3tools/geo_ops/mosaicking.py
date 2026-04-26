from collections.abc import Callable
from typing import Concatenate
from pathlib import Path

import numpy as np

from pym3tools.types import MosaicMethod, CubioTuple
from .uniform_gridding import resample_to_super_grid
from .data_transfer_classes import ResampledMosaic


def key_reducer(stack: np.ndarray, key: np.ndarray) -> np.ndarray:
    return np.take_along_axis(stack, key, axis=3)[..., 0]


def min_incidence_mosaic(
    stack: np.ndarray,
    inc_ang_list: list[np.ndarray],
) -> np.ndarray:
    inc_stack = np.stack(inc_ang_list, axis=-1)
    inc_stack[np.isnan(inc_stack)] = 999
    key = np.argmin(inc_stack, axis=-1, keepdims=True)[..., None]
    return key_reducer(stack, key)


def max_albedo_mosaic(stack: np.ndarray, *_) -> np.ndarray:
    albedo_stack = np.mean(stack, axis=2, keepdims=True)
    albedo_stack[np.isnan(albedo_stack)] = -999
    key = np.argmax(albedo_stack, axis=-1)[..., None]
    return key_reducer(stack, key)


def mean_mosaic(stack: np.ndarray, *_):
    return np.nanmean(stack, axis=-1)


MOSAIC_DISPATCHER: dict[
    MosaicMethod, Callable[Concatenate[np.ndarray, ...], np.ndarray]
] = {
    "Mean": mean_mosaic,
    "MaxAlbedo": max_albedo_mosaic,
    "MinimumIncidenceAngle": min_incidence_mosaic,
}


def mosaic_from_aligned_grids(
    aligned_grid_list: list[np.ndarray],
    method: MosaicMethod,
    incidence_angle_list: list[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Input arrays should be in the BIP (Lat, Long, Bands) format. Incidence
    angle backplanes should be 2D.
    """
    stack = np.stack(aligned_grid_list, axis=-1)
    mosaic = MOSAIC_DISPATCHER[method](stack, incidence_angle_list)
    return mosaic


def mosaic_arrays(
    input_list: list[str] | list[Path] | list[CubioTuple],
    method: MosaicMethod,
    incidence_angle_list: (
        list[Path] | list[str] | list[CubioTuple] | None
    ) = None,
    photometry_cube: bool = False,
) -> ResampledMosaic:
    aligned_list = resample_to_super_grid(input_list)
    if incidence_angle_list is not None:
        if photometry_cube:
            aligned_photo = resample_to_super_grid(incidence_angle_list)
            aligned_inc = [i[:, :, 0] for i in aligned_photo.data]
        else:
            aligned_inc = resample_to_super_grid(incidence_angle_list).data
    else:
        aligned_inc = None

    mosaic = mosaic_from_aligned_grids(aligned_list.data, method, aligned_inc)

    return ResampledMosaic(
        mosaic, aligned_list.gtrans, aligned_list.bounds, aligned_list.crs
    )
