from collections.abc import Callable
from pathlib import Path
from typing import Concatenate

import cubio as cb
import numpy as np

from pym3tools.types import CubioTuple, MosaicMethod

from .data_transfer_classes import ResampledMosaic
from .uniform_gridding import resample_to_super_grid


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
    incidence_angle_list: (list[Path] | list[str] | list[CubioTuple] | None) = None,
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


def mosaic_from_filepaths(
    photo_fp_list: list[Path],
    rfl_fp_list: list[Path],
    mosaic_name: str,
    output_directory: Path,
) -> None:
    """
    Creates and saves mosaics for M3 image files.

    Parameters
    ----------
    photo_fp_list: list[Path]
        List of filepaths to photometry backplane images
    rfl_fp_list: list[Path]
        List of filepaths to reflectance cubes, same ordering as photo_fp_list
    mosaic_name: str
        Name of the output mosaic
    output_directory: Path
        Directory location to save the mosaic
    """
    if not output_directory.is_dir():
        output_directory.mkdir(parents=True, exist_ok=True)

    rfl_mosaic = mosaic_arrays(rfl_fp_list, "Mean", photo_fp_list, photometry_cube=True)
    inc_mosaic = mosaic_arrays(photo_fp_list, "Mean")

    mosaic_dict: dict[str, tuple[ResampledMosaic, Path]] = {
        f"{mosaic_name}_RFL": (rfl_mosaic, rfl_fp_list[0]),
        f"{mosaic_name}_INC": (inc_mosaic, photo_fp_list[0]),
    }

    for name, (mosaic, sample) in mosaic_dict.items():
        sample_cc, _ = cb.cube_from_json(sample)
        mosaic_cc, mosaic_cd = cb.cube_from_numpy(
            mosaic.data,
            "BIP",
            name=name,
            crs=mosaic.crs,
            gtrans=mosaic.gtrans,
            measvals=sample_cc.measurement_values,
            bandlbls=sample_cc.band_names,
        )

        cb.write_envi(mosaic_cc, mosaic_cd, "BIL", output_directory)
