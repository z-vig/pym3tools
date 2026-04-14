from collections.abc import Callable
import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pym3tools.data_retrieval.data_directory import M3DataPaths
from pym3tools.rdn2rfl.pipeline_state import (
    PipelineState,
    CompletedFlag,
    get_standard_dset_attrs,
)
from pym3tools.save_models.pipeline_cache_schema import PipelineCache
from pym3tools.types import TopoCorrectionMethod


def XL(i: float, e: float, _g: float):
    """
    Lommel-Seeliger Photometric Limb Darkening Factor.
    """
    d2r = np.pi / 180  # Degrees to Radians
    return np.cos(i * d2r) / (np.cos(e * d2r) + np.cos(i * d2r))


def LL(i: float, e: float, g: float):
    """
    Lunar Lambert limb-darkening polynomial factor.
    """
    d2r = np.pi / 180  # Degrees to Radians
    A = -0.019
    B = 0.242 * 10**-3
    C = -1.46 * 10**-6

    def L(g):
        return 1 + A * g + B * g**2 + C * g**3

    return L(g) * XL(i, e, g) + (1 - L(g)) * np.cos(i * d2r)


def compute_limb_darkening(
    photometry_cube: np.ndarray,
    normalized_geometry: tuple[int, int, int] = (30, 0, 30),
    method: TopoCorrectionMethod = "Lommel-Seeliger",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the limb darkening correction factors for each pixel in a data array.

    Parameters
    ----------
    photometry_cube: np.ndarray
        Observation geometry data. The first three slices in the third axis
        must be: {incidence angle, emission_angle, phase_angle}
    normalized_geometry: Tuple of integers, optional
        Geometry to normalize the limb-darkening factor to. Default is (30, 0,
        30), signifying 30 degree incidence, 0 degree emission, 30 degree
        phase.
    method: TopoCorrectionMethod, optional
        Method of limb-darkening correction. Either "Lommel-Seeliger" (default)
        or "Lunar Lambert".

    Returns
    -------
    ldf: np.ndarray
        Limb-Darkening Factors
    ldf_norm: np.ndarray
        Limb-darkening factors at the normalized geometry. Must divide this by
        `ldf` to normalize factors.
    """
    # Limb Darkening Function dispatcher
    ldf_dispatcher: dict[TopoCorrectionMethod, Callable] = {
        "Lommel-Seeliger": XL,
        "Lunar Lambert": LL,
    }

    incidence_angle = photometry_cube[:, :, 0]
    emission_angle = photometry_cube[:, :, 1]
    phase_angle = photometry_cube[:, :, 2]

    incidence_angle[incidence_angle > 87] = 87

    limb_darkening_factor = ldf_dispatcher[method](
        incidence_angle, emission_angle, phase_angle
    )

    ldf_norm = ldf_dispatcher[method](*normalized_geometry)

    return limb_darkening_factor, ldf_norm


def get_phase_function_rgi(catalog: M3DataPaths) -> RegularGridInterpolator:
    pattern = re.compile(r"\s\d.\d{9}")
    with open(catalog.photometry_correction.tab) as f:
        phase_function_lookup = np.array(
            [re.findall(pattern, i) for i in f.readlines()[1:]],
            dtype=np.float32,
        )
        phase_function_lookup = phase_function_lookup[:101, :]
    x = np.arange(phase_function_lookup.shape[0])
    y = np.arange(phase_function_lookup.shape[1])
    return RegularGridInterpolator((x, y), phase_function_lookup)


def compute_f_alpha(
    phase_angle_array: np.ndarray,
    f_alpha_rgi: RegularGridInterpolator,
    spectrum_size: int,
    normalized_geometry: tuple[int, int, int] = (30, 0, 30),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes both f_alpha for a given phase angle by interpolating from the
    PDS-provided lookup table.

    Parameters
    ----------
    phase_angle_array: np.ndarray
        Phase angle map
    f_alpha_rgi: RegularGridInterpolator
        Interpolator object derived from PDS Lookup Table.
    spectrum_size: int
        Number of channels.
    normalized_geometry: Tuple of integers
        Geometry to normalize the phase function to. Default is (30, 0, 30),
        signifying 30 degree incidence, 0 degree emission, 30 degree phase.

    Returns
    -------
    f_alpha: np.ndarray
        Cube of wavelength-dependent phase function factors
    f_alpha_norm: np.ndarray
        Cube of phase function factors at the normalization geometry. Must
        divide this by `f_alpha` to normalize factors.
    """
    phase_angle_array[phase_angle_array > 100] = 100

    flat = phase_angle_array.ravel()
    mask = ~np.isnan(flat)
    valid_values = flat[mask]

    tiled_phase_array = np.stack(
        [
            np.repeat(valid_values, spectrum_size),
            np.tile(np.arange(0, spectrum_size), valid_values.size),
        ],
        axis=-1,
    )

    f_alpha = f_alpha_rgi(tiled_phase_array)

    normalization_factor = f_alpha_rgi(
        np.stack(
            [
                normalized_geometry[-1] * np.ones(spectrum_size),
                np.arange(0, spectrum_size),
            ],
            axis=-1,
        )
    )

    f_alpha = np.reshape(f_alpha, (valid_values.size, spectrum_size))
    f_alpha_full = np.full((flat.size, spectrum_size), np.nan)
    f_alpha_full[mask, :] = f_alpha
    f_alpha_full = np.reshape(
        f_alpha_full, (*phase_angle_array.shape, spectrum_size)
    )
    f_alpha_norm = normalization_factor[None, None, :]
    return f_alpha_full, f_alpha_norm


def cosine_correction(i: np.ndarray) -> np.ndarray:
    """
    Gets the factors of a simple lambert cosine correction. Multiply these
    factors over a cube to correct for Lambert Topography as in the Clark et
    al., 2011 Thermal Correction.

    Parameters
    ----------
    i: NDArray
        Incidence Angle Map.

    Returns
    -------
    lambert_factors: NDArray
        1/cosine(i) factor map.
    """
    return 1 / np.cos(i)


def multiply_by_photometric_coefficients(
    data: np.ndarray, coefs: np.ndarray
) -> np.ndarray:
    if coefs.shape != data.shape:
        raise ValueError(
            f"Invalid dimensions for coefficients ({coefs.shape})"
        )
    return data * coefs


def get_photometric_coefficients(
    photo_cube: np.ndarray,
    limb_darkening_method: TopoCorrectionMethod,
    phase_function_rgi: RegularGridInterpolator,
    wvl: list[float],
):
    ldf, ldf_norm = compute_limb_darkening(
        photo_cube, method=limb_darkening_method
    )
    nwvl = len(wvl)
    f_alpha, f_alpha_norm = compute_f_alpha(
        photo_cube[:, :, -1], phase_function_rgi, nwvl
    )
    photometric_coefficients = (ldf_norm / ldf)[:, :, None] * (
        f_alpha_norm / f_alpha
    )
    return photometric_coefficients


def modify_state(
    state: PipelineState, photometric_coefficients: np.ndarray
) -> PipelineState:
    state.data = multiply_by_photometric_coefficients(
        np.array(state.data), photometric_coefficients
    )
    state.flags |= CompletedFlag.PHOTO_CORR_APPLIED
    return state


def write_to_cache(
    cache: PipelineCache,
    output: PipelineState,
    timestamp: str,
    photo_coefs: np.ndarray,
    topo_correction: TopoCorrectionMethod,
) -> None:
    cubeattrs = get_standard_dset_attrs(timestamp, output)
    coefattrs = get_standard_dset_attrs(timestamp, output)
    bckplnattrs = get_standard_dset_attrs(timestamp, output)

    bckplnattrs.update(
        {"bandlbls": ["i", "e", "g"], "nbands": 3, "measvals": [0.0, 1.0, 2.0]}
    )

    cache.photometric_corrected.cube.write(
        np.array(output.data, dtype=np.float32)
    )
    cache.photometric_corrected.cube.set_attrs(cubeattrs)

    cache.photometric_corrected.photometric_coefficients.write(photo_coefs)
    cache.photometric_corrected.photometric_coefficients.set_attrs(coefattrs)

    cache.photometric_corrected.photometry_backplane.write(
        output.obs.photometry_cube
    )
    cache.photometric_corrected.photometry_backplane.set_attrs(bckplnattrs)

    cache.photometric_corrected.set_attrs(
        {
            "flags": output.flags,
            "timestamp": timestamp,
            "topography_correction": topo_correction,
        }
    )
