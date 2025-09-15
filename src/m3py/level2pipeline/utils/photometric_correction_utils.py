# Standard Libraries
from typing import Callable, Tuple

# Dependencies
import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator


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
    obs_geom: np.ndarray,
    normalized_geometry: Tuple[int, int, int] = (30, 0, 30),
    method: str = "Lommel-Seeliger",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the limb darkening correction factors for each pixel in a data array.

    Parameters
    ----------
    obs_geom: np.ndarray
        Observation geometry data. The first three slices in the third axis
        must be: {incidence angle, emission_angle, phase_angle}
    normalized_geometry: Tuple of integers, optional
        Geometry to normalize the limb-darkening factor to. Default is (30, 0,
        30), signifying 30 degree incidence, 0 degree emission, 30 degree
        phase.
    method: str, optional
        Method of limb-darkening correction. Either "Lommel-Seeliger" (default)
        or "Lunar_Lambert".

    Returns
    -------
    ldf: np.ndarray
        Limb-Darkening Factors
    ldf_norm: np.ndarray
        Limb-darkening factors at the normalized geometry. Must divide this by
        `ldf` to normalize factors.
    """
    # Limb Darkening Function dispatcher
    ldf_dispatcher: dict[str, Callable] = {
        "Lommel-Seeliger": XL,
        "Lunar_Lambert": LL,
    }

    incidence_angle = obs_geom[:, :, 0]
    emission_angle = obs_geom[:, :, 1]
    phase_angle = obs_geom[:, :, 2]

    incidence_angle[incidence_angle > 87] = 87

    limb_darkening_factor = ldf_dispatcher[method](
        incidence_angle, emission_angle, phase_angle
    )

    ldf_norm = ldf_dispatcher[method](*normalized_geometry)

    return limb_darkening_factor, ldf_norm


def compute_f_alpha(
    phase_angle_array: np.ndarray,
    f_alpha_rgi: RegularGridInterpolator,
    spectrum_size: int,
    normalized_geometry: Tuple[int, int, int] = (30, 0, 30),
) -> Tuple[np.ndarray, np.ndarray]:
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

    f_alpha = f_alpha_rgi(
        np.stack(
            [
                np.repeat(valid_values, spectrum_size),
                np.tile(np.arange(0, spectrum_size), valid_values.size),
            ],
            axis=-1,
        )
    )

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


def cosine_correction(i: npt.NDArray) -> npt.NDArray:
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
