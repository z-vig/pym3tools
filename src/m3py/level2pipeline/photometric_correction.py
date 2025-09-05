# Standard Libraries
from typing import Optional, Tuple, Callable

# Dependencies
import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type: ignore

# Relative Imports
from m3py.level2pipeline.step import PipelineState
from .step import Step


def XL(i: float, e: float, _g: float):
    """
    Lommel-Seeliger Photometric Limb Darkening Factor.
    """
    d2r = np.pi / 180  # Degrees to Radians
    return np.cos(i * d2r) / (np.cos(e * d2r) + np.cos(i * d2r))


def LL(i: float, e: float, g: float):
    d2r = np.pi / 180  # Degrees to Radians
    A = -0.019
    B = 0.242 * 10**-3
    C = -1.46 * 10**-6

    def L(g):
        return 1 + A * g + B * g**2 + C * g**3

    return L(g) * XL(i, e, g) + (1 - L(g)) * np.cos(i * d2r)


def compute_f_alpha(
    phase_angle_array: np.ndarray,
    f_alpha_rgi: RegularGridInterpolator,
    spectrum_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
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

    ############
    # Normalization factor being applied may be a mistake, as it is already
    # applied in the PDS release (maybe??)
    ############

    normalization_factor = f_alpha_rgi(
        np.stack(
            [30 * np.ones(spectrum_size), np.arange(0, spectrum_size)],
            axis=-1,
        )
    )

    f_alpha = np.reshape(f_alpha, (valid_values.size, spectrum_size))
    f_alpha_full = np.full((flat.size, spectrum_size), np.nan)
    f_alpha_full[mask, :] = f_alpha
    f_alpha_full = np.reshape(
        f_alpha_full, (*phase_angle_array.shape, spectrum_size)
    )
    f_alpha_norm = normalization_factor[None, None, :] / f_alpha_full
    return f_alpha_full, f_alpha_norm


def photometric_correction_single_spectrum(
    spectrum: np.ndarray,
    incidence_angle: float,
    emission_angle: float,
    phase_angle: float,
    f_alpha: np.ndarray,
    limb_darkening: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a photometric correction to a single spectrum.

    Parameters
    ----------
    spectrum: np.ndarray
        Spectrum to be corrected.
    incidence_angle: float
        Angle of incident sunlight to surface normal.
    emission_angle: float
        Angle from the surface normal to the detector.
    phase_angle: float
        Angle between incident sunlight and the detector.
    f_alpha: np.ndarray
        Normalized phase coefficients. If the phase function is gamma, this is
        equivalent to gamma(30) / gamma(g). Must be the same size as spectrum.
    limb_darkening: str, optional
        Either "Lommel-Seeliger" (default) or "Lunar_Lambert".
    """
    # Handling invalid limb-darkening argument.
    valid_limb_darkening_options = ["Lommel-Seeliger", "Lunar_Lambert"]
    if limb_darkening not in valid_limb_darkening_options:
        raise ValueError(
            f"{limb_darkening} is an invalid limb-darkening option."
        )

    if np.isfinite(incidence_angle):
        if limb_darkening == "Lommel-Seeliger":
            if np.cos(incidence_angle * (np.pi / 180)) < 0.05:
                LDF = XL(30, 0, 30) / XL(
                    np.acos(0.05), emission_angle, phase_angle
                )
            elif np.cos(incidence_angle * (np.pi / 180)) > 0.05:
                LDF = XL(30, 0, 30) / XL(
                    incidence_angle, emission_angle, phase_angle
                )
            else:
                raise ValueError(
                    f"{incidence_angle} is an invalid incidence angle value."
                )
        elif limb_darkening == "Lunar_Lambert":
            LDF = LL(30, 0, 30) / LL(
                incidence_angle, emission_angle, phase_angle
            )
        else:
            raise ValueError()  # Case handled above

        photometric_coefficients = LDF * f_alpha
    elif np.isnan(incidence_angle):
        photometric_coefficients = np.full(f_alpha.size, np.nan)
    else:
        raise ValueError(
            f"{incidence_angle} is an invalid incidence angle value."
        )

    corrected_spectrum = spectrum * photometric_coefficients

    return corrected_spectrum, photometric_coefficients


def compute_limb_darkening_correction_factor(
    obs_geom: np.ndarray, method: str = "Lommel-Seeliger"
) -> np.ndarray:
    """
    Finds the limb darkening correction factors for each pixel in a data array.

    Parameters
    ----------
    obs_geom: np.ndarray
        Observation geometry data. The first three slices in the third axis
        must be: {incidence angle, emission_angle, phase_angle}
    method: str
        Method of limb-darkening correction. Either "Lommel-Seeliger" (default)
        or "Lunar_Lambert".
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

    limb_darkening_normalization = ldf_dispatcher[method](30, 0, 30)

    return limb_darkening_normalization / limb_darkening_factor


class PhotometricCorrection(Step):
    def run(self, state: PipelineState) -> PipelineState:
        return super().run(state)
