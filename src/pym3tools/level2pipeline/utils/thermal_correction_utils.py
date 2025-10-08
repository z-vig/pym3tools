# Standard Libraries
from typing import Tuple, Final
from dataclasses import dataclass, field, MISSING

# Dependencies
import numpy as np

# Some constants
h: Final[float] = 6.626 * 10**-34  # J*s, planck's constant
k_b: Final[float] = 1.381 * 10**-23  # J/K, boltzmann's constant
c: Final[float] = 2.998 * 10**8  # m/s, speed of lig


# Data Organization Classes
@dataclass
class RefWvl:
    target: int
    index: int
    actual: float


@dataclass
class RefWvlSet:
    A: RefWvl = field(default_factory=lambda: RefWvl(1550, 0, 0))
    B: RefWvl = field(default_factory=lambda: RefWvl(2350, 0, 0))
    C: RefWvl = field(default_factory=lambda: RefWvl(2700, 0, 0))
    D: RefWvl = field(default_factory=lambda: RefWvl(2280, 0, 0))
    E: RefWvl = field(default_factory=lambda: RefWvl(2590, 0, 0))

    @classmethod
    def from_data(cls, wvls: np.ndarray) -> "RefWvlSet":
        initializing_dict: dict[str, RefWvl] = {}
        for k, v in cls.__dataclass_fields__.items():
            if v.default_factory is not MISSING:
                target_wvl = v.default_factory().target
            else:
                raise ValueError("Error initializing RefWvlSet")
            wvl_index, actual_wvl = find_wvl(wvls, target_wvl)
            initializing_dict[k] = RefWvl(target_wvl, wvl_index, actual_wvl)
        return cls(**initializing_dict)


# Helper Functions
def find_wvl(wvls: np.ndarray, targetwvl: float) -> Tuple[int, float]:
    """
        findλ(λ.targetλ)

    Given a list of wavelengths, `wvls`, find the index of a `targetwvl` and
    the actual wavelength closest to your target.

    Parameters
    ----------
    wvls: np.ndarray
        Wavelength array to search in.
    targetwvl:
        Wavelength to search for.

    Returns
    -------
    idx: int
        Index of the found wavelength.
    wvl: float
        Actual wavelength that is closest to the target wavelength (at idx).
    """

    idx = np.argmin(np.abs(wvls - targetwvl))
    return int(idx), wvls[idx]


def linear_projection(
    data: np.ndarray, refwvl: RefWvlSet, initial: bool
) -> np.ndarray:
    if initial:
        y_proj = (
            (
                (data[:, :, refwvl.B.index] - data[:, :, refwvl.A.index])
                / (refwvl.B.target - refwvl.A.target)
            )
            * (refwvl.C.target - refwvl.A.target)
        ) + data[:, :, refwvl.A.index]
    else:
        y_proj = (
            (
                (data[:, :, refwvl.E.index] - data[:, :, refwvl.D.index])
                / (refwvl.E.actual - refwvl.D.actual)
            )
            * (refwvl.C.actual - refwvl.D.actual)
        ) + data[:, :, refwvl.D.index]
    return y_proj


# Calculation Functions
def get_temp(B: np.ndarray, e, w: float, F: np.ndarray):
    """
    Gets the temperature given a spectral thermal component.

    Parameters
    ----------
    B: Thermal component.
    e: Emissivity (constant)
    w: wavelength of calculation
    F: solar spectrum
    """
    return (h * c / (w * k_b)) * (
        np.log(((2 * h * c**2 * e) / ((B * 10**6 * F / np.pi) * w**5)) + 1)
    ) ** -1


def get_temp_photometric(B, e, w, F, phi):
    return (h * c / (w * k_b)) * (
        np.log(
            ((2 * h * c**2 * e) / ((F * B * 10**6 / (phi * np.pi)) * w**5)) + 1
        )
    ) ** -1


def get_thermal_spectrum(wvl, temp, e, solar_spec, solar_dist):
    B = ((2 * h * c**2) / (wvl**5)) * (
        1 / (np.exp((h * c) / (wvl * k_b * temp)) - 1)
    )
    F = solar_spec
    therm_spec = (solar_dist**2 * e * B * 10**-6 * np.pi) / F
    return therm_spec
