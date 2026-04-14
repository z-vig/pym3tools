"""
Clark Thermal Correction
---
Functions for implementing the Clark et al., 2011 thermal correction. This
method is an iterative correction that estimates the thermal component of the
spectrum by interpolating a straight line between two reference wavelengths.
The thermal component is then removed and the striaght line method is repeated
until a sufficient amount of thermal emission is removed. The method is
outlined in the following publication:

Clark, R. N., Pieters, C. M., Green, R. O., Boardman, J. W., & Petro, N. E.
(2011). Thermal removal from near-infrared imaging spectroscopy data of the
Moon. Journal of Geophysical Research, 116.
https://doi.org/10.1029/2010je003751
"""

from dataclasses import dataclass, field, MISSING

import numpy as np

from pym3tools2.constants import h, k_b, c
from pym3tools2.data_retrieval.data_directory import M3DataPaths
from pym3tools2.rdn2rfl.pipeline_services.solar_removal import (
    retrieve_solar_data,
)
from pym3tools2.rdn2rfl.pipeline_state import PipelineState
from pym3tools2.save_models.pipeline_cache_schema import PipelineCache
from .base_thermal_correction import BaseCorrection


def find_wvl(wvls: np.ndarray, targetwvl: float) -> tuple[int, float]:
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


# ==== Data Organization Classes ====
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


# ==== Calculation Functions =====
def linear_projection(
    data: np.ndarray, refwvl: RefWvlSet, initial: bool
) -> np.ndarray:
    """
    Returns the linear projection between two reference wavelengths at a third
    reference wavelength.
    """
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


def get_temp(B: np.ndarray, e: np.ndarray, w: float, F: np.ndarray):
    """
    Gets the temperature given a spectral thermal component.

    Parameters
    ----------
    B: Thermal component
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


def initial_thermal_estimate(
    catalog: M3DataPaths, data: np.ndarray, wvl: np.ndarray, refset: RefWvlSet
):
    """Get the initial thermal estimate before starting iterative removal."""
    # Getting solar spectrum data
    sol = retrieve_solar_data(catalog)

    # Unscaling data by solar distance
    data = data * sol.solar_distance**2

    # Linear Projection between A and B to C
    projection = linear_projection(data, refset, initial=True)

    # Subtracting projection from actual data to get thermal component
    thermal_component = data[:, :, refset.C.index] - projection
    temp_undefined = thermal_component < 0  # Undefined correction mask
    thermal_component[temp_undefined] = np.nan  # Applying mask

    emiss = 1 - data[:, :, refset.A.index]  # Constant Emissivity estimate

    # Finding what index to pull solar radiance from
    Fidx = np.argmin(np.abs(sol.solar_wvl - refset.C.actual))

    initial_temp = get_temp(
        thermal_component,
        emiss,
        refset.C.actual * 10**-9,
        sol.solar_spectrum[Fidx],
    )

    initial_blackbody = get_thermal_spectrum(
        wvl[None, None, :] * 10**-9,
        initial_temp[:, :, None],
        emiss,
        sol.solar_spectrum[None, None, :],
        sol.solar_distance,
    )

    initial_thermal_removed = data - initial_blackbody

    return initial_thermal_removed, initial_temp


def iterative_step():
    """One step in the iterative thermal correction."""
    pass


class Clark(BaseCorrection):
    def __init__(self) -> None:
        super().__init__()

    def modify_state(
        self, state: PipelineState, catalog: M3DataPaths
    ) -> tuple[PipelineState, np.ndarray]:
        raise NotImplementedError("TBW")

    def write_to_cache(
        self,
        cache: PipelineCache,
        output: PipelineState,
        timestamp: str,
        temp_map: np.ndarray,
    ) -> None:
        raise NotImplementedError("TBW")
