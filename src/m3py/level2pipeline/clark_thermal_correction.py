# Standard Libraries
from dataclasses import dataclass, field, MISSING
from typing import Tuple, Final
import re

# Dependencies
import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type: ignore
import h5py as h5  # type: ignore

# Relative Imports
from .step import Step, PipelineState
from .photometric_correction import (
    get_rgi, compute_f_alpha, compute_limb_darkening_correction_factor
)
# Top-level imports
from m3py.io.read_m3 import get_wavelengths

# Some constants
h: Final[float] = 6.626 * 10**-34  # J*s, planck's constant
k_b: Final[float] = 1.381 * 10**-23  # J/K, boltzmann's constant
c: Final[float] = 2.998 * 10**8  # m/s, speed of lig


# Helper functions
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


# Thermal Correction Functions
def linear_projection(
    data: np.ndarray,
    refwvl: RefWvlSet,
    initial: bool
) -> np.ndarray:
    if initial:
        y_proj = (((data[:, :, refwvl.B.index] - data[:, :, refwvl.A.index]) /
                  (refwvl.B.actual - refwvl.A.actual)) *
                  (refwvl.C.actual - refwvl.A.actual)) +\
                  data[:, :, refwvl.A.index]
    else:
        y_proj = (((data[:, :, refwvl.E.index] - data[:, :, refwvl.D.index]) /
                  (refwvl.E.actual - refwvl.D.actual)) *
                  (refwvl.C.actual - refwvl.D.actual)) +\
                  data[:, :, refwvl.D.index]
    return y_proj


def get_temp(B, e, w, F):
    return (h * c / (w * k_b)) *\
        (np.log(((2 * h * c**2 * e) / (F * B * w**5)) + 1))**-1


def get_temp_photometric(B, e, w, F, phi):
    return (h * c / (w * k_b)) *\
        (np.log(((2 * h * c**2 * e * phi) / (F * B * w**5)) + 1))**-1


def get_thermal_spectrum(wvl, temp, solar_spec, solar_dist):
    B = ((2 * h * c**2) / (wvl ** 5)) *\
        (1 / (np.exp((h * c) / (wvl * k_b * temp)) - 1))
    F = 10**6 * solar_spec / np.pi
    therm_spec = (B / F) * solar_dist**2
    return therm_spec


class ClarkThermalCorrection(Step):
    def __init__(self, name, max_iterations: int = 12, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.max_iterations = max_iterations

    def _load_context_variables(
        self,
        state: PipelineState
    ) -> Tuple[
        RefWvlSet, np.ndarray, np.ndarray, float, RegularGridInterpolator
    ]:
        reference_wvl = RefWvlSet.from_data(state.wvl)
        solspec_parse = re.compile(r"\s*(\d{2,4}.\d{6})")
        _, bbl = get_wavelengths(self.manager)
        with open(self.manager.cal_dir.solar_spectrum) as f:
            data_array = np.array(
                [re.findall(solspec_parse, i) for i in f.readlines()],
                dtype=np.float32
            )
            solar_wvl = data_array[bbl, 0]
            solar_spec = data_array[bbl, 1]

        solar_distance_pattern = re.compile(
            r"SOLAR_DISTANCE\s*=\s(\d.\d*)\s<AU>"
        )
        with open(self.manager.pds_dir.l1.lbl) as f:
            solar_distance = float(re.findall(
                solar_distance_pattern, f.read()
            )[0])

        pattern = re.compile(r"\s\d.\d{9}")
        with open(self.manager.cal_dir.phase_function) as f:
            phase_function_lookup = np.array(
                [re.findall(pattern, i) for i in f.readlines()[1:]],
                dtype=np.float32
            )
            phase_function_lookup = phase_function_lookup[:100, bbl]

        f_alpha_rgi = get_rgi(phase_function_lookup)

        return (
            reference_wvl, solar_wvl, solar_spec, solar_distance, f_alpha_rgi
        )

    def run(self, state: PipelineState) -> PipelineState:
        # Creating temperature logging array
        self.temp_log = np.full(
            (*state.data.shape[:2], self.max_iterations+1), np.nan
        )

        self.final_correction = state.data.copy()
        iter_counter = 0

        # Pre-loading context variables
        refwvl, sol_wvl, sol_spec, sol_dist, rgi =\
            self._load_context_variables(state)

        state.data = state.data * sol_dist ** 2

        initial_thermal_component = state.data[:, :, refwvl.C.index] -\
            linear_projection(state.data, refwvl, initial=True)

        initial_thermal_component[initial_thermal_component < 0] = np.nan

        initial_emissivity = 1 - state.data[:, :, refwvl.A.index]

        Fidx = np.argmin(np.abs(sol_wvl - refwvl.C.actual))

        initial_temp = get_temp(
            initial_thermal_component,
            initial_emissivity,
            refwvl.C.actual * 10**-9,
            10**6 * sol_spec[Fidx] / np.pi
        )

        initial_thermal_spectra = get_thermal_spectrum(
            state.wvl[None, None, :] * 10**-9,
            initial_temp[:, :, None],
            sol_spec[None, None, :],
            sol_dist
        )

        initial_thermal_removed = state.data - initial_thermal_spectra

        # Phase Function Factor
        f_alpha_norm = compute_f_alpha(state.obs[:, :, 2], rgi, state.wvl.size)

        # Limb-Darkening Factor
        ldf = compute_limb_darkening_correction_factor(state.obs)

        self.photo_coefs = ldf[:, :, None] * f_alpha_norm

        next_step = self.photo_coefs * initial_thermal_removed

        self.temp_log[:, :, iter_counter] = initial_temp

        correction_exists = ~np.isnan(next_step)
        self.final_correction[correction_exists] = next_step[correction_exists]

        while True:
            wvl_dependent_emiss = 1 - next_step
            next_thermal_component = next_step[:, :, refwvl.C.index] -\
                linear_projection(next_step, refwvl, initial=False)
            next_thermal_component[next_thermal_component < 0] = np.nan

            next_temp = get_temp_photometric(
                next_thermal_component,
                wvl_dependent_emiss[:, :, refwvl.C.index],
                refwvl.C.actual * 10**-9,
                10**6 * sol_spec[Fidx] / np.pi,
                self.photo_coefs[:, :, refwvl.C.index]
            )

            next_thermal_spectra = get_thermal_spectrum(
                state.wvl[None, None, :] * 10**-9,
                next_temp[:, :, None],
                sol_spec[None, None, :],
                sol_dist
            )

            next_thermal_removed = state.data - next_thermal_spectra

            next_step = self.photo_coefs * next_thermal_removed

            iter_counter += 1
            print(f"Iteration Count: {iter_counter}")

            if iter_counter < self.max_iterations:
                self.temp_log[:, :, iter_counter] = next_temp
                correction_exists = ~np.isnan(next_step)
                self.final_correction[correction_exists] =\
                    next_step[correction_exists]
            else:
                self.temp_log[:, :, iter_counter] = next_temp
                correction_exists = ~np.isnan(next_step)
                self.final_correction[correction_exists] =\
                    next_step[correction_exists]
                break

            if np.all(
                np.abs(next_temp - self.temp_log[:, :, iter_counter-1]) < 2
            ):
                self.temp_log[:, :, iter_counter] = next_temp
                correction_exists = ~np.isnan(next_step)
                self.final_correction[correction_exists] =\
                    next_step[correction_exists]
                break

        new_state = PipelineState(
            data=self.final_correction,
            wvl=state.wvl,
            obs=state.obs,
            georef=state.georef
        )

        return new_state

    def save(self, output: PipelineState) -> None:
        super().save(output)
        with h5.File(self.manager.cache, "r+") as f:
            g = f[self.name]
            assert isinstance(g, h5.Group)
            g.create_dataset("photometric_coefficients",
                             data=self.photo_coefs, dtype="f4")
            g.create_dataset("temp", data=self.temp_log, dtype="f4")
