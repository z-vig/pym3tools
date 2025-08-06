# Standard Libraries
from typing import Tuple

# Dependencies
import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type: ignore
import h5py as h5  # type: ignore

# Relative Imports
from .step import Step, PipelineState
from .photometric_correction import (
    compute_f_alpha, compute_limb_darkening_correction_factor
)
from .utils.thermal_correction_utils import (
    RefWvlSet, linear_projection, get_temp, get_thermal_spectrum,
    get_temp_photometric
)
from .utils.data_fetching_utils import (
    get_solar_correction_values, get_phase_function_rgi
)


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

        solar_spec, solar_wvl, solar_distance =\
            get_solar_correction_values(self.manager)

        f_alpha_rgi = get_phase_function_rgi(self.manager)

        return (
            reference_wvl, solar_wvl, solar_spec, solar_distance, f_alpha_rgi
        )

    def _initial_temp_correction(
        self,
        state: PipelineState,
        refwvl: RefWvlSet,
        sol_spec: np.ndarray,
        sol_wvl: np.ndarray,
        sol_dist: float,
        rgi: RegularGridInterpolator
    ):

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

        photo_coefs = ldf[:, :, None] * f_alpha_norm

        initial_correction = photo_coefs * initial_thermal_removed

        return initial_correction, initial_temp, photo_coefs

    def run(self, state: PipelineState) -> PipelineState:
        # Pre-loading context variables
        refwvl, sol_wvl, sol_spec, sol_dist, rgi =\
            self._load_context_variables(state)

        # Creating temperature logging array
        self.temp_log = np.full(
            (*state.data.shape[:2], self.max_iterations+1), np.nan
        )

        self.final_correction = state.data.copy()
        iter_counter = 0

        next_step, initial_temp, self.photo_coefs =\
            self._initial_temp_correction(
                state, refwvl, sol_spec, sol_wvl, sol_dist, rgi
            )

        self.temp_log[:, :, iter_counter] = initial_temp

        correction_exists = ~np.isnan(next_step)
        self.final_correction[correction_exists] = next_step[correction_exists]

        while True:
            wvl_dependent_emiss = 1 - next_step
            next_thermal_component = next_step[:, :, refwvl.C.index] -\
                linear_projection(next_step, refwvl, initial=False)
            next_thermal_component[next_thermal_component < 0] = np.nan

            Fidx = np.argmin(np.abs(sol_wvl - refwvl.C.actual))

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
