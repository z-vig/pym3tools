# Standard Libraries
from typing import Tuple, Optional

# Dependencies
import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type: ignore
import h5py as h5  # type: ignore
from rasterio.coords import BoundingBox  # type: ignore

# Relative Imports
from .step import Step, PipelineState
from .photometric_correction import (
    compute_f_alpha,
    compute_limb_darkening_correction_factor,
)
from .utils.thermal_correction_utils import (
    RefWvlSet,
    linear_projection,
    get_temp,
    get_thermal_spectrum,
    get_temp_photometric,
)
from .utils.data_fetching_utils import (
    get_solar_correction_values,
    get_phase_function_rgi,
)

# Top-level imports
from m3py.io.read_m3 import read_m3, Window
from m3py.io.read_m3_georef import read_m3_georef
from m3py.formats.m3_data_format import SUP


def get_geometry_correction(
    state: PipelineState, rgi: RegularGridInterpolator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns photometric correction factors and cosine incidence correction
    factors.

    Parameters
    ----------
    state: PipelineState
        PipelineState object.
    rgi: RegularGridInterpolator
        Interpolation of the PDS-provided phase function.

    Returns
    -------
    photo_coefs: np.ndarray
        Photometric coefficients, normalized to 30, 0, 30.
    cos_correction: np.ndarray
        Cosine of the incidence angle per pixel.
    """
    # Phase Function Factor
    f_alpha, f_alpha_norm = compute_f_alpha(
        state.obs[:, :, 2], rgi, state.wvl.size
    )

    # Limb-Darkening Factor
    ldf = compute_limb_darkening_correction_factor(state.obs)

    photo_coefs = ldf[:, :, None] * f_alpha_norm

    cos_correction = (
        np.cos((np.pi / 180) * state.obs[:, :, 0])[:, :, None] * f_alpha
    )

    return photo_coefs, cos_correction


class ClarkThermalCorrection(Step):
    def __init__(
        self,
        name,
        max_iterations: int = 12,
        use_pds_temperatures: bool = False,
        georeferenced: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self.max_iterations = max_iterations
        self.use_pds_temperatures = use_pds_temperatures
        self.georef = georeferenced

    def _load_context_variables(
        self, state: PipelineState
    ) -> Tuple[
        RefWvlSet, np.ndarray, np.ndarray, float, RegularGridInterpolator
    ]:
        reference_wvl = RefWvlSet.from_data(state.wvl)

        solar_spec, solar_wvl, solar_distance = get_solar_correction_values(
            self.manager
        )

        f_alpha_rgi = get_phase_function_rgi(self.manager)

        return (
            reference_wvl,
            solar_wvl,
            solar_spec,
            solar_distance,
            f_alpha_rgi,
        )

    def _initial_temp_correction(
        self,
        state: PipelineState,
        refwvl: RefWvlSet,
        sol_spec: np.ndarray,
        sol_wvl: np.ndarray,
        sol_dist: float,
    ):

        state.data = state.data * sol_dist**2

        initial_thermal_component = state.data[
            :, :, refwvl.C.index
        ] - linear_projection(state.data, refwvl, initial=True)

        print(f"LOG: {initial_thermal_component[1031, 232]}")

        initial_thermal_component[initial_thermal_component < 0] = np.nan

        initial_emissivity = 1 - state.data[:, :, refwvl.A.index]

        Fidx = np.argmin(np.abs(sol_wvl - refwvl.C.actual))

        initial_temp = get_temp(
            initial_thermal_component,
            initial_emissivity,
            refwvl.C.actual * 10**-9,
            sol_spec[Fidx],
        )
        print(f"LOG: {initial_temp[1031, 232]}")

        initial_thermal_spectra = get_thermal_spectrum(
            state.wvl[None, None, :] * 10**-9,
            initial_temp[:, :, None],
            initial_emissivity[:, :, None],
            sol_spec[None, None, :],
            sol_dist,
        )

        initial_thermal_removed = state.data - initial_thermal_spectra

        return (
            initial_thermal_removed,
            initial_temp,
        )

    def run(self, state: PipelineState) -> PipelineState:
        # Pre-loading context variables
        refwvl, sol_wvl, sol_spec, sol_dist, rgi = (
            self._load_context_variables(state)
        )

        # Getting geometry correction factors
        self.photo_coefs, self.cos_correction = get_geometry_correction(
            state, rgi
        )

        # Creating temperature logging array
        self.temp_log = np.full(
            (*state.data.shape[:2], self.max_iterations + 1), np.nan
        )

        if self.use_pds_temperatures:
            if self.georef is None:
                raise ValueError(
                    "If using PDS Temperatures, `georeference` must be a bool."
                )
            print(
                "Skipping iterative temperature solution, using pre-defined"
                " temperature values."
            )
            window = Window(
                state.georef.col_offset,
                state.georef.col_offset,
                state.georef.width,
                state.georef.height,
            )
            bbox = BoundingBox(
                left=state.georef.left_bound,
                bottom=state.georef.bottom_bound,
                right=state.georef.right_bound,
                top=state.georef.top_bound,
            )

            if self.georef:
                pds_temps = read_m3_georef(self.manager, bbox, "SUP")[:, :, 1]
            else:
                pds_temps = read_m3(
                    self.manager.pds_dir.l2.sup_img,
                    SUP,
                    self.manager.acq_type,
                    window=window,
                )[:, :, 1]

            pds_temps[pds_temps == 0.1] = np.nan
            pds_temps[pds_temps == -999] = np.nan

            thermal_spec = get_thermal_spectrum(
                state.wvl[None, None, :] * 10**-9,
                pds_temps[:, :, None],
                1 - state.data,
                sol_spec[None, None, :],
                sol_dist,
            )

            thermal_spec[~np.isfinite(thermal_spec[:, :, 0]), :] = np.zeros(
                thermal_spec.shape[2]
            )[None, None, :]

            self.temp_log[:, :, 0] = pds_temps

            new_state = PipelineState(
                data=self.photo_coefs * (state.data - thermal_spec),
                wvl=state.wvl,
                obs=state.obs,
                georef=state.georef,
            )
            return new_state

        self.final_correction = state.data.copy()
        iter_counter = 0

        initial_thermal_removed, initial_temp = self._initial_temp_correction(
            state, refwvl, sol_spec, sol_wvl, sol_dist
        )

        self.temp_log[:, :, iter_counter] = initial_temp

        correction_exists = ~np.isnan(initial_thermal_removed)
        self.final_correction[correction_exists] = (
            initial_thermal_removed[correction_exists]
            * self.photo_coefs[correction_exists]
        )

        next_step = initial_thermal_removed / self.cos_correction

        while True:
            wvl_dependent_emiss = 1 - next_step
            next_thermal_component = next_step[
                :, :, refwvl.C.index
            ] - linear_projection(next_step, refwvl, initial=False)
            next_thermal_component[next_thermal_component < 0] = np.nan

            Fidx = np.argmin(np.abs(sol_wvl - refwvl.C.actual))

            next_temp = get_temp_photometric(
                next_thermal_component,
                wvl_dependent_emiss[:, :, refwvl.C.index],
                refwvl.C.actual * 10**-9,
                sol_spec[Fidx],
                1 / self.cos_correction[:, :, refwvl.C.index],
            )

            next_thermal_spectra = get_thermal_spectrum(
                state.wvl[None, None, :] * 10**-9,
                next_temp[:, :, None],
                wvl_dependent_emiss,
                sol_spec[None, None, :],
                sol_dist,
            )

            next_thermal_removed = state.data - next_thermal_spectra

            next_step = next_thermal_removed / self.cos_correction

            iter_counter += 1
            print(f"Iteration Count: {iter_counter}")

            self.temp_log[:, :, iter_counter] = next_temp
            correction_exists = ~np.isnan(next_step)
            self.final_correction[correction_exists] = (
                next_thermal_removed[correction_exists]
                * self.photo_coefs[correction_exists]
            )
            if iter_counter == self.max_iterations:
                break

            if np.all(
                np.abs(next_temp - self.temp_log[:, :, iter_counter - 1]) < 2
            ):
                break

        new_state = PipelineState(
            data=self.final_correction,
            wvl=state.wvl,
            obs=state.obs,
            georef=state.georef,
        )

        return new_state

    def save(self, output: PipelineState) -> None:
        super().save(output)
        with h5.File(self.manager.cache, "r+") as f:
            g = f[self.name]
            assert isinstance(g, h5.Group)
            g.create_dataset(
                "photometric_coefficients", data=self.photo_coefs, dtype="f4"
            )
            g.create_dataset("temp", data=self.temp_log, dtype="f4")
