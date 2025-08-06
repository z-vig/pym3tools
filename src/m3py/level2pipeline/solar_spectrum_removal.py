# Dependencies
import numpy as np
import h5py as h5  # type: ignore

# Relative Imports
from .step import Step, PipelineState
from .utils.data_fetching_utils import get_solar_correction_values


class SolarSpectrumRemoval(Step):
    def run(self, state: PipelineState) -> PipelineState:
        self.solar_spec, solar_wvl, solar_distance =\
            get_solar_correction_values(self.manager)

        with np.errstate(divide='ignore', invalid='ignore'):
            solar_spec_removed = (10**6 * state.data * np.pi) /\
                (10**6 * self.solar_spec[None, None, :] / solar_distance**2)

            solar_spec_removed = np.nan_to_num(
                solar_spec_removed, nan=-999, posinf=-999, neginf=-999
            )

        new_state = PipelineState(
            data=solar_spec_removed,
            wvl=state.wvl,
            obs=state.obs,
            georef=state.georef
        )

        return new_state

    def save(self, output: PipelineState) -> None:
        super().save(output)
        with h5.File(self.manager.cache, "r+") as f:
            g = f[self.name]
            g.attrs["solar_spectrum"] = self.solar_spec
