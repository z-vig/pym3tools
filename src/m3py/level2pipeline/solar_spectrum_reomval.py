# Standard Libraries
import re

# Dependencies
import numpy as np
import h5py as h5  # type: ignore

# Relative Imports
from m3py.level2pipeline.step import PipelineState
from .step import Step

# Top-Level Imports
from m3py.io.read_m3 import get_wavelengths


class SolarSpectrumReadError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class SolarSpectrumRemoval(Step):
    def run(self, state: PipelineState) -> PipelineState:
        solspec_parse = re.compile(r"\s*(\d{2,4}.\d{6})")
        _, bbl = get_wavelengths(self.manager)
        with open(self.manager.cal_dir.solar_spectrum) as f:
            data_array = np.array(
                [re.findall(solspec_parse, i) for i in f.readlines()],
                dtype=np.float32
            )
            solar_wvl = data_array[bbl, 0]
            self.solar_spec = data_array[bbl, 1]

        solar_distance_pattern = re.compile(
            r"SOLAR_DISTANCE\s*=\s(\d.\d*)\s<AU>"
        )
        with open(self.manager.pds_dir.l1.lbl) as f:
            solar_distance = float(re.findall(
                solar_distance_pattern, f.read()
            )[0])

        if not np.allclose(solar_wvl, state.wvl):
            raise SolarSpectrumReadError(
                "The solar spectrum wavelength values do not match the data"
                "wavelength values."
            )

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
