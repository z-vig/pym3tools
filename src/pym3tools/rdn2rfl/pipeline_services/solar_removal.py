from dataclasses import dataclass

import numpy as np
import xarray as xr
import re

from pym3tools.data_retrieval import M3DataPaths
from pym3tools.rdn2rfl.step_model import PipelineState
from pym3tools.rdn2rfl.pipeline_state import (
    CompletedFlag,
    get_standard_dset_attrs,
    get_1d_dset_attrs,
)
from pym3tools.save_models.pipeline_cache_schema import PipelineCache


@dataclass
class SolarData:
    solar_wvl: np.ndarray
    solar_spectrum: np.ndarray
    solar_distance: float
    is_scaled: bool = False

    def scale_solar_spectrum(self) -> None:
        self.solar_spectrum = self.solar_spectrum / (
            np.pi * self.solar_distance**2
        )
        self.is_scaled = True


def _fetch_solar_spectrum(
    catalog: M3DataPaths,
) -> tuple[np.ndarray, np.ndarray]:
    solar_wvl: list[float] = []
    solar_rdn: list[float] = []
    with open(catalog.solar_spectrum.tab) as f:
        for i in f.readlines():
            row = [float(j) for j in re.split(r"\s+", i.strip())]
            solar_wvl.append(row[0])
            solar_rdn.append(row[1])

    return np.array(solar_wvl), np.array(solar_rdn)


def _fetch_solar_distance(catalog: M3DataPaths) -> float:
    solar_dist_ptrn = re.compile(r"SOLAR_DISTANCE\s*=\s(\d.\d*)\s<AU>")
    with open(catalog.L1_lbl) as f:
        solar_distance = float(re.findall(solar_dist_ptrn, f.read())[0])
    return solar_distance


def retrieve_solar_data(catalog: M3DataPaths) -> SolarData:
    solar_wvl, solar_rdn = _fetch_solar_spectrum(catalog)
    solar_dist = _fetch_solar_distance(catalog)

    return SolarData(solar_wvl, solar_rdn, solar_dist)


def _divide_out_solar_spectrum(
    data: np.ndarray | xr.DataArray, solar_spectrum: np.ndarray
):
    if solar_spectrum.ndim > 1:
        raise ValueError(
            f"Invalid Solar Spectrum size: {solar_spectrum.shape}"
        )
    return data / solar_spectrum[None, None, :]


def modify_state(state: PipelineState, solar_data: SolarData):
    if not solar_data.is_scaled:
        raise RuntimeError("Solar Spectrum has not been scaled correctly.")

    state.data = _divide_out_solar_spectrum(
        state.data, solar_data.solar_spectrum
    )
    state.flags |= CompletedFlag.SOLAR_REMOVED

    return state


def write_to_cache(
    timestamp: str,
    cache: PipelineCache,
    output: PipelineState,
    solar_data: SolarData,
) -> None:
    cubeattrs = get_standard_dset_attrs(timestamp, output)
    solspecattrs = get_1d_dset_attrs(
        timestamp, solar_data.solar_spectrum, solar_data.solar_wvl.tolist()
    )

    cache.solar_removed.cube.write(np.array(output.data, dtype=np.float32))
    cache.solar_removed.cube.set_attrs(cubeattrs)

    cache.solar_removed.solarspectrum.write(solar_data.solar_spectrum)
    cache.solar_removed.solarspectrum.set_attrs(solspecattrs)

    cache.solar_removed.set_attrs(
        {
            "timestamp": timestamp,
            "flags": output.flags,
            "solar_distance": solar_data.solar_distance,
        }
    )
