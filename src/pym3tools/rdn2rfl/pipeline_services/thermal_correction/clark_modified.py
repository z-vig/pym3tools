"""
Clark Thermal Correction (Modified)
---
Functions for implementing a modified version of the Clark et al., 2011 thermal
correction. In this case, the PDS-included temperatures are used to directly
remove a planck blackbody from each spectrum. This is in place of
re-calculating all the temperatures as in the regular Clark et al., 2011
method. (See `clark.py`). This method aims to more accurately recreate
the PDS-included thermal correction.

Clark, R. N., Pieters, C. M., Green, R. O., Boardman, J. W., & Petro, N. E.
(2011). Thermal removal from near-infrared imaging spectroscopy data of the
Moon. Journal of Geophysical Research, 116.
https://doi.org/10.1029/2010je003751
"""

import numpy as np
from cubio import cubedata_from_json_file

from pym3tools.data_retrieval.data_directory import M3DataPaths
from pym3tools.rdn2rfl.data_transfer_classes import GeoreferencingGeometry
from pym3tools.rdn2rfl.pipeline_state import (
    PipelineState,
    CompletedFlag,
    get_standard_dset_attrs,
)
from pym3tools.save_models.pipeline_cache_schema import PipelineCache

from .base_thermal_correction import BaseCorrection
from .clark import get_thermal_spectrum
from ..solar_removal import retrieve_solar_data


def get_pds_temperatures(catalog: M3DataPaths) -> np.ndarray:
    _, sup_img = cubedata_from_json_file(catalog.sup.json)
    sup_img.transpose_to("BIP")
    return sup_img.array.values[:, :, 1]


def resample_pds_temps(
    pds_temps: np.ndarray, geom: GeoreferencingGeometry
) -> np.ndarray:
    return geom.swath_to_gridded_data(pds_temps)


def get_thermal_cube(
    catalog: M3DataPaths,
    data: np.ndarray,
    wvl: np.ndarray,
    temp_map: np.ndarray,
):
    sol = retrieve_solar_data(catalog)
    return get_thermal_spectrum(
        wvl[None, None, :] * 1e-9,
        temp_map[:, :, None],
        1 - data,
        sol.solar_spectrum[None, None, :],
        sol.solar_distance,
    )


def remove_thermal_component(data: np.ndarray, thermal_cube: np.ndarray):
    if data.ndim != thermal_cube.ndim:
        raise ValueError(f"Invalid thermal cube size: {thermal_cube.shape}")
    return data - thermal_cube


class ClarkModified(BaseCorrection):
    def __init__(self) -> None:
        super().__init__()

    def modify_state(
        self, state: PipelineState, catalog: M3DataPaths
    ) -> tuple[PipelineState, np.ndarray]:
        pds_temps_swath = get_pds_temperatures(catalog)
        pds_temps_resmp = resample_pds_temps(pds_temps_swath, state.geom)
        therm = get_thermal_cube(
            catalog,
            np.array(state.data),
            np.array(state.wavelengths),
            pds_temps_resmp,
        )
        state.data = remove_thermal_component(np.array(state.data), therm)
        state.flags |= CompletedFlag.THERMAL_REMOVED
        return state, pds_temps_resmp

    def write_to_cache(
        self,
        cache: PipelineCache,
        output: PipelineState,
        timestamp: str,
        temp_map: np.ndarray,
    ) -> None:
        cubeattrs = get_standard_dset_attrs(timestamp, output)
        tempattrs = get_standard_dset_attrs(timestamp, output)

        tempattrs.update(
            {"bandlbls": ["temp"], "nbands": 1, "measvals": [0.0]}
        )

        cache.thermal_corrected.cube.write(np.array(output.data))
        cache.thermal_corrected.cube.set_attrs(cubeattrs)

        cache.thermal_corrected.temperature_map.write(temp_map)
        cache.thermal_corrected.temperature_map.set_attrs(tempattrs)

        cache.thermal_corrected.set_attrs(
            {
                "flags": output.flags,
                "method": "Clark_Modified",
                "timestamp": timestamp,
            }
        )
