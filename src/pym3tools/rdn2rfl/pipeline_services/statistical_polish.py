from dataclasses import dataclass
from typing import overload, Literal

import numpy as np
import xarray as xr
import re

from pym3tools2.data_retrieval import M3DataPaths
from pym3tools2.rdn2rfl.step_model import PipelineState
from pym3tools2.rdn2rfl.pipeline_state import (
    CompletedFlag,
    get_standard_dset_attrs,
    get_1d_dset_attrs,
)
from pym3tools2.save_models.pipeline_cache_schema import PipelineCache


@dataclass
class StatPolishCoefficients:
    wvl: np.ndarray
    vals: np.ndarray


def retrieve_statpol_coefs(
    catalog: M3DataPaths,
) -> StatPolishCoefficients:
    """
    Retrieves statistical polishing coefficients.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Statpol Wavelengths, Statpol Coefficients
    """
    statpol_wvl: list[float] = []
    statpol_coefs: list[float] = []
    with open(catalog.statistical_polish.tab) as f:
        for line in f.readlines():
            num_line = [float(i) for i in re.split(r"\s+", line.strip())]
            statpol_wvl.append(num_line[1])
            statpol_coefs.append(num_line[2])

    return StatPolishCoefficients(
        np.array(statpol_wvl), np.array(statpol_coefs)
    )


@overload
def _multiply_statpol_coefs(
    data: np.ndarray, coefs: np.ndarray
) -> np.ndarray: ...


@overload
def _multiply_statpol_coefs(
    data: xr.DataArray, coefs: np.ndarray
) -> xr.DataArray: ...


def _multiply_statpol_coefs(
    data: np.ndarray | xr.DataArray, coefs: np.ndarray
) -> np.ndarray | xr.DataArray:
    if coefs.ndim > 1:
        raise ValueError(f"Invalid StatPol Coefficients shape: {coefs.shape}")
    return data * coefs[None, None, :]


def modify_state(state: PipelineState, coefs: StatPolishCoefficients):
    state.data = _multiply_statpol_coefs(state.data, coefs.vals)
    state.flags |= CompletedFlag.STATPOL_APPLIED
    return state


def write_to_cache(
    cache: PipelineCache,
    output: PipelineState,
    timestamp: str,
    coefs: StatPolishCoefficients,
    instr_state: Literal["Warm", "Cold"],
) -> None:
    cubeattrs = get_standard_dset_attrs(timestamp, output)
    statpolattrs = get_1d_dset_attrs(timestamp, coefs.vals, coefs.wvl.tolist())

    cache.stat_polished.cube.write(np.array(output.data, dtype=np.float32))
    cache.stat_polished.cube.set_attrs(cubeattrs)

    cache.stat_polished.statpol_coefficients.write(coefs.vals)
    cache.stat_polished.statpol_coefficients.set_attrs(statpolattrs)

    cache.stat_polished.set_attrs(
        {
            "timestamp": timestamp,
            "flags": output.flags,
            "instrument_state": instr_state,
        }
    )
