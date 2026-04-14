from enum import IntFlag, auto
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from cubio.geotools.models import GeotransformModel

from .data_transfer_classes import GeoreferencingGeometry

from pym3tools2.save_models.attribute_models import (
    StandardDatasetAttrs,
    OneDimensionDatasetAttrs,
)
from pym3tools2.constants import MOON_GCS_PRJ
from pym3tools2.rdn2rfl.retrieve_terrain_data import M3Geometry


class StepCompletionState(IntFlag):
    Incomplete = 0
    Complete = auto()
    Partial = auto()


class CompletedFlag(IntFlag):
    NONE = 0
    CROPPED = auto()
    GEOREFERENCED = auto()
    SOLAR_REMOVED = auto()
    STATPOL_APPLIED = auto()
    THERMAL_REMOVED = auto()
    PHOTO_CORR_APPLIED = auto()
    SSA_CONVERTED = auto()


@dataclass
class PipelineState:
    data: np.ndarray | xr.DataArray
    wavelengths: list[float]
    obs: M3Geometry[np.ndarray]
    crs: str = MOON_GCS_PRJ
    gtrans: GeotransformModel = field(default_factory=GeotransformModel.null)
    geom: GeoreferencingGeometry = field(
        default_factory=GeoreferencingGeometry
    )
    flags: CompletedFlag = CompletedFlag.NONE
    history: list = field(default_factory=list)

    def mark(self, step: CompletedFlag, **metadata):
        self.flags |= step
        self.history.append((step, metadata))


def get_standard_dset_attrs(
    timestamp: str,
    output: PipelineState,
) -> StandardDatasetAttrs:
    return {
        "timestamp": timestamp,
        "nrows": output.data.shape[0],
        "ncols": output.data.shape[1],
        "nbands": output.data.shape[2],
        "measvals": output.wavelengths,
        "bandlbls": [
            f"Band {n+1} ({i}nm)" for n, i in enumerate(output.wavelengths)
        ],
        "crs": output.crs,
        "geotransform": output.gtrans.togdal(),
    }


def get_1d_dset_attrs(
    timestamp: str, data_1d: np.ndarray | list, measvals: list[float]
) -> OneDimensionDatasetAttrs:
    return {
        "timestamp": timestamp,
        "length": len(data_1d),
        "measvals": measvals,
    }
