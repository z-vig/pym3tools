import numpy as np

from pym3tools.rdn2rfl.pipeline_state import (
    PipelineState,
    CompletedFlag,
)
from pym3tools.data_retrieval.data_directory import M3DataPaths
from pym3tools.save_models.pipeline_cache_schema import PipelineCache
from pym3tools.types import ThermalCorrectionMethod

from .base_thermal_correction import BaseCorrection
from .clark import Clark
from .clark_modified import ClarkModified
from .shkuratov import Shkuratov
from .li_milliken import LiMilliken
from .wohler_grumpe import WohlerGrumpe

thermal_correction_dispatcher: dict[
    ThermalCorrectionMethod, BaseCorrection
] = {
    "Clark": Clark(),
    "Clark_Modified": ClarkModified(),
    "Li_Milliken": LiMilliken(),
    "Shkuratov": Shkuratov(),
    "Wohler_Grumpe": WohlerGrumpe(),
}


def modify_state(
    state: PipelineState,
    catalog: M3DataPaths,
    method: ThermalCorrectionMethod,
) -> tuple[PipelineState, np.ndarray]:
    corr_method = thermal_correction_dispatcher[method]
    state, temp_map = corr_method.modify_state(state, catalog)
    state.flags |= CompletedFlag.THERMAL_REMOVED
    return state, temp_map


def write_to_cache(
    cache: PipelineCache,
    output: PipelineState,
    timestamp: str,
    temp_map: np.ndarray,
    method: ThermalCorrectionMethod,
) -> None:
    corr_method = thermal_correction_dispatcher[method]
    corr_method.write_to_cache(cache, output, timestamp, temp_map)
