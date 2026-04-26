import numpy as np

from pym3tools.data_retrieval.data_directory import M3DataPaths
from pym3tools.rdn2rfl.pipeline_state import PipelineState
from pym3tools.save_models.pipeline_cache_schema import PipelineCache

from .base_thermal_correction import BaseCorrection


class WohlerGrumpe(BaseCorrection):
    def __init__(self) -> None:
        super().__init__()

    def modify_state(
        self, state: PipelineState, catalog: M3DataPaths
    ) -> tuple[PipelineState, np.ndarray]:
        raise NotImplementedError("TBW")

    def write_to_cache(
        self,
        cache: PipelineCache,
        output: PipelineState,
        timestamp: str,
        temp_map: np.ndarray,
    ) -> None:
        raise NotImplementedError("TBW")
