import numpy as np
from abc import ABC, abstractmethod

from pym3tools.data_retrieval.data_directory import M3DataPaths
from pym3tools.rdn2rfl.pipeline_state import PipelineState
from pym3tools.save_models.pipeline_cache_schema import PipelineCache


class BaseCorrection(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def modify_state(
        self, state: PipelineState, catalog: M3DataPaths
    ) -> tuple[PipelineState, np.ndarray]: ...

    @abstractmethod
    def write_to_cache(
        self,
        cache: PipelineCache,
        output: PipelineState,
        timestamp: str,
        temp_map: np.ndarray,
    ) -> None: ...
