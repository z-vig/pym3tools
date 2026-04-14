from datetime import datetime

from .pipeline_state import PipelineState
from pym3tools2.data_retrieval import M3DataPaths
from pym3tools2.save_models.pipeline_cache_schema import PipelineCache
from pym3tools2.constants import TIME_FMT


class Step:
    """
    Base Class for each step in the M3 processing pipeline.

    Parameters
    ----------
    name: str
        Name of the step.
    enabled: bool, optional
        Toggles whether or not to form the step in the pipeline. Default is
        True.
    save_output: bool, optional.
        Toggles whether or not to save the data after the step is performed.
        Default is False.
    """

    def __init__(
        self, name: str, enabled: bool = True, save_output: bool = False
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.save_output = save_output

        self._catalog: M3DataPaths | None = None
        self._cache: PipelineCache | None = None

    @property
    def catalog(self) -> M3DataPaths:
        if self._catalog is None:
            raise ValueError(f"Catalog has not been set for {self.name} step.")
        return self._catalog

    @catalog.setter
    def catalog(self, value: M3DataPaths) -> None:
        self._catalog = value

    @property
    def cache(self) -> PipelineCache:
        if self._cache is None:
            raise ValueError(f"Cache has not been set for {self.name} step.")
        return self._cache

    @cache.setter
    def cache(self, value: PipelineCache) -> None:
        self._cache = value

    def run(self, state: PipelineState) -> PipelineState:
        raise NotImplementedError("Subclasses must implement run()")

    def save(self, output: PipelineState) -> None:
        raise NotImplementedError("Subclasses must implement save()")

    def execute(self, state: PipelineState) -> PipelineState:
        # Skip step if it is not enabled.
        if not self.enabled:
            print(f"Skipping {self.name}")
            return state

        # Run the step and return its output after saving, if applicable.
        print(f"Running {self.name}...")
        output = self.run(state)

        if self.save_output:
            self.save(output)

        return output

    @staticmethod
    def _time() -> str:
        return datetime.now().strftime(TIME_FMT)
