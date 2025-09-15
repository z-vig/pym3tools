# Standard Libraries
import os
from pathlib import Path
from typing import Optional
from enum import Enum
from dataclasses import dataclass

# Dependencies
import numpy as np
import h5py as h5  # type: ignore
import yaml

# Top-Level Imports
from m3py.PDSretrieval.file_manager import M3FileManager
from m3py.metadata_models import GeorefData

PathLike = str | os.PathLike | Path


class StepCompletionState(Enum):
    Incomplete = 0
    Complete = 1
    Cached = 2
    LoadedFromCache = 3


@dataclass
class PipelineFlags:
    cropped: StepCompletionState
    georeferenced: StepCompletionState
    external_DEM_used: StepCompletionState
    IoverF_corrected: StepCompletionState
    statistical_polishing_applied: StepCompletionState
    thermal_removed: StepCompletionState
    photometrically_corrected: StepCompletionState
    converted_to_ssa: StepCompletionState

    @classmethod
    def default(cls):
        return cls(
            cropped=StepCompletionState.Incomplete,
            georeferenced=StepCompletionState.Incomplete,
            external_DEM_used=StepCompletionState.Incomplete,
            IoverF_corrected=StepCompletionState.Incomplete,
            statistical_polishing_applied=StepCompletionState.Incomplete,
            thermal_removed=StepCompletionState.Incomplete,
            photometrically_corrected=StepCompletionState.Incomplete,
            converted_to_ssa=StepCompletionState.Incomplete,
        )


@dataclass
class PipelineState:
    data: np.ndarray
    wvl: np.ndarray
    obs: np.ndarray
    georef: GeorefData
    flags: PipelineFlags


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
        self._manager: Optional[M3FileManager] = None

    def set_file_manager(self, manager: M3FileManager):
        self._manager = manager

    @property
    def manager(self) -> M3FileManager:
        assert (
            self._manager is not None
        ), f"M3FileManager has not been set for step {self.name}"
        return self._manager

    def run(self, state: PipelineState) -> PipelineState:
        raise NotImplementedError("Subclasses must implement run()")

    def save(self, output: PipelineState) -> None:
        print(f"Saving {self.name} to {self.manager.cache} as {self.name}.")
        output.data[output.data == -999] = np.nan
        output.obs[output.obs == -999] = np.nan
        with h5.File(self.manager.cache, "r+") as f:
            g = f.create_group(self.name)
            g.create_dataset("data", data=output.data, dtype="f4")
            g.create_dataset("obs", data=output.obs, dtype="f4")
            g.attrs["wavelengths"] = output.wvl
            g.attrs["pipeline_flags"] = [
                getattr(output.flags, i.name).value
                for i in output.flags.__dataclass_fields__.values()
            ]

        with open(self.manager.georef_dir.metageo, "w") as f:
            yaml.dump(output.georef.model_dump(), f)

    def load(self) -> Optional[PipelineState]:
        # Testing whether or not step is cached.
        self._cached_step = False
        with h5.File(self.manager.cache, "r") as f:
            if self.name in f:
                self._cached_step = True

        if not self._cached_step:
            return None

        with open(self.manager.georef_dir.metageo) as f:
            raw = yaml.safe_load(f)
            georef = GeorefData(**raw)

        with h5.File(self.manager.cache, "r") as f:
            g = f[self.name]

            cached_state = PipelineState(
                data=g["data"][...],  # type:ignore
                wvl=g.attrs["wavelengths"],  # type:ignore
                obs=g["obs"][...],  # type:ignore
                georef=georef,
                flags=PipelineFlags(
                    *[
                        StepCompletionState(i)
                        for i in g.attrs["pipeline_flags"][...]  # type: ignore
                    ]
                ),
            )

            print(cached_state.data.shape, cached_state.obs.shape)
        return cached_state

    def execute(self, state: PipelineState) -> PipelineState:
        # Skip step if it is not enabled.
        if not self.enabled:
            print(f"Skipping {self.name}")
            return state

        # Return cached step if it exists.
        cached_step = self.load()
        if cached_step is not None:
            print(f"Reading {self.name} from cache...")
            return cached_step

        # Run the step and return its output after saving, if applicable.
        print(f"Running {self.name}...")
        output = self.run(state)

        if self.save_output:
            self.save(output)

        return output
