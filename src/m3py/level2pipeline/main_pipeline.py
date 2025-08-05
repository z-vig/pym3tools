# Standard Libraries
import os
from pathlib import Path
from typing import Sequence

# Dependencies
# from rasterio.coords import BoundingBox

# Relative Imports
from .step import Step, PipelineState

# Top-Level Imports
from m3py.metadata_models import GeorefData
from m3py.PDSretrieval.file_manager import M3FileManager
from m3py.io.read_m3 import read_m3, get_wavelengths
from m3py.constants import L1, OBS

type PathType = str | os.PathLike


class M3Level2Pipeline():
    """
    Main pipeline controller for L1 to L2 M3 pipeline.
    """
    def __init__(self, steps: Sequence[Step], manager: M3FileManager) -> None:
        self.steps = steps
        for step in self.steps:
            step.set_file_manager(manager)

        print("Initializing Pipeline...")
        data = read_m3(manager.pds_dir.l1.rdn_img, L1, manager.acq_type)
        obs = read_m3(manager.pds_dir.l1.obs_img, OBS, manager.acq_type)
        wvl, bbl = get_wavelengths(manager)

        # Getting rid of bad bands
        data = data[:, :, bbl]
        wvl = wvl[bbl]

        georef = GeorefData.from_numpy(data)

        self.state = PipelineState(data, wvl, obs, georef)

    def run(self) -> PipelineState:
        state = self.state
        for step in self.steps:
            state = step.execute(state)
        return state


def process_m3(config_file: PathType) -> None:
    config_file = Path(config_file)
    return None
