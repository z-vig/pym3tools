# Standard Libraries
import os
from pathlib import Path
from typing import Sequence

# Dependencies

# Relative Imports
from .step import Step

# Top-Level Imports
from m3py.PDSretrieval.file_config import M3FileConfig
from m3py.io import read_m3
from m3py.constants import L1

type PathType = str | os.PathLike


class M3Level2Pipeline():
    """
    Main pipeline controller for L1 to L2 M3 pipeline.
    """
    def __init__(self, steps: Sequence[Step], config: M3FileConfig) -> None:
        self.steps = steps
        self.data = read_m3(config.pds_dir.l1.rdn_img, L1, config.acq_type)

    def run(self):
        for stp in self.steps:
            stp.run(self.data)


def process_m3(config_file: PathType) -> None:
    config_file = Path(config_file)
    return None
