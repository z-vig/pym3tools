# Standard Libraries

# Dependencies
# import h5py as h5
import numpy as np

# Top-Level Imports
from m3py.PDSretrieval import M3FileConfig

# Relative Imports
from .step import Step


class InitCache(Step):
    def __init__(self, config: M3FileConfig) -> None:
        self.config = config

    def run(self, data: np.ndarray):
        print(self.config.root)
