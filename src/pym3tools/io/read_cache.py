# Standard Libraries

# Dependencies
import numpy as np

# import h5py as h5  # type: ignore

# Relative Imports

# Top-Level Imports
from pym3tools.PDSretrieval.file_manager import M3FileManager


def read_pipeline_cache(manager: M3FileManager) -> np.ndarray:
    return np.zeros(5)
