# Standard Libraries
import os

# Local Imports
from .m3_data_class_DEPRECATED import M3Data

# Top-Level Imports
from m3py.constants import L0


class L0Data(M3Data):
    """
    Class for reading and storing Level 0 data.
    """
    def __init__(self, img_path: str | os.PathLike):
        super().__init__(img_path, L0)
