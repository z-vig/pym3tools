# Standard Libraries
import os
from pathlib import Path
import shutil
import re

# Dependencies
import h5py as h5  # type: ignore

# Relative Imports
from .pds_dir import PDSDir
from .cal_dir import CalDir
from .georef_dir import GeorefDir
from .file_retrieval_patterns import FileRetrievalPatterns


class M3FileManager:
    root: os.PathLike
    data_ID_long: str
    data_ID: str
    acq_type: str
    pds_dir: PDSDir
    cal_dir: CalDir
    georef_dir: GeorefDir
    """
    Stores data for M3 file locations for a single M3 stripe.

    Parameters
    ----------
    root: str | os.PathLike
        Path to root directory holding multiple M3 datasets.
    data_id: str
        Data ID of M3 stripe. This should correspond to the name of a directory
        within the root directory.

    Attributes
    ----------
    root: PathLike
        Root directory for all processing data.
    data_ID: str
        Short form of the data ID, only including the timestamp.
    data_ID_long: str
        Long form of the data ID
    acq_type: str
        Either "global" or "targeted". Reflects M3 acquisition mode.
    pds_dir: PathLike
        Planetary Data System directory. Contains raw downloads from the PDS.
    cal_dir: PathLike
        Calibration directory. Contains calibration data from PDS.
    georef_dir: PathLike
        Georeferenced Directory. Contains all GeoTiff, georeferenced data.
    """
    def __init__(
        self,
        root: str | os.PathLike,
        data_id: str,
        reset_cache: bool = False
    ):
        self.root = Path(root, data_id)
        self.cache = Path(self.root, "pipeline_cache.hdf5")
        self.data_ID_long = data_id
        self.data_ID = re.findall(FileRetrievalPatterns.short_id, data_id)[0]
        self.acq_type = re.findall(FileRetrievalPatterns.acq_type, data_id)[0]
        if self.acq_type == "G":
            self.acq_type = "global"
        elif self.acq_type == "T":
            self.acq_type = "targeted"
        else:
            raise ValueError("Invalid Acquisition Type.")

        if Path(root, f"{self.data_ID_long}_urls.txt").is_file():
            print("Initializing new stripe directory...")
            self._initialize_directories(
                Path(root, f"{self.data_ID_long}_urls.txt")
            )

        if reset_cache:
            self._reset_cache()

        self.pds_dir = PDSDir(Path(self.root, "pds_data"), self.data_ID_long)
        self.cal_dir = CalDir(Path(self.root, "cal_data"), self.data_ID_long)
        self.georef_dir = GeorefDir(Path(self.root, "georef_data"),
                                    self.data_ID_long)

    def _initialize_directories(self, urls_file: os.PathLike):
        Path(self.root).mkdir()

        dir_names = [
            "pds_data",
            "cal_data",
            "georef_data"
        ]

        for i in dir_names:
            Path(self.root, i).mkdir()

        shutil.copyfile(
            urls_file,
            Path(self.root, "pds_data", f"{self.data_ID_long}_urls.txt")
        )
        shutil.move(
            urls_file,
            Path(self.root, "cal_data", f"{self.data_ID_long}_urls.txt")
        )

        self._reset_cache()

    def _reset_cache(self):
        with h5.File(self.cache, "w") as ds:
            ds.attrs["DATA_ID"] = self.data_ID_long
            ds.attrs["ACQUISITION_MODE"] = self.acq_type

    def __str__(self):
        tree_string = (
            f"{self.root}\n"
            f"\u251c\u2500\u2500\u2500{self.pds_dir}"
            f"\u251c\u2500\u2500\u2500{self.cal_dir}"
        )
        return tree_string
