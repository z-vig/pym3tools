# Standard Libraries
import os
from pathlib import Path
import shutil
import re

# Relative Imports
from .pds_dir import PDSDir
from .cal_dir import CalDir
from .file_retrieval_patterns import FileRetrievalPatterns


class M3FileConfig:
    """
    Stores data for M3 file locations for a single M3 stripe.

    Parameters
    ----------
    root: str | os.PathLike
        Path to root directory holding multiple M3 datasets.
    data_id: str
        Data ID of M3 stripe. This should correspond to the name of a directory
        within the root directory.
    """
    def __init__(self, root: str | os.PathLike, data_id: str):
        self.root = Path(root, data_id)
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

        self.pds_dir = PDSDir(Path(self.root, "pds_data"), self.data_ID_long)
        self.cal_dir = CalDir(Path(self.root, "cal_data"), self.data_ID_long)

    def _initialize_directories(self, urls_file: os.PathLike):
        self.root.mkdir()

        dir_names = [
            "pds_data",
            "cal_data",
            "hdf5_files",
            "georeferenced_data"
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

    def _validate(self):
        if not self.root.is_dir():
            self.root.mkdir()

    def __str__(self):
        tree_string = (
            f"{self.root}\n"
            f"\u251c\u2500\u2500\u2500{self.pds_dir}"
            f"\u251c\u2500\u2500\u2500{self.cal_dir}"
        )
        return tree_string
