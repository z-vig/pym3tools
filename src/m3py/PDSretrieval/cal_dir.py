# Standard Libraries
from pathlib import Path
import os
import re

# Relative Imports
from .file_retrieval_patterns import FileRetrievalPatterns
from .retrieve_urls import retrieve_urls


class CalDir:
    def __init__(self, parent: os.PathLike, data_id: str):
        self.root = Path(parent)
        self.retrieval = Path(self.root, f"{data_id}_urls.txt")
        self.acq_type = re.findall(FileRetrievalPatterns.acq_type, data_id)[0]

        with open(self.retrieval, "r") as f:
            fread = f.read()
            pattern = FileRetrievalPatterns.global_caldata \
                if self.acq_type == "G" \
                else FileRetrievalPatterns.targeted_caldata
            urls = re.findall(pattern, fread)

        attr_dict = {
            "SOLAR_SPEC": "solar_spectrum",
            "STAT_POL_1": "statistical_polish1",
            "STAT_POL_2": "statistical_polish2",
            "F_ALPHA_HIL": "phase_function",
            "GRND_TRU_1": "ground_truth1",
            "GRND_TRU_2": "ground_truth2"
        }

        for i in urls:
            caltype = Path(i).stem[Path(i).stem.find("_", 13) + 1:]
            if Path(i).suffix == ".TAB":
                setattr(
                    self, attr_dict[caltype], Path(self.root, Path(i).name)
                )

        file_path_dict = {
            i: Path(self.root, Path(i).name)
            for i in urls
            if not Path(self.root, Path(i).name).is_file()
        }  # This contains only files that do not exist yet.

        print(f"{len(file_path_dict)} calibration files will be downloaded.")
        retrieve_urls(file_path_dict)

    def __str__(self):
        tree_string = (
            f"{self.root.name}\n"
        )
        for k, v in vars(self).items():
            if k not in ("root", "acq_type"):
                tree_string += "\u2502   \u251c\u2500\u2500\u2500" \
                               f"{v}\n"
        return tree_string
