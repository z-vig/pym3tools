# Standard Libraries
import os
from pathlib import Path
from enum import Enum

# Dependencies
import yaml

# Top-Level Imports
from m3py.selenography.gcp_loaders import read_gcps
from m3py.metadata_models import GeorefData


PathLike = str | os.PathLike | Path


class AnalysisScope(Enum):
    REGIONAL = "regional"
    GLOBAL = "global"


class GroundControlPointsEmptyError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class TooManyGroundControlPointsError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GeorefDir:
    root: PathLike
    gcps: PathLike
    rdn: PathLike
    obs: PathLike
    """
    File manager for the georeferenced data directory.

    Parameters
    ----------
    root: PathLike
        Path to the Georeference Directory root.
    data_ID: str
        Long form of the M3 data ID. M3(G|T)(Year)(Month)(Day)T(Time).
    """

    def __init__(self, root: PathLike, data_ID: str):
        self.root = Path(root)
        self.gcps = Path(self.root, data_ID).with_suffix(".gcps")
        self.rdn = Path(self.root, "rdn.tif")
        self.obs = Path(self.root, "obs.tif")
        self.metageo = Path(self.root, "georeference.yaml")
        with open(self.metageo, "w") as f:
            yaml.dump(GeorefData.empty().model_dump(), f)

        if not self.gcps.is_file():
            self.analysis_scope = AnalysisScope.GLOBAL
            print(
                "Ground Control Points not found, analysis scope set to: "
                f"{self.analysis_scope.name}"
            )
        elif self.gcps.is_file():
            self.analysis_scope = AnalysisScope.REGIONAL
            print(
                f"Ground Control Points found at: {self.gcps}, analysis scope "
                f"set to: {self.analysis_scope.name}"
            )
            gcp_list = read_gcps(self.gcps)
            if len(gcp_list) == 0:
                raise GroundControlPointsEmptyError(
                    f"{self.gcps} contains no Ground Control Points."
                )

            if len(gcp_list) > 100:
                raise TooManyGroundControlPointsError(
                    f"{self.gcps} contains too many Ground Control Points"
                    f" ({len(gcp_list)}) for a command line call to GDAL for"
                    "Windows Powershell."
                )
