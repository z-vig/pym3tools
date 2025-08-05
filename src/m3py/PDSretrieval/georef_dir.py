# Standard Libraries
import os
from pathlib import Path

# Dependencies
import yaml

# Top-Level Imports
from m3py.selenography.gcp_loaders import read_gcps
from m3py.metadata_models import GeorefData


PathLike = str | os.PathLike | Path


class GroundControlPointsNotFoundError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GroundControlPointsEmptyError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GeorefDir():
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
            raise GroundControlPointsNotFoundError(
                f"Ground Control Points have not been created at {self.gcps}."
                " Use the georeferencer to create them."
            )
        elif self.gcps.is_file():
            gcp_list = read_gcps(self.gcps)
            if len(gcp_list) == 0:
                raise GroundControlPointsEmptyError(
                    f"{self.gcps} contains no Ground Control Points."
                )
