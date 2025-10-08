# Standard Libraries
from typing import Optional, Any
from pathlib import Path

# Dependencies
from rasterio.coords import BoundingBox  # type: ignore
from pydantic import BaseModel

# Top-Level Imports
from pym3tools.types import PathLike
from pym3tools.PDSretrieval.file_manager import M3FileManager

# Relative Imports
from .main_pipeline import M3Level2Pipeline
from .crop import Crop
from .georeference import Georeference
from .terrain_model import TerrainModel
from .solar_spectrum_removal import SolarSpectrumRemoval
from .statistical_polish import StatisticalPolish
from .clark_thermal_correction import ClarkThermalCorrection


class BaseConfig(BaseModel):
    pipeline_type: str
    root: str
    data_ID: str
    parameters: dict[str, Any]


# ====================
#   Regional Pipelines
# ====================
def run_regional_pipeline(
    root_path: PathLike,
    data_ID: str,
    bbox: BoundingBox,
    slope_path: Optional[PathLike] = None,
    aspect_path: Optional[PathLike] = None,
    cache_name: Optional[str] = None,
) -> None:
    """
    Will run the regional M3 pipeline. The steps include:
    - Cropping M3 stamp to a known region.
    - Georeferencing this data using User-Defined Ground Control Points.
    - Creating a new terrain model using User-Defined slope and aspect maps.
    - Applies I/F solar spectrum reflectance correction.
    - Applies standard Statistical Polishing
    - Applies the Clark et al., 2010 Thermal Correction

    Parameters
    ----------
    root_path: PathLike
        Path to root directory of M3 stamp data.
    data_ID: str
        M3 stamp ID.
    bbox: BoundingBox
        Rasterio bounding box.
    slope_path: PathLike
        Path to custom slope data.
    aspect_path: PathLike
        Path to custom aspect data.
    cache_name: str
        Name of the pipeline cache save file.
    """
    if cache_name is None:
        cache = "pipeline_cache.hdf5"
    else:
        cache = Path(cache_name).with_suffix(".hdf5").__str__()

    m3_manager = M3FileManager(
        root=root_path,
        data_id=data_ID,
        cache_name=cache,
        reset_cache=True,
    )

    steps = [
        Crop(
            "cropped_data",
            bbox,
            save_output=True,
        ),
        Georeference("georeference", save_output=True),
        TerrainModel(
            "terrain_model", slope_path, aspect_path, save_output=True
        ),
        SolarSpectrumRemoval("solar_spectrum_removal", save_output=True),
        StatisticalPolish("statistical_polishing", save_output=True),
        ClarkThermalCorrection(
            "thermal_correction", max_iterations=3, save_output=True
        ),
    ]

    pipeline = M3Level2Pipeline(steps, m3_manager)
    pipeline.run()


class RegionalConfig(BaseConfig):
    left_bound: float
    bottom_bound: float
    right_bound: float
    top_bound: float
    slope_data_path: Optional[str]
    aspect_data_path: Optional[str]

    @classmethod
    def from_base(cls, cfg: BaseConfig) -> "RegionalConfig":
        return cls(
            pipeline_type=cfg.pipeline_type,
            root=cfg.root,
            data_ID=cfg.data_ID,
            parameters=cfg.parameters,
            **cfg.parameters,
        )


def _run_regional(params: RegionalConfig) -> None:
    bbox = BoundingBox(
        params.left_bound,
        params.bottom_bound,
        params.right_bound,
        params.top_bound,
    )
    run_regional_pipeline(
        params.root,
        params.data_ID,
        bbox,
        params.slope_data_path,
        params.aspect_data_path,
        "pipeline_cache.hdf5",
    )


def _run_regional_pds_terrain(params: RegionalConfig) -> None:
    bbox = BoundingBox(
        params.left_bound,
        params.bottom_bound,
        params.right_bound,
        params.top_bound,
    )
    run_regional_pipeline(
        params.root,
        params.data_ID,
        bbox,
        cache_name="pipeline_cache_pds_terrain.hdf5",
    )


# ==================
#   Global Pipelines
# ==================
# TBW
