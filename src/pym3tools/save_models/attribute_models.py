from typing import TypedDict, Literal
from pym3tools2.types import ThermalCorrectionMethod, TopoCorrectionMethod


class StandardDatasetAttrs(TypedDict):
    timestamp: str
    ncols: int
    nrows: int
    nbands: int
    measvals: list[float]
    bandlbls: list[str]
    geotransform: tuple[float, float, float, float, float, float]
    crs: str


class OneDimensionDatasetAttrs(TypedDict):
    timestamp: str
    length: int
    measvals: list[float]


class CroppedAttrs(TypedDict):
    timestamp: str
    flags: int
    left_bound: float
    bottom_bound: float
    right_bound: float
    top_bound: float
    height: int
    width: int


class GeoreferencedAttrs(TypedDict):
    timestamp: str
    flags: int
    ngcps: int
    center: tuple[float, float]
    geotransform: tuple[float, float, float, float, float, float]
    crs: str


class SolarRemovedAttrs(TypedDict):
    timestamp: str
    flags: int
    solar_distance: float


class StatPolishedAttrs(TypedDict):
    timestamp: str
    flags: int
    instrument_state: Literal["Warm", "Cold"]


class ThermalCorrectedAttrs(TypedDict):
    timestamp: str
    flags: int
    method: ThermalCorrectionMethod


class PhotometricCorrectedAttrs(TypedDict):
    timestamp: str
    flags: int
    topography_correction: TopoCorrectionMethod
