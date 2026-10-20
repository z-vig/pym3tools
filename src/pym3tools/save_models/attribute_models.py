"""
Attribute models for the HDf5 compliant pipeline cache
"""

from typing import Literal, TypedDict

from pym3tools.types import (
    DestripingMethod,
    ThermalCorrectionMethod,
    TopoCorrectionMethod,
)


class StandardDatasetAttrs(TypedDict):
    """
    Attributes for a standard 3D spectral cube dataset.
    """

    timestamp: str
    ncols: int
    nrows: int
    nbands: int
    measvals: list[float]
    bandlbls: list[str]
    geotransform: tuple[float, float, float, float, float, float]
    crs: str


class OneDimensionDatasetAttrs(TypedDict):
    """
    Attributes for a 1D dataset such as wavelength values or correction factors.
    """

    timestamp: str
    length: int
    measvals: list[float]


class DestripedAttrs(TypedDict):
    """
    Attributes for data after the destriping step.
    """

    timestamp: str
    flags: int
    method: DestripingMethod


class CroppedAttrs(TypedDict):
    """
    Attributes related to data cropping.
    """

    timestamp: str
    flags: int
    left_bound: float
    bottom_bound: float
    right_bound: float
    top_bound: float
    height: int
    width: int


class GeoreferencedAttrs(TypedDict):
    """
    Attributes related to georeferencing using ground control points (GCPs).
    """

    timestamp: str
    flags: int
    ngcps: int
    center: tuple[float, float]
    geotransform: tuple[float, float, float, float, float, float]
    crs: str


class SolarRemovedAttrs(TypedDict):
    """
    Attributes after solar spectrum removal.
    """

    timestamp: str
    flags: int
    solar_distance: float


class StatPolishedAttrs(TypedDict):
    """
    Attributes following statistical polishing.
    """

    timestamp: str
    flags: int
    instrument_state: Literal["Warm", "Cold"]


class ThermalCorrectedAttrs(TypedDict):
    """
    Attributes after applying a thermal correction to the data.
    """

    timestamp: str
    flags: int
    method: ThermalCorrectionMethod


class PhotometricCorrectedAttrs(TypedDict):
    """
    Attrbiutes after photometric correction.
    """

    timestamp: str
    flags: int
    topography_correction: TopoCorrectionMethod
