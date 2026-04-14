from dataclasses import dataclass, field
from typing import Self, TypeVar, Generic

from cubio.geotools.models import BoundingBoxModel
from pyresample.geometry import SwathDefinition, AreaDefinition
from pyresample import kd_tree
import numpy as np
import xarray as xr

MOON_RADIUS = 1737400


@dataclass
class PixelResolution:
    pix_per_deg: float
    deg_per_pix: float
    m_per_pix: float  # Estimation

    @classmethod
    def from_array(
        cls,
        max_lat: float,
        min_lat: float,
        height: int,
        radius_of_body: float = MOON_RADIUS,
    ) -> Self:
        lat_height = max_lat - min_lat
        dpp: float = lat_height / height  # Degrees per pixel
        return cls(
            pix_per_deg=1 / dpp,
            deg_per_pix=dpp,
            m_per_pix=radius_of_body * dpp,
        )


def _define_gridded_area(
    gridded_data: np.ndarray, gridded_data_bbox: BoundingBoxModel, prj: str
) -> AreaDefinition:
    return AreaDefinition(
        area_id="gridded_input",
        description="Data input for the `resample_gridded_data()` method.",
        proj_id="proj_same_as_area",
        projection=prj,
        width=gridded_data.shape[1],
        height=gridded_data.shape[0],
        area_extent=gridded_data_bbox.as_extent(),
    )


@dataclass
class Window:
    row: slice = slice(0, None)
    col: slice = slice(0, None)

    def apply_to(self, data: np.ndarray) -> np.ndarray:
        return data[self.row, self.col]


@dataclass
class GeoreferencingGeometry:
    _swath: SwathDefinition | None = None
    _swath_window: Window = field(default_factory=Window)
    _area: AreaDefinition | None = None

    @property
    def swath(self) -> SwathDefinition:
        if self._swath is None:
            raise ValueError("Swath has not been set.")
        return self._swath

    @swath.setter
    def swath(self, val: SwathDefinition) -> None:
        self._swath = val

    @property
    def swath_window(self) -> Window:
        if self._swath_window is None:
            raise ValueError("Swath has not been set.")
        return self._swath_window

    def set_swath_window(self, row_slice: slice, column_slice: slice) -> None:
        self._swath_window.row = row_slice
        self._swath_window.col = column_slice

    @property
    def area(self) -> AreaDefinition:
        if self._area is None:
            raise ValueError("Area has not been set.")
        return self._area

    @area.setter
    def area(self, val: AreaDefinition) -> None:
        self._area = val

    def get_radius_of_influence(
        self, radius_of_body: float = MOON_RADIUS
    ) -> float:
        m_per_degree = np.pi * radius_of_body / 180
        return (
            5
            * m_per_degree
            * (self.area.pixel_size_x + self.area.pixel_size_y)
            / 2
        )

    def swath_to_gridded_data(
        self,
        data: np.ndarray,
        radius_of_body: float = MOON_RADIUS,
        apply_swath_window: bool = True,
    ) -> np.ndarray:
        if apply_swath_window:
            data = self.swath_window.apply_to(data)
        if self.swath.shape[:2] != data.shape[:2]:
            print(self.swath.shape[:2], data.shape[:2])
            raise ValueError("Swath and data do not match. Set swath window?")
        return kd_tree.resample_nearest(
            self.swath,
            data,
            self.area,
            self.get_radius_of_influence(radius_of_body),
            epsilon=0.5,  # type: ignore
            fill_value=np.nan,  # type: ignore
        )

    def resample_gridded_data(
        self,
        gridded_data: np.ndarray,
        gridded_data_bbox: BoundingBoxModel,
        radius_of_body: float = MOON_RADIUS,
    ) -> np.ndarray:
        input_area = _define_gridded_area(
            gridded_data, gridded_data_bbox, self.area.crs_wkt
        )
        roi = self.get_radius_of_influence(radius_of_body)
        resamp = kd_tree.resample_nearest(
            input_area,
            gridded_data,
            self.area,
            roi,
            epsilon=0.5,  # type: ignore
            fill_value=np.nan,  # type: ignore
        )
        if not isinstance(resamp, np.ndarray):
            raise RuntimeError("Invalid resampling return type.")
        return resamp

    def gridded_data_to_swath(
        self,
        gridded_data: np.ndarray,
        gridded_data_bbox: BoundingBoxModel,
        radius_of_body: float = MOON_RADIUS,
    ) -> np.ndarray:
        input_area = _define_gridded_area(
            gridded_data, gridded_data_bbox, self.area.crs_wkt
        )
        return kd_tree.resample_nearest(
            input_area,
            gridded_data,
            self.swath,
            self.get_radius_of_influence(radius_of_body),
            epsilon=0.5,  # type: ignore
            fill_value=np.nan,  # type: ignore
        )


ARR = TypeVar("ARR", bound=np.ndarray | xr.DataArray)


@dataclass
class CropResult(Generic[ARR]):
    cropped_data: ARR
    longitudes: ARR
    latitudes: ARR
