from dataclasses import dataclass
from typing import TypeVar, Generic, TypedDict
from pathlib import Path

from cubio import cubedata_from_json_file
import numpy as np
import xarray as xr

from pym3tools2.data_retrieval import M3DataPaths
from pym3tools2.constants import DEG2RAD
from .data_transfer_classes import GeoreferencingGeometry


A = TypeVar("A", bound=np.ndarray | xr.DataArray)


class ArrayShape(TypedDict):
    ncols: int
    nrows: int
    nbands: int


@dataclass
class M3Geometry(Generic[A]):
    solaz: A
    solze: A
    m3azi: A
    m3zen: A
    slope: A
    aspct: A
    in_radians: bool = False
    georeferenced: bool = False
    _shape: ArrayShape | None = None
    _photometry_cube: np.ndarray | None = None

    @property
    def geom_dict(self) -> dict[str, A]:
        return {
            "solaz": self.solaz,
            "solze": self.solze,
            "m3azi": self.m3azi,
            "m3zen": self.m3zen,
            "slope": self.slope,
            "aspct": self.aspct,
        }

    @property
    def cube(self) -> np.ndarray:
        return np.stack(
            [np.array(i) for i in self.geom_dict.values()], axis=-1
        )

    @property
    def shape(self) -> ArrayShape:
        if self._shape is None:
            raise RuntimeError("Shape is not set.")
        return self._shape

    @shape.setter
    def shape(self, value: ArrayShape) -> None:
        self._shape = value

    @property
    def photometry_cube(self) -> np.ndarray:
        """A cube with bands (i, e, g)."""
        if self._photometry_cube is None:
            self._photometry_cube = np.stack(
                [self.calc_i(), self.calc_e(), self.calc_g()], axis=-1
            )
        return self._photometry_cube

    def convert_to_radians(self) -> None:
        for k, v in self.geom_dict.items():
            setattr(self, k, DEG2RAD * v)
        self.in_radians = True

    def calc_i(self) -> np.ndarray:
        if not self.in_radians:
            raise RuntimeError("Geometry in degrees, not radians.")
        arg = np.cos(self.solze) * np.cos(self.slope) + np.sin(
            self.solze
        ) * np.sin(self.slope) * np.cos(self.solaz - self.aspct)
        incidence_angle = (180 / np.pi) * np.acos(arg)
        return incidence_angle

    def calc_e(self) -> np.ndarray:
        if not self.in_radians:
            raise RuntimeError("Geometry in degrees, not radians.")
        arg = np.cos(self.m3zen) * np.cos(self.slope) + np.sin(
            self.m3zen
        ) * np.sin(self.slope) * np.cos(self.m3azi - self.aspct)
        emission_angle = (180 / np.pi) * np.acos(arg)
        return emission_angle

    def calc_g(self) -> np.ndarray:
        if not self.in_radians:
            raise RuntimeError("Geometry in degrees, not radians.")
        arg = np.cos(self.m3zen) * np.cos(self.solze) + np.sin(
            self.m3zen
        ) * np.sin(self.solze) * np.cos(self.solaz - self.m3azi)
        phase_angle = (180 / np.pi) * np.acos(arg)
        return phase_angle


def crop_m3geom(
    m3geom: M3Geometry[np.ndarray], rowslice: slice, colslice: slice
) -> M3Geometry[np.ndarray]:
    builder = {
        "solaz": m3geom.solaz[rowslice, colslice],
        "solze": m3geom.solze[rowslice, colslice],
        "m3azi": m3geom.m3azi[rowslice, colslice],
        "m3zen": m3geom.m3zen[rowslice, colslice],
        "slope": m3geom.slope[rowslice, colslice],
        "aspct": m3geom.aspct[rowslice, colslice],
    }
    return M3Geometry[np.ndarray](
        builder["solaz"],
        builder["solze"],
        builder["m3azi"],
        builder["m3zen"],
        builder["slope"],
        builder["aspct"],
    )


def resample_m3geom(
    m3geom: M3Geometry, georef_geometry: GeoreferencingGeometry
) -> M3Geometry[np.ndarray]:
    builder = {
        "solaz": m3geom.solaz,
        "solze": m3geom.solze,
        "m3azi": m3geom.m3azi,
        "m3zen": m3geom.m3zen,
        "slope": m3geom.slope,
        "aspct": m3geom.aspct,
    }
    for k, v in m3geom.geom_dict.items():
        resamp = georef_geometry.swath_to_gridded_data(np.array(v))
        builder[k] = resamp
    new_m3geom = M3Geometry[np.ndarray](**builder)
    new_m3geom.in_radians = m3geom.in_radians
    new_m3geom.georeferenced = True
    new_m3geom._shape = m3geom._shape
    return new_m3geom


def xarray_m3geom_to_numpy(
    m3geom: M3Geometry[xr.DataArray],
) -> M3Geometry[np.ndarray]:
    new_m3geom = M3Geometry[np.ndarray](
        m3geom.solaz.values,
        m3geom.solze.values,
        m3geom.m3azi.values,
        m3geom.m3zen.values,
        m3geom.slope.values,
        m3geom.aspct.values,
    )
    new_m3geom.in_radians = m3geom.in_radians
    new_m3geom.georeferenced = m3geom.georeferenced
    new_m3geom._shape = m3geom._shape
    return new_m3geom


def load_sphere_geometry_data(
    catalog: M3DataPaths, georef_geom: GeoreferencingGeometry | None = None
) -> M3Geometry[np.ndarray]:
    _, obs_cube = cubedata_from_json_file(catalog.obs.json)
    obs_cube.transpose_to("BIP")
    solar_azimuth = obs_cube.array[:, :, 0]
    solar_zenith = obs_cube.array[:, :, 1]
    m3_azimuth = obs_cube.array[:, :, 2]
    m3_zenith = obs_cube.array[:, :, 3]
    slope = obs_cube.array[:, :, 7]
    aspect = obs_cube.array[:, :, 8]

    m3geom = M3Geometry[xr.DataArray](
        solar_azimuth, solar_zenith, m3_azimuth, m3_zenith, slope, aspect
    )
    m3geom.shape = ArrayShape(
        ncols=obs_cube.shape.ncolumns, nrows=obs_cube.shape.nrows, nbands=9
    )
    if georef_geom is not None:
        return resample_m3geom(m3geom, georef_geom)
    return xarray_m3geom_to_numpy(m3geom)


def replace_terrain(
    m3geom: M3Geometry[np.ndarray],
    georef_geom: GeoreferencingGeometry,
    slope_json_path: Path | str,
    aspect_json_path: Path | str,
) -> M3Geometry[np.ndarray]:
    _, new_slp = cubedata_from_json_file(slope_json_path)
    _, new_asp = cubedata_from_json_file(aspect_json_path)
    new_slp.transpose_to("BIP")
    new_asp.transpose_to("BIP")

    new_slp.set_masking(True)
    new_asp.set_masking(True)

    new_slp.add_nodata_mask()
    new_asp.add_nodata_mask()

    conversion = DEG2RAD if m3geom.in_radians else 1

    slope_args = (new_slp.array.values[:, :, 0] * conversion, new_slp.bounds)
    aspect_args = (new_asp.array.values[:, :, 0] * conversion, new_asp.bounds)

    if m3geom.georeferenced:
        print("resampling gridded data...")
        m3geom.slope = georef_geom.resample_gridded_data(*slope_args)
        m3geom.aspct = georef_geom.resample_gridded_data(*aspect_args)
    else:
        print("gridded data to swath...")
        m3geom.slope = georef_geom.gridded_data_to_swath(*slope_args)
        m3geom.aspct = georef_geom.gridded_data_to_swath(*aspect_args)

    return m3geom
