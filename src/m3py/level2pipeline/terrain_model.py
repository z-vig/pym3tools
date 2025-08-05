# Standard Libraries
import os
from pathlib import Path
import tempfile as tf
from dataclasses import dataclass

# Deendencies
import rasterio as rio  # type: ignore
import numpy as np

# Relative Imports
from .step import Step, PipelineState

# Top-Level Imports
from m3py.selenography.basic_pixel_alignment import align_pixels

PathLike = str | os.PathLike | Path


class AngularUnitsError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class M3Geometry:
    solaz: np.ndarray
    solze: np.ndarray
    m3az: np.ndarray
    m3ze: np.ndarray
    phase: np.ndarray
    solen: np.ndarray
    m3len: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    cosi: np.ndarray
    radians: bool

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "M3Geometry":
        obs[obs == -999] = np.nan
        keys = list(cls.__dataclass_fields__.keys())
        values = {k: obs[:, :, i] for i, k in enumerate(keys[:-1])}
        return cls(**values, radians=False)

    def convert_to_rad(self):
        keys = list(self.__dataclass_fields__.keys())[:-1]
        deg_to_rad = np.pi/180
        [setattr(self, k, deg_to_rad * getattr(self, k)) for k in keys]
        self.radians = True


def calc_i(geo: M3Geometry) -> np.ndarray:
    if not geo.radians:
        raise AngularUnitsError(
            "Before calculating incidence angle, geo must be in Radians. Try"
            "running geo.convert_to_rad()"
        )
    arg = np.cos(geo.solze) * np.cos(geo.slope) +\
        np.sin(geo.solze) * np.sin(geo.slope) * np.cos(geo.solaz - geo.aspect)
    incidence_angle = (180/np.pi) * np.acos(arg)
    return incidence_angle


def calc_e(geo: M3Geometry) -> np.ndarray:
    if not geo.radians:
        raise AngularUnitsError(
            "Before calculating emission angle, geo must be in Radians. Try"
            "running geo.convert_to_rad()"
        )
    arg = np.cos(geo.m3ze) * np.cos(geo.slope) +\
        np.sin(geo.m3ze) * np.sin(geo.slope) * np.cos(geo.m3az - geo.aspect)
    emission_angle = (180/np.pi) * np.acos(arg)
    return emission_angle


def calc_g(geo: M3Geometry) -> np.ndarray:
    if not geo.radians:
        raise AngularUnitsError(
            "Before calculating phase angle, geo must be in Radians. Try"
            "running geo.convert_to_rad()"
        )
    arg = np.cos(geo.m3ze) * np.cos(geo.solze) +\
        np.sin(geo.m3ze) * np.sin(geo.solze) * np.cos(geo.solaz - geo.m3az)
    phase_angle = (180/np.pi) * np.acos(arg)
    return phase_angle


class TerrainModel(Step):
    def __init__(
        self,
        name: str,
        slope_path: PathLike,
        aspect_path: PathLike,
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        self.slope_path = slope_path
        self.aspect_path = aspect_path

    def run(self, state: PipelineState) -> PipelineState:
        data_temp = tf.NamedTemporaryFile(suffix=".tif")
        aligned_slope_temp = tf.NamedTemporaryFile(suffix=".tif")
        aligned_aspect_temp = tf.NamedTemporaryFile(suffix=".tif")
        data_temp.close()
        aligned_slope_temp.close()
        aligned_aspect_temp.close()

        profile = {
            "driver": "GTiff",
            "dtype": state.data.dtype,
            "nodata": state.georef.nodata,
            "width": state.data.shape[1],
            "height": state.data.shape[0],
            "count": 1,
            "crs": state.georef.crs,
            "transform": state.georef.geotransform.to_affine()
        }

        with rio.open(data_temp.name, "w", **profile) as ds:
            ds.write(state.data[:, :, 0], 1)

        align_pixels(
            data_temp.name, self.slope_path, dst_path=aligned_slope_temp.name
        )

        align_pixels(
            data_temp.name, self.aspect_path, dst_path=aligned_aspect_temp.name
        )

        with rio.open(aligned_slope_temp.name, "r", driver="GTiff") as ds:
            slope_map = ds.read()
        with rio.open(aligned_aspect_temp.name, "r", driver="GTiff") as ds:
            aspect_map = ds.read()

        slope_map = slope_map[0, :, :]
        aspect_map = aspect_map[0, :, :]

        if slope_map.shape[:2] != state.obs.shape[:2]:
            raise ValueError(f"Shape of the slope map ({slope_map.shape}) "
                             f"does not match obs ({state.obs.shape})")
        if aspect_map.shape[:2] != state.obs.shape[:2]:
            raise ValueError(f"Shape of the slope map ({aspect_map.shape}) "
                             f"does not match obs ({state.obs.shape})")

        m3geom = M3Geometry.from_obs(state.obs)
        m3geom.slope = slope_map
        m3geom.aspect = aspect_map

        m3geom.convert_to_rad()
        incidence_map = calc_i(m3geom)
        emission_map = calc_e(m3geom)
        phase_map = calc_g(m3geom)

        new_state = PipelineState(
            data=state.data,
            wvl=state.wvl,
            obs=np.concat(
                [
                    incidence_map[:, :, np.newaxis],
                    emission_map[:, :, np.newaxis],
                    phase_map[:, :, np.newaxis],
                    slope_map[:, :, np.newaxis],
                    aspect_map[:, :, np.newaxis]
                ],
                axis=2
            ),
            georef=state.georef
        )

        return new_state
