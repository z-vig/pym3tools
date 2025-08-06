# Standard Libraries
import os
from pathlib import Path
import tempfile as tf

# Deendencies
import rasterio as rio  # type: ignore
import numpy as np

# Relative Imports
from .step import Step, PipelineState
from .utils.terrain_model_utils import calc_i, calc_e, calc_g, M3Geometry

# Top-Level Imports
from m3py.selenography.basic_pixel_alignment import align_pixels

PathLike = str | os.PathLike | Path


class TerrainModel(Step):
    def __init__(
        self,
        name: str,
        slope_path: PathLike,
        aspect_path: PathLike,
        use_PDS_terrain_model: bool = False,
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        self.slope_path = slope_path
        self.aspect_path = aspect_path
        self._use_PDS = use_PDS_terrain_model

    def _compute_obs_from_aligned_terrain(
        self,
        state: PipelineState
    ) -> np.ndarray:
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

        new_obs = np.concat([
                incidence_map[:, :, np.newaxis],
                emission_map[:, :, np.newaxis],
                phase_map[:, :, np.newaxis],
                slope_map[:, :, np.newaxis],
                aspect_map[:, :, np.newaxis]
            ], axis=2)

        return new_obs

    def run(self, state: PipelineState) -> PipelineState:

        if self._use_PDS:
            new_obs = np.empty((*state.data.shape[:2], 5))
        else:
            new_obs = self._compute_obs_from_aligned_terrain(state)

        new_state = PipelineState(
            data=state.data,
            wvl=state.wvl,
            obs=new_obs,
            georef=state.georef
        )

        return new_state
