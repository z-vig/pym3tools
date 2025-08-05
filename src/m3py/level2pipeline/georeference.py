# Standard Libraries
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

# Dependencies
import numpy as np
import rasterio as rio  # type: ignore

# Relative Imports
from .step import Step, PipelineState

# Top-Level Imports
from m3py.selenography.gcp_utils import apply_gcps
from m3py.metadata_models import AffineDict

PathLike = str | os.PathLike | Path


class Georeference(Step):
    def __init__(self, name: str, verbose: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self._verbose = verbose

    def run(self, state: PipelineState) -> PipelineState:
        rdn_temp_file = NamedTemporaryFile(suffix=".tif")
        obs_temp_file = NamedTemporaryFile(suffix=".tif")
        rdn_temp_file.close()
        obs_temp_file.close()

        offsets = state.georef.row_offset, state.georef.col_offset

        apply_gcps(
            state.data,
            self.manager.georef_dir.gcps,
            rdn_temp_file.name,
            input_array_offsets=offsets,
            verbose=self._verbose
        )

        apply_gcps(
            state.obs,
            self.manager.georef_dir.gcps,
            obs_temp_file.name,
            input_array_offsets=offsets,
            verbose=self._verbose
        )

        with rio.open(rdn_temp_file.name, "r", driver="GTiff") as ds:
            cropped_rdn = ds.read()
            gtrans = ds.transform
            state.georef.geotransform = AffineDict(
                a=gtrans.a,
                b=gtrans.b,
                c=gtrans.c,
                d=gtrans.d,
                e=gtrans.e,
                f=gtrans.f
            )
            state.georef.crs = ds.crs.to_wkt()

        with rio.open(obs_temp_file.name, "r", driver="GTiff") as ds:
            cropped_obs = ds.read()

        cropped_rdn = np.transpose(cropped_rdn, (1, 2, 0))
        cropped_obs = np.transpose(cropped_obs, (1, 2, 0))

        new_state = PipelineState(
            data=cropped_rdn,
            wvl=state.wvl,
            obs=cropped_obs,
            georef=state.georef
        )

        return new_state
