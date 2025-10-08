# Standard Libraries
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

# Dependencies
import numpy as np
import rasterio as rio  # type: ignore
import yaml

# Relative Imports
from .step import Step, PipelineState, StepCompletionState

# Top-Level Imports
from pym3tools.selenography.gcp_utils import apply_gcps
from pym3tools.selenography.gcp_writer import write_gcp_file_from_loc
from pym3tools.metadata_models import AffineDict

PathLike = str | os.PathLike | Path


class AnalysisScopeError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Georeference(Step):
    def __init__(self, name: str, verbose: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self._verbose = verbose

    def run(self, state: PipelineState) -> PipelineState:
        if self.manager.analysis_scope.value == "global":
            gcp_temp_file = NamedTemporaryFile(suffix=".gcps")
            gcp_temp_file.close()
            with rio.open(self.manager.pds_dir.l1.loc_img) as f:
                loc = np.transpose(f.read(), (1, 2, 0))
            write_gcp_file_from_loc(
                loc,
                gcp_temp_file.name,
                self.manager.pds_dir.l1.rdn_img,
                state.georef.row_offset,
                state.georef.col_offset,
                state.georef.height,
                state.georef.width,
                0,
            )
            self.manager.georef_dir.gcps = gcp_temp_file.name
            # raise AnalysisScopeError(
            #     f"{self.manager.data_ID_long} is in "
            #     f"{self.manager.analysis_scope.name} analysis scope. If "
            #     "georeferencing is to be applied, Ground Control Points must"
            #     "be added."
            # )
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
            verbose=self._verbose,
        )

        apply_gcps(
            state.obs,
            self.manager.georef_dir.gcps,
            obs_temp_file.name,
            input_array_offsets=offsets,
            verbose=self._verbose,
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
                f=gtrans.f,
            )
            state.georef.crs = ds.crs.to_wkt()

        with rio.open(obs_temp_file.name, "r", driver="GTiff") as ds:
            cropped_obs = ds.read()

        cropped_rdn = np.transpose(cropped_rdn, (1, 2, 0))
        cropped_obs = np.transpose(cropped_obs, (1, 2, 0))

        new_flags = state.flags
        new_flags.georeferenced = StepCompletionState.Complete

        new_state = PipelineState(
            data=cropped_rdn,
            wvl=state.wvl,
            obs=cropped_obs,
            georef=state.georef,
            flags=new_flags,
        )

        return new_state

    def save(self, output: PipelineState) -> None:
        super().save(output)
        with open(self.manager.georef_dir.metageo, "w") as f:
            yaml.dump(output.georef.model_dump(), f)
