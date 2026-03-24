# Standard Libraries
import os
from pathlib import Path

# Dependencies
import yaml
from cubio.geotools.models import ImageOffset
from cubio.geotools.georeference_from_gcps import georeference_image
from cubio.geotools.georeference_satellite_swath import ProjectionDefinition
from cubio.data.crs_wkt_strings import GeographicCRS

# Relative Imports
from .step import Step, PipelineState, StepCompletionState

# Top-Level Imports
from pym3tools.metadata_models import AffineDict

PathLike = str | os.PathLike | Path


class AnalysisScopeError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Georeference(Step):
    def __init__(self, name: str, verbose: bool = True, **kwargs):
        super().__init__(name, **kwargs)
        self._verbose = verbose

    def run(self, state: PipelineState) -> PipelineState:
        if self.manager.analysis_scope.value == "global":
            raise NotImplementedError("You gotta have gcps right now.")

        prj4 = ProjectionDefinition(
            "GruithuisenRegion",
            "LunarGeographic",
            "GCS coordinates for the Moon",
            proj4_str="+proj=longlat +R=1737400 +no_defs +type=crs",
            crs_wkt_str=GeographicCRS.GCS_MOON_2000,
        )
        georef_rdn, _ = georeference_image(
            cubio_json_file=Path(self.manager.pds_dir.l1.rdn_img).with_suffix(
                ".json"
            ),
            gcps_file=Path(self.manager.georef_dir.gcps),
            prj_definition=prj4,
            unref_cube_array=state.data,
            new_gcps_offset=ImageOffset(
                row=state.georef.row_offset,
                column=state.georef.col_offset,
                height=state.data.shape[0],
                width=state.data.shape[1],
            ),
            apply_cropping=False,
        )
        georef_obs, georef_gtrans_obs = georeference_image(
            cubio_json_file=Path(self.manager.pds_dir.l1.obs_img).with_suffix(
                ".json"
            ),
            gcps_file=Path(self.manager.georef_dir.gcps),
            prj_definition=prj4,
            unref_cube_array=state.obs,
            new_gcps_offset=ImageOffset(
                row=state.georef.row_offset,
                column=state.georef.col_offset,
                height=state.data.shape[0],
                width=state.data.shape[1],
            ),
            apply_cropping=False,
        )

        print("GEOREF SHAPE:", georef_rdn.shape)

        state.georef.crs = str(prj4.crs_wkt_str)
        _affine = georef_gtrans_obs.toaffine()
        state.georef.geotransform = AffineDict(
            a=_affine.a,
            b=_affine.b,
            c=_affine.c,
            d=_affine.d,
            e=_affine.e,
            f=_affine.f,
        )
        new_bounds = georef_gtrans_obs.get_bbox(
            height=georef_obs.shape[0], width=georef_obs.shape[1]
        )
        state.georef.top_bound = new_bounds.top
        state.georef.bottom_bound = new_bounds.bottom
        state.georef.left_bound = new_bounds.left
        state.georef.right_bound = new_bounds.right

        print(
            "TEST BBOX: ",
            state.georef.left_bound,
            state.georef.bottom_bound,
            state.georef.right_bound,
            state.georef.top_bound,
        )

        new_flags = state.flags
        new_flags.georeferenced = StepCompletionState.Complete

        new_state = PipelineState(
            data=georef_rdn,
            wvl=state.wvl,
            bbl=state.bbl,
            obs=georef_obs,
            georef=state.georef,
            flags=new_flags,
        )

        return new_state

    def save(self, output: PipelineState) -> None:
        super().save(output)
        with open(self.manager.georef_dir.metageo, "w") as f:
            yaml.dump(output.georef.model_dump(), f)
