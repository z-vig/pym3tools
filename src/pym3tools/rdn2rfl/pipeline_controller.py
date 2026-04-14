from pathlib import Path
from typing import Sequence

from cubio import cubedata_from_json_file
import h5py
import numpy as np
from pyresample.geometry import SwathDefinition, AreaDefinition

from pym3tools2.data_retrieval.data_directory import M3DataPaths
from pym3tools2.save_models.pipeline_cache_schema import (
    PipelineCache,
    PIPELINE_SCHEMA,
)
from pym3tools2.rdn2rfl.retrieve_terrain_data import (
    load_sphere_geometry_data,
    replace_terrain,
)
from pym3tools2.constants import MOON_GCS_PRJ

from .step_model import Step, PipelineState


class M3Level2Pipeline:
    """
    Main pipeline controller for L1 to L2 processing of M3 data from NASA's
    Planetary Data System.
    """

    def __init__(
        self,
        steps: Sequence[Step],
        file_catalog: M3DataPaths | Path | str,
        pipeline_cache: Path | str,
        custom_slope: Path | str | None = None,
        custom_aspect: Path | str | None = None,
        overwrite_cache: bool = False,
        crs: str = MOON_GCS_PRJ,
    ) -> None:
        self.steps = steps
        self.catalog: M3DataPaths
        if not isinstance(file_catalog, M3DataPaths):
            self.catalog = M3DataPaths(Path(file_catalog))
        else:
            self.catalog = file_catalog

        rdn_ctxt, rdn_data = cubedata_from_json_file(self.catalog.rdn.json)
        rdn_data.transpose_to("BIP")

        _, loc_cube = cubedata_from_json_file(self.catalog.loc.json)
        loc_cube.transpose_to("BIP")
        longitudes = np.array(loc_cube.array.values[:, :, 0])
        longitudes = ((longitudes + 180) % 360) - 180
        latitudes = loc_cube.array.values[:, :, 1]

        m3geom = load_sphere_geometry_data(self.catalog)
        m3geom.convert_to_radians()

        if Path(pipeline_cache).exists() and not overwrite_cache:
            raise FileExistsError(
                f"Pipeline cache already exists as {pipeline_cache}. To "
                "overwrite this data, set overwrite_cache=True"
            )

        self._open_cache = h5py.File(str(pipeline_cache), "w")
        print("Initializing cache...")
        PIPELINE_SCHEMA.initialize(self._open_cache)
        cache = PipelineCache(self._open_cache)

        for i in self.steps:
            i.cache = cache
            i.catalog = self.catalog

        self.state = PipelineState(
            rdn_data.array, rdn_ctxt.measurement_values, m3geom
        )

        self.state.geom.swath = SwathDefinition(
            lons=longitudes, lats=latitudes
        )

        dset_h, dset_w = latitudes.shape
        self.state.geom.area = AreaDefinition(
            area_id="NullArea",
            description=(
                "Area defined as the height and width of the M3 dataset and"
                " an extent starting at (0, 0) in the upper left"
            ),
            proj_id="default_moon",
            projection=crs,
            width=dset_w,
            height=dset_h,
            area_extent=(
                longitudes[-1, 0],
                latitudes[-1, 0],
                longitudes[0, -1],
                latitudes[0, -1],
            ),
        )

        if (custom_slope is not None) and (custom_aspect is not None):
            print("Using custom slope and aspect maps.")
            self.state.obs = replace_terrain(
                self.state.obs, self.state.geom, custom_slope, custom_aspect
            )

    def run(self) -> PipelineState:
        state = self.state
        for step in self.steps:
            state = step.execute(state)
        self._open_cache.close()
        return state


def process_m3(config_file: Path) -> None:
    config_file = Path(config_file)
    return None
