# Standard Libraries
import os
from pathlib import Path

# Dependencies
from rasterio.coords import BoundingBox  # type: ignore
import h5py as h5  # type: ignore

# Relative Imports
from .step import Step, PipelineState

# Top-Level Imports
from m3py.io.read_m3 import read_m3
from m3py.formats.m3_data_format import LOC
from m3py.selenography.crop import regional_crop

PathLike = str | os.PathLike | Path


class Crop(Step):
    def __init__(self, name: str, bbox: BoundingBox, **kwargs):
        super().__init__(name, **kwargs)
        self.bbox = bbox

    def run(self, state: PipelineState) -> PipelineState:
        loc_arr = read_m3(
            self.manager.pds_dir.l1.loc_img, LOC, self.manager.acq_type
        )

        cropped_data, row_offset, col_offset, height, width = regional_crop(
            state.data, loc_arr, self.bbox
        )

        state.georef.row_offset = int(row_offset)
        state.georef.col_offset = int(col_offset)
        state.georef.height = int(height)
        state.georef.width = int(width)

        self._new_georef = state.georef

        new_state = PipelineState(
            data=cropped_data,
            wvl=state.wvl,
            obs=state.obs[
                row_offset : row_offset + height,  # noqa
                col_offset : col_offset + width,  # noqa
                :,
            ],
            georef=state.georef,
        )

        return new_state

    def save(self, output: PipelineState) -> None:
        super().save(output)
        with h5.File(self.manager.cache, "r+") as f:
            g = f[self.name]
            assert isinstance(g, h5.Group)
            g.attrs["bbox"] = [
                self.bbox.left,
                self.bbox.bottom,
                self.bbox.right,
                self.bbox.top,
            ]
            g.attrs["window"] = self._new_georef.to_list()
