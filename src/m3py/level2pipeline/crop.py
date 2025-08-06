# Standard Libraries
import os
from pathlib import Path

# Deendencies
from rasterio.coords import BoundingBox  # type: ignore

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
            self.manager.pds_dir.l1.loc_img,
            LOC,
            self.manager.acq_type
        )

        cropped_data, row_offset, col_offset, height, width =\
            regional_crop(state.data, loc_arr, self.bbox)

        state.georef.row_offset = int(row_offset)
        state.georef.col_offset = int(col_offset)
        state.georef.height = int(height)
        state.georef.width = int(width)

        new_state = PipelineState(
            data=cropped_data,
            wvl=state.wvl,
            obs=state.obs[
                row_offset:row_offset+height,
                col_offset:col_offset+width,
                :
            ],
            georef=state.georef
        )

        return new_state
