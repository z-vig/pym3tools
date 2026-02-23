# Standard Libraries
import os
from pathlib import Path

# Dependencies
from rasterio.coords import BoundingBox  # type: ignore
import h5py as h5  # type: ignore

# Relative Imports
from .step import Step, PipelineState, StepCompletionState

# Top-Level Imports
from pym3tools.io.read_m3_binary import read_m3
from pym3tools.formats.m3_data_format import LOC
from pym3tools.selenography.crop import regional_crop

PathLike = str | os.PathLike | Path


class Crop(Step):
    def __init__(
        self,
        name: str,
        bbox: BoundingBox | None = None,
        offsets: tuple[int, int, int, int] | None = None,
        **kwargs,
    ):
        """
        Cropping Step for the M3 L2 pipeline.

        Parameters
        ----------
        name : str
            Name of the step.
        bbox : BoundingBox | None, optional
            Bounding box for the crop. If this is not provided, image offset
            must be provided.
        offsets : tuple[int, int, int, int] | None, optional
            Image offset. If provided, these numbers take precedence for the
            crop operation. Tuple = (row_offset, column_offset, height, width).
        """
        super().__init__(name, **kwargs)
        self.bbox = bbox
        self.offsets = offsets

    def run(self, state: PipelineState) -> PipelineState:
        loc_arr = read_m3(
            self.manager.pds_dir.l1.loc_img, LOC, self.manager.acq_type
        )

        if self.offsets is not None:
            row_offset, col_offset, height, width = self.offsets
            rowslice = slice(row_offset, row_offset + height)
            colslice = slice(col_offset, col_offset + width)
            cropped_data = state.data[rowslice, colslice, :]
            loc_arr_crop = loc_arr[rowslice, colslice, :]
            state.georef.left_bound = float(loc_arr_crop[0, 0, 0])
            state.georef.bottom_bound = float(loc_arr_crop[-1, -1, 1])
            state.georef.right_bound = float(loc_arr_crop[-1, -1, 0])
            state.georef.top_bound = float(loc_arr_crop[0, 0, 1])
        else:
            if self.bbox is None:
                raise ValueError(
                    "If image ofsets are not provided, a bounding box must be"
                    "set."
                )
            cropped_data, row_offset, col_offset, height, width = (
                regional_crop(state.data, loc_arr, self.bbox)
            )
            state.georef.left_bound = self.bbox.left
            state.georef.bottom_bound = self.bbox.bottom
            state.georef.right_bound = self.bbox.right
            state.georef.top_bound = self.bbox.top

        state.georef.row_offset = int(row_offset)
        state.georef.col_offset = int(col_offset)
        state.georef.height = int(height)
        state.georef.width = int(width)

        self._new_georef = state.georef
        new_flags = state.flags
        new_flags.cropped = StepCompletionState.Complete

        new_state = PipelineState(
            data=cropped_data,
            wvl=state.wvl,
            bbl=state.bbl,
            obs=state.obs[
                row_offset : row_offset + height,  # noqa
                col_offset : col_offset + width,  # noqa
                :,
            ],
            georef=state.georef,
            flags=new_flags,
        )

        return new_state

    def save(self, output: PipelineState) -> None:
        super().save(output)
        with h5.File(self.manager.cache, "r+") as f:
            g = f[self.name]
            assert isinstance(g, h5.Group)
            g.attrs["bbox"] = self._new_georef.bbox_to_list()
            g.attrs["window"] = self._new_georef.window_to_list()
