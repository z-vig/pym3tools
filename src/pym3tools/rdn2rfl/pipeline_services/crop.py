from cubio.geotools.models import BoundingBoxModel
from cubio import cubedata_from_json_file
import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition

from pym3tools2.save_models.pipeline_cache_schema import PipelineCache
from pym3tools2.rdn2rfl.step_model import PipelineState
from pym3tools2.rdn2rfl.pipeline_state import (
    CompletedFlag,
    get_standard_dset_attrs,
)
from pym3tools2.data_retrieval.data_directory import M3DataPaths
from pym3tools2.rdn2rfl.data_transfer_classes import CropResult
from pym3tools2.array_utils import boolean_to_slice


def crop_data_from_loc_backplane(
    data: np.ndarray | xr.DataArray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    bbox: BoundingBoxModel,
) -> CropResult:
    """
    Crop the input data using the LOC backplane. This method is used when
    GCPs are not provided. It extracts the longitude and latitude values
    from the LOC backplane and crops the input data accordingly.
    """
    latmean: np.ndarray = np.mean(latitudes, axis=1)
    row_idx = (bbox.bottom < latmean) & (bbox.top > latmean)
    row_slice = boolean_to_slice(row_idx)

    lon_vertical_cropped = longitudes[row_slice, :]

    lonmean: np.ndarray = np.mean(lon_vertical_cropped, axis=0)
    col_idx = (bbox.left < lonmean) & (bbox.right > lonmean)
    col_slice = boolean_to_slice(col_idx)

    cropped_arr = data[row_slice, col_slice, :]

    return CropResult[np.ndarray](
        np.array(cropped_arr),
        longitudes[row_slice, col_slice],
        latitudes[row_slice, col_slice],
    )


def load_loc_backplane(catalog: M3DataPaths) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads LOC backplane into memory as latitudes and longitudes.

    Returns
    -------
    longitudes, latitudes: tuple[np.ndarray]
        Longitude and Latitude backplanes.
    """
    _, loc = cubedata_from_json_file(catalog.loc.json)
    loc.transpose_to("BIP")

    longitudes = np.array(loc.array.values[:, :, 0])
    longitudes = ((longitudes + 180) % 360) - 180
    latitudes = loc.array.values[:, :, 1]

    return longitudes, latitudes


def crop_data(
    state: PipelineState, bbox: BoundingBoxModel, catalog: M3DataPaths
) -> CropResult[np.ndarray]:
    """
    This method determines the cropping method based on the presence of
    GCPs. If GCPs are provided, it uses the GCP-based cropping method.
    Otherwise, it falls back to using the LOC backplane for cropping.
    """
    longitudes, latitudes = load_loc_backplane(catalog)
    return crop_data_from_loc_backplane(
        state.data, longitudes, latitudes, bbox
    )


def get_cropped_area(
    state: PipelineState, crs: str, bbox: BoundingBoxModel
) -> AreaDefinition:
    return AreaDefinition(
        area_id="cropping_area",
        description="Area initialized during crop step",
        proj_id="bbox_proj",
        projection=crs,
        width=state.data.shape[1],
        height=state.data.shape[0],
        area_extent=bbox.as_extent(),
    )


def modify_state(
    state: PipelineState, crop_data: np.ndarray, crop_area: AreaDefinition
) -> PipelineState:
    state.data = crop_data
    state.geom.area = crop_area
    state.flags |= CompletedFlag.CROPPED
    return state


def write_to_cache(
    cache: PipelineCache,
    timestamp: str,
    output: PipelineState,
    cropped_loc: np.ndarray,
    bbox: BoundingBoxModel,
) -> None:
    cube_attrs = get_standard_dset_attrs(timestamp, output)
    latlong_attrs = get_standard_dset_attrs(timestamp, output)

    latlong_attrs["measvals"] = [0, 1]
    latlong_attrs["bandlbls"] = ["Longitude", "Latitude"]

    cache.cropped.cube.write(np.array(output.data))
    cache.cropped.cube.set_attrs(cube_attrs)

    cache.cropped.latlong.write(cropped_loc)
    cache.cropped.latlong.set_attrs(latlong_attrs)

    cache.cropped.set_attrs(
        {
            "timestamp": timestamp,
            "flags": output.flags,
            "left_bound": bbox.left,
            "bottom_bound": bbox.bottom,
            "right_bound": bbox.right,
            "top_bound": bbox.top,
            "height": output.data.shape[0],
            "width": output.data.shape[1],
        }
    )
