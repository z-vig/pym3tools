from dataclasses import dataclass
from typing import ClassVar, Self

from pym3tools.rdn2rfl.pipeline_state import (
    PipelineState,
    CompletedFlag,
    get_standard_dset_attrs,
)
from pym3tools.constants import MOON_RADIUS
from pym3tools.save_models.pipeline_cache_schema import PipelineCache
from pym3tools.data_retrieval.data_directory import M3DataPaths
from pym3tools.rdn2rfl.retrieve_terrain_data import (
    resample_m3geom,
    # crop_m3geom,
)

from cubio import cubedata_from_json_file
from cubio.geotools.models import (
    BoundingBoxModel,
    GCPGroup,
    GeotransformModel,
    PointModel,
)
from cubio.geotools.generate_geoloc_backplane import latlong_from_gcp_group
from pyresample.geometry import SwathDefinition, AreaDefinition
import numpy as np
import xarray as xr


@dataclass
class LatLongRetrieval:
    """Stores the results of lat/long retrieval"""

    latitudes: np.ndarray
    longitudes: np.ndarray
    row_slice: slice
    col_slice: slice


@dataclass
class PixelResolution:
    """Class for handling pixel resolution calculations and conversions."""

    pix_per_deg: float
    deg_per_pix: float
    m_per_pix: float  # Estimation
    radius_of_body: ClassVar[float] = MOON_RADIUS

    @classmethod
    def from_array(cls, max_lat: float, min_lat: float, height: int) -> Self:
        """Calculate pixel resolution from latitude bounds and array height."""
        lat_height = max_lat - min_lat
        dpp: float = lat_height / height  # Degrees per pixel
        return cls(
            pix_per_deg=1 / dpp,
            deg_per_pix=dpp,
            m_per_pix=np.pi * cls.radius_of_body * dpp / 180,
        )


def loc_from_gcp(
    gcps_grp: GCPGroup, uncropped_data: np.ndarray | xr.DataArray
) -> tuple[np.ndarray, np.ndarray, slice, slice]:
    if gcps_grp is None:
        raise ValueError("GCPS should not be None.")
    x, y = (
        gcps_grp.offset.col_slice,
        gcps_grp.offset.row_slice,
    )
    cropped_arr = np.array(uncropped_data[y, x])
    latlongarr = latlong_from_gcp_group(gcps_grp, cropped_arr)
    longitudes = latlongarr[:, :, 1]
    latitudes = latlongarr[:, :, 0]

    return longitudes, latitudes, x, y


def get_area_dimensions(
    latitudes: np.ndarray, arr_height: int, bbox: BoundingBoxModel
) -> tuple[int, int]:
    """
    Subtracts the max latitude from the minimum latitude and divides
    by the height of the entire image to get the # of pixels per degree.
    Returns the array size for the resulting area to match the pixel
    resolution.
    """
    max_lat = float(latitudes[0, :].max())
    min_lat = float(latitudes[-1, :].min())
    res = PixelResolution.from_array(
        max_lat=max_lat, min_lat=min_lat, height=arr_height
    )

    area_height = (bbox.top - bbox.bottom) * res.pix_per_deg
    area_width = (bbox.right - bbox.left) * res.pix_per_deg

    return int(area_height), int(area_width)


# ==== Retrieving latitude/longitude ====
def retrieve_latlong(
    state: PipelineState,
    cache: PipelineCache,
    catalog: M3DataPaths,
    gcps: GCPGroup | None,
) -> LatLongRetrieval:
    fromcache = CompletedFlag.CROPPED in state.flags
    if fromcache:
        print("From Cache...")
        loc = cache.cropped.latlong.read()
        longitudes = loc[:, :, 0]
        latitudes = loc[:, :, 1]
        xs = slice(0, loc.shape[1])
        ys = slice(0, loc.shape[0])
    elif not fromcache and (gcps is not None):
        print("From GCPs...")
        longitudes, latitudes, xs, ys = loc_from_gcp(gcps, state.data)
    else:
        print("From PDS...")
        _, loc_cube = cubedata_from_json_file(catalog.loc.json)
        loc_cube.transpose_to("BIP")
        longitudes = np.array(loc_cube.array.values[:, :, 0])
        longitudes = ((longitudes + 180) % 360) - 180
        latitudes = loc_cube.array.values[:, :, 1]
        xs = slice(0, loc_cube.shape.ncolumns)
        ys = slice(0, loc_cube.shape.nrows)

    return LatLongRetrieval(latitudes, longitudes, ys, xs)


# ==== Setting the bounding box ====
def set_bbox(
    bbox_handler: BoundingBoxModel | None,
    latlong: LatLongRetrieval,
    gcps: GCPGroup | None,
) -> tuple[BoundingBoxModel, int]:
    bbox: BoundingBoxModel
    bbox_set = isinstance(bbox_handler, BoundingBoxModel)
    if (gcps is None) and (not bbox_set):
        bbox = BoundingBoxModel(
            left=latlong.longitudes.min(),
            bottom=latlong.latitudes.min(),
            right=latlong.longitudes.max(),
            top=latlong.latitudes.max(),
            name="preset_bbox",
        )
        ngcps = 0
    elif (gcps is not None) and (not bbox_set):
        bbox = BoundingBoxModel(
            left=gcps.map_x.min(),
            bottom=gcps.map_y.min(),
            right=gcps.map_x.max(),
            top=gcps.map_y.max(),
            name="GCP_bbox",
        )
        ngcps = gcps.ngcp
    elif bbox_set:
        if gcps is None:
            ngcps = 0
        else:
            ngcps = gcps.ngcp
        if not isinstance(bbox_handler, BoundingBoxModel):
            raise RuntimeError("Invalid bbox handler.")
        bbox = bbox_handler
    else:
        raise RuntimeError("Conditional branch missed.")

    return bbox, ngcps


def make_resampling_geometries(
    latlong: LatLongRetrieval, prj: str, bbox: BoundingBoxModel
) -> tuple[SwathDefinition, AreaDefinition]:
    # ==== Defining the satellite swath ====
    swath = SwathDefinition(latlong.longitudes, latlong.latitudes)

    # ==== Calculating appropriate area dimensions ====
    area_height, area_width = get_area_dimensions(
        latlong.latitudes, latlong.latitudes.shape[0], bbox
    )

    # ==== Defining the georeferenced area ====
    area = AreaDefinition(
        "pipeline_area",
        "area defined for all current pipeline needs",
        "user-defined projection",
        prj,
        area_width,
        area_height,
        bbox.as_extent(),
    )

    return swath, area


# ==== Setting lat/long array ====
def get_new_loc(area: AreaDefinition) -> np.ndarray:
    return np.stack(area.get_proj_coords(), axis=-1)


def get_new_geotransform(area: AreaDefinition) -> GeotransformModel:
    return GeotransformModel(
        upperleft=PointModel(
            x=area.pixel_upper_left[0], y=area.pixel_upper_left[1]
        ),
        xres=area.resolution[0],
        row_rotation=0,
        yres=area.resolution[1],
        col_rotation=0,
    )


def modify_state(
    state: PipelineState,
    latlong: LatLongRetrieval,
    geotransform: GeotransformModel,
    swath: SwathDefinition,
    area: AreaDefinition,
    prj: str,
) -> PipelineState:
    # ==== Updating pipeline state ====
    state.crs = prj
    state.gtrans = geotransform
    state.geom.swath = swath
    state.geom.area = area
    state.geom.set_swath_window(latlong.row_slice, latlong.col_slice)
    state.data = state.geom.swath_to_gridded_data(np.array(state.data))
    # cropped_m3geom = crop_m3geom(
    #     state.obs, latlong.row_slice, latlong.col_slice
    # )
    state.obs = resample_m3geom(state.obs, state.geom)
    state.flags |= CompletedFlag.GEOREFERENCED

    return state


def write_to_cache(
    timestamp: str,
    output: PipelineState,
    loc: np.ndarray,
    cache: PipelineCache,
    prj: str,
    ngcps: int,
) -> None:
    # ==== Attribute Config ====
    cube_attrs = get_standard_dset_attrs(timestamp, output)
    latlong_attrs = get_standard_dset_attrs(timestamp, output)
    obs_attrs = get_standard_dset_attrs(timestamp, output)

    # ==== Setting pipeline cache attributes ====
    centroid = (
        float(np.mean(loc[:, :, 0])),
        float(np.mean(loc[:, :, 1])),
    )

    latlong_attrs["measvals"] = [0, 1]
    latlong_attrs["bandlbls"] = ["Longitude", "Latitude"]

    obs_measvals: list[float] = []
    obs_lbls: list[str] = []
    for n, i in enumerate(output.obs.geom_dict.keys()):
        obs_measvals.append(float(n))
        obs_lbls.append(i)
    obs_attrs["measvals"] = obs_measvals
    obs_attrs["bandlbls"] = obs_lbls

    # ==== Saving Cube Data ====
    cache.georeferenced.cube.write(np.array(output.data, dtype=np.float32))
    cache.georeferenced.cube.set_attrs(cube_attrs)

    # ==== Saving Lat/Long Data ====
    cache.georeferenced.latlong.write(loc)
    cache.georeferenced.latlong.set_attrs(latlong_attrs)

    # ==== Saving OBS Data ====
    cache.georeferenced.obs.write(output.obs.cube)
    cache.georeferenced.obs.set_attrs(obs_attrs)

    # ==== Saving Step Attributes ====
    cache.georeferenced.set_attrs(
        {
            "timestamp": timestamp,
            "flags": int(output.flags),
            "ngcps": ngcps,
            "center": centroid,
            "geotransform": output.gtrans.togdal(),
            "crs": prj,
        }
    )
