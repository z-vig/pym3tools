# Standard Libraries
from typing import TypedDict
from functools import partial
from pathlib import Path

# Relative Imports
# from .read_m3_binary import read_m3

# Dependencies
import numpy as np
from rasterio.coords import BoundingBox  # type: ignore

# Top-Level Imports
import pym3tools.formats.m3_data_format as fmt
from pym3tools.PDSretrieval.file_manager import M3FileManager
from pym3tools.types import PathLike

# from pym3tools.selenography.crop import regional_crop

from cubio.geotools.georeference_from_gcps import georeference_image
from cubio.geotools.georeference_satellite_swath import ProjectionDefinition
from cubio.data.crs_wkt_strings import GeographicCRS
from cubio.geotools.models import BoundingBoxModel


class M3DatasetNameError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def _read_m3_georef(
    img_path: PathLike,
    loc_path: PathLike,
    img_data_format: fmt.M3DataFormat,
    loc_data_format: fmt.M3DataFormat,
    acq_type: str,
    bbox: BoundingBox,
    gcp_path: PathLike,
) -> np.ndarray:
    # img = read_m3(img_path, img_data_format, acq_type)
    # loc = read_m3(loc_path, loc_data_format, acq_type)

    # img_cropped, row_off, col_off, _, _ = regional_crop(img, loc, bbox)

    prj4 = ProjectionDefinition(
        "GruithuisenRegion",
        "LunarGeographic",
        "GCS coordinates for the Moon",
        proj4_str="+proj=longlat +R=1737400 +no_defs +type=crs",
        crs_wkt_str=GeographicCRS.GCS_MOON_2000,
    )
    bbox_model = BoundingBoxModel(
        left=bbox.left,
        bottom=bbox.bottom,
        right=bbox.right,
        top=bbox.top,
        name="pleasework",
    )
    img_georef, gtrans = georeference_image(
        Path(img_path).with_suffix(".json"),
        Path(gcp_path),
        prj4,
        georef_extent=bbox_model,
        apply_cropping=True,
    )

    return img_georef


def read_m3_georef(
    manager: M3FileManager, bbox: BoundingBox, dataset_name: str
) -> np.ndarray:
    """
    Reads and georeferences an M3 dataset for a specified region and dataset
    type.

    This function selects the appropriate image and data format based on the
    dataset name, crops the image to the provided bounding box, applies ground
    control points (GCPs), and returns the georeferenced image as a NumPy
    array.

    Parameters
    ----------
    manager: M3FileManager
        File manager containing paths to M3 data products.
    bbox: BoundingBox
        Bounding box specifying the region of interest.
    dataset_name: str
        Name of the dataset to read (e.g., 'RDN', 'RFL', 'LOC', 'OBS', 'SUP').

    Returns:
        np.ndarray: Georeferenced image array with shape (height, width, bands)
    """
    partial_read = partial(
        _read_m3_georef,
        loc_path=manager.pds_dir.l1.loc_img,
        loc_data_format=fmt.LOC,
        acq_type=manager.acq_type,
        gcp_path=manager.georef_dir.gcps,
        bbox=bbox,
    )

    class PartialArgs(TypedDict):
        img_path: PathLike
        img_data_format: fmt.M3DataFormat

    dispatcher: dict[str, PartialArgs] = {
        "RDN": {
            "img_path": manager.pds_dir.l1.rdn_img,
            "img_data_format": fmt.L1,
        },
        "RFL": {
            "img_path": manager.pds_dir.l2.rfl_img,
            "img_data_format": fmt.L2,
        },
        "LOC": {
            "img_path": manager.pds_dir.l1.loc_img,
            "img_data_format": fmt.LOC,
        },
        "OBS": {
            "img_path": manager.pds_dir.l1.obs_img,
            "img_data_format": fmt.OBS,
        },
        "SUP": {
            "img_path": manager.pds_dir.l2.sup_img,
            "img_data_format": fmt.SUP,
        },
    }

    if dataset_name not in dispatcher.keys():
        raise M3DatasetNameError(
            f"{dataset_name} is not a valid M3 Dataset. Choose one of"
            f"{dispatcher.keys()}"
        )

    img = partial_read(**dispatcher[dataset_name])

    return img
