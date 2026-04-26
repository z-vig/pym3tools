from pathlib import Path
from typing import TypeGuard

import numpy as np
from cubio import cubedata_from_json_file, CubeData
from cubio.geotools.models import BoundingBoxModel, GeotransformModel
from pyresample.geometry import AreaDefinition
from pyresample.kd_tree import resample_nearest

from pym3tools.constants import MOON_RADIUS
from pym3tools.types import CubioTuple, is_cubio_tuple

from .data_transfer_classes import ResampledList


def read_json_list(fp_list) -> list[CubioTuple]:
    cubio_list: list[CubioTuple] = []
    for i in fp_list:
        cc, cd = cubedata_from_json_file(i)
        cd.transpose_to("BIP")
        cd.set_masking(True)
        cd.add_nodata_mask()
        cubio_list.append((cc, cd))
    return cubio_list


def get_radius_of_influence(
    area: AreaDefinition, radius_of_body: float = MOON_RADIUS
) -> float:
    m_per_degree = np.pi * radius_of_body / 180
    return 5 * m_per_degree * (area.pixel_size_x + area.pixel_size_y) / 2


def get_super_bounds(data_list: list[CubioTuple]) -> BoundingBoxModel:
    bounds = np.empty((len(data_list), 4))
    for n, (cc, cd) in enumerate(data_list):
        lats = cd.array.coords["Latitude"]
        lons = cd.array.coords["Longitude"]
        latmin, latmax = (float(lats.min()), float(lats.max()))
        lonmin, lonmax = (float(lons.min()), float(lons.max()))
        bounds[n] = (latmin, latmax, lonmin, lonmax)

    return BoundingBoxModel(
        left=bounds[:, 2].min(),
        bottom=bounds[:, 0].min(),
        right=bounds[:, 3].max(),
        top=bounds[:, 1].max(),
        name="super_bounds",
    )


def get_bulk_geotransform(
    data_list: list[CubioTuple], super_bbox: BoundingBoxModel
) -> GeotransformModel:
    res = np.empty((len(data_list), 2))
    for n, (cc, cd) in enumerate(data_list):
        res[n] = (cc.geotransform.xres, cc.geotransform.yres)

    mean_xres, mean_yres = (np.mean(res[:, 0]), np.mean(res[:, 1]))

    return GeotransformModel(
        upperleft=super_bbox.top_left,
        xres=float(mean_xres),
        yres=float(mean_yres),
        row_rotation=0,
        col_rotation=0,
    )


def get_crs(data_list: list[CubioTuple]) -> str:
    crs_list: list[str] = [i.crs for i, _ in data_list]
    test = np.array([i == crs_list[0] for i in crs_list])
    if not np.all(test):
        raise ValueError("Not all images have the same CRS.")
    return crs_list[0]


def get_image_size(
    super_bounds: BoundingBoxModel, bulk_gtrans: GeotransformModel
) -> tuple[int, int]:
    height = (super_bounds.top - super_bounds.bottom) / bulk_gtrans.yres
    width = (super_bounds.right - super_bounds.left) / bulk_gtrans.xres
    return int(abs(height)), int(abs(width))


def get_area_list(
    crs: str, data_list: list[CubioTuple]
) -> list[tuple[CubeData, AreaDefinition]]:
    area_list = []
    for n, (cc, cd) in enumerate(data_list):
        area = AreaDefinition(
            f"array{n}",
            "component",
            "moon",
            crs,
            width=cd.shape.ncolumns,
            height=cd.shape.nrows,
            area_extent=cd.bounds.as_extent(),
        )
        area_list.append((cd, area))
    return area_list


def get_total_area(
    crs: str, height: int, width: int, super_bounds: BoundingBoxModel
) -> AreaDefinition:
    total_area = AreaDefinition(
        "total",
        "total_area",
        "moon",
        crs,
        height=height,
        width=width,
        area_extent=super_bounds.as_extent(),
    )
    return total_area


def is_cubio_list(
    x: list[Path] | list[str] | list[CubioTuple],
) -> TypeGuard[list[CubioTuple]]:
    return isinstance(x, list) and all(is_cubio_tuple(i) for i in x)


def is_path_str_list(
    x: list[Path] | list[str] | list[CubioTuple],
) -> TypeGuard[list[Path] | list[str]]:
    return all(isinstance(i, (Path, str)) for i in x)


def resample_to_super_grid(
    input_list: list[Path] | list[str] | list[CubioTuple],
) -> ResampledList:
    if is_cubio_list(input_list):
        data_list = input_list
    elif is_path_str_list(input_list):
        data_list = read_json_list(input_list)
    else:
        raise ValueError("Invalid input type.")

    super_bounds = get_super_bounds(data_list)
    bulk_gtrans = get_bulk_geotransform(data_list, super_bounds)
    crs = get_crs(data_list)
    area_list = get_area_list(crs, data_list)

    height, width = get_image_size(super_bounds, bulk_gtrans)
    total_area = get_total_area(crs, height, width, super_bounds)

    resampled_list = []
    for ar in area_list:
        roi = get_radius_of_influence(total_area)
        dat = ar[0].array.values
        area_def = ar[1]
        resamp = resample_nearest(
            area_def,
            dat,
            total_area,
            roi,
            epsilon=0.5,  # type: ignore
            fill_value=np.nan,  # type: ignore
        )
        resampled_list.append(resamp)

    return ResampledList(resampled_list, bulk_gtrans, super_bounds, crs)
