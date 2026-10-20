import cubio as cb
import matplotlib.pyplot as plt
import numpy as np
from cubio.geotools.models import (
    BoundingBoxModel,
    GeotransformModel,
    PointModel,
)
from pyresample import kd_tree
from pyresample.geometry import AreaDefinition
from tqdm import tqdm

MOON_RADIUS = 1_737_400  # m


def get_bbox(cc: cb.CubeContext) -> BoundingBoxModel:
    return cc.geotransform.get_bbox(height=cc.nrows, width=cc.ncols)


def get_area(cc: cb.CubeContext) -> AreaDefinition:
    bbox = get_bbox(cc)

    return AreaDefinition(
        area_id="area",
        description="area",
        proj_id="prj",
        projection=cc.crs,
        width=cc.ncols,
        height=cc.nrows,
        area_extent=bbox.as_extent(),
    )


def get_new_geotransform(area: AreaDefinition) -> GeotransformModel:
    gtrans = GeotransformModel(
        upperleft=PointModel(
            x=area.pixel_upper_left[0], y=area.pixel_upper_left[1]
        ),
        xres=area.resolution[0],
        row_rotation=0,
        yres=area.resolution[1],
        col_rotation=0,
    )
    return gtrans


def get_radius_of_influence(
    area: AreaDefinition, radius_of_body: float = MOON_RADIUS
) -> float:
    m_per_degree = np.pi * radius_of_body / 180
    return 5 * m_per_degree * (area.pixel_size_x + area.pixel_size_y) / 2


def resample_cube(
    src_area: AreaDefinition, data: np.ndarray, trg_area: AreaDefinition
) -> np.ndarray:
    if data.ndim == 2:
        data = data[:, :, None]

    resamp_data = np.empty([trg_area.height, trg_area.width, data.shape[2]])
    for n in tqdm(range(data.shape[2])):
        resamp_data[:, :, n] = kd_tree.resample_nearest(
            source_geo_def=src_area,
            data=data[:, :, n],
            target_geo_def=trg_area,
            radius_of_influence=get_radius_of_influence(src_area),
            epsilon=0.5,  # type: ignore
            fill_value=np.nan,  # type: ignore
        )

    return resamp_data


# === Doing stuff ===
trgcc, trgcd = cb.cube_from_json(
    "D:/moon_data/m3/Gruithuisen_Region/M3T_GDOMES_MOSAIC/M3T_GRUIT_INC.json"
)
trgcd.transpose_to("BIP")

glbcc, glbcd = cb.cube_from_json(
    "D:/moon_data/m3/Gruithuisen_Region/M3G_GDOMES_MOSAIC/M3G_GRUIT_RFL.json"
)
glbcd.transpose_to("BIP")

trg_area = get_area(trgcc)
glb_area = get_area(glbcc)

trg_downsampled = resample_cube(trg_area, trgcd.array.values, glb_area)

plt.figure()
plt.imshow(trgcd.array.values[:, :, 0], cmap="Greys_r")
plt.title("Original")

plt.figure()
plt.imshow(glbcd.array.values[:, :, 0], cmap="Greys_r")
plt.title("Global")

plt.figure()
plt.imshow(trg_downsampled[:, :, 0], cmap="Greys_r")
plt.title("Resampled")

cc, cd = cb.cube_from_numpy(
    trg_downsampled,
    "BIP",
    glbcc,
    name="M3T_GRUIT_INC_GLOBALRESPIXELS",
    measvals=trgcc.measurement_values,
    bandlbls="default",
    bbl=trgcc.bad_bands,
)

cb.write_envi(
    cc, cd, "BIL", "D:/moon_data/m3/Gruithuisen_Region/M3T_GDOMES_MOSAIC/"
)

plt.show()
