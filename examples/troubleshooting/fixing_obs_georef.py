# flake8: noqa

from cubio import cubedata_from_json_file, cube_from_numpy, write_envi
from cubio.geotools.models import GCPGroup
import numpy as np
from pyresample.geometry import SwathDefinition, AreaDefinition

import matplotlib.pyplot as plt

from pym3tools.data_retrieval import M3DataPaths
from pym3tools.rdn2rfl.retrieve_terrain_data import (
    load_sphere_geometry_data,
)
from pym3tools.constants import MOON_GCS_PRJ
from pym3tools.rdn2rfl.data_transfer_classes import GeoreferencingGeometry
from pym3tools.rdn2rfl.pipeline_services.georeference import (
    get_new_geotransform,
    loc_from_gcp,
)

catalog = M3DataPaths("D:/moon_data/m3/Gruithuisen_Region/M3G20090208T160125/")
slope_fp = "D:/moon_data/derived_products/gruithuisen_region/Gruit_Slope.json"
aspect_fp = (
    "D:/moon_data/derived_products/gruithuisen_region/Gruit_Aspect.json"
)

slope_ctxt, slope_data = cubedata_from_json_file(slope_fp)
slope_data.transpose_to("BIP")
aspect_ctxt, aspect_data = cubedata_from_json_file(aspect_fp)
aspect_data.transpose_to("BIP")
rdn_ctxt, rdn_data = cubedata_from_json_file(catalog.rdn.json)
rdn_data.transpose_to("BIP")

xs = slice(0, 304)
ys = slice(5500, 6500)
_, loc_cube = cubedata_from_json_file(catalog.loc.json)
loc_cube.transpose_to("BIP")
longitudes = np.array(loc_cube.array.values[ys, xs, 0])
longitudes = ((longitudes + 180) % 360) - 180
latitudes = loc_cube.array.values[ys, xs, 1]

m3geom = load_sphere_geometry_data(catalog)
m3geom.convert_to_radians()

gcps = GCPGroup.from_gcps_file(
    "D:/moon_data/m3/Gruithuisen_Region/gcps_files/M3G20090208T160125.gcps"
)
gcp_lon, gcp_lat, gcp_xs, gcp_ys = loc_from_gcp(gcps, rdn_data.array)


dset_h, dset_w = latitudes.shape
area = AreaDefinition(
    area_id="NullArea",
    description=(
        "Area defined as the height and width of the M3 dataset and"
        " an extent starting at (0, 0) in the upper left"
    ),
    proj_id="default_moon",
    projection=MOON_GCS_PRJ,
    width=dset_w,
    height=dset_h,
    area_extent=(
        longitudes[-1, 0],
        latitudes[-1, 0],
        longitudes[0, -1],
        latitudes[0, -1],
    ),
)

swath = SwathDefinition(lons=longitudes, lats=latitudes)
# swath = SwathDefinition(lons=gcp_lon, lats=gcp_lat)

geom = GeoreferencingGeometry()
geom.swath = swath
geom.area = area
# geom.set_swath_window(ys, xs)
geom.set_swath_window(gcp_ys, gcp_xs)

new_gtrans = get_new_geotransform(area)
new_gtrans.force_northup()

# test_rdn = geom.swath_to_gridded_data(rdn_data.array.values)
test_rdn = rdn_data.array.values[ys, xs]
test_slope = geom.gridded_data_to_swath(
    slope_data.array.values, slope_data.bounds
)

arr_dict = {"rdn": test_rdn, "slope": test_slope}

for k, v in arr_dict.items():
    cc, cd = cube_from_numpy(v, k, MOON_GCS_PRJ, new_gtrans)
    write_envi(cc, cd, "BIL", "D:/moon_data/m3/Gruithuisen_Region/_")

# plt.figure()
# plt.imshow(test_rdn[:, :, 10], cmap="Grays_r")

# plt.figure()
# plt.imshow(test_slope, cmap="Grays_r")
# plt.show()
