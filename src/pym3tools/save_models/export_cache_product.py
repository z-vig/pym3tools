from typing import Literal

from cubio import CubeContext, CubeData, cube_from_numpy
from cubio.geotools.models import GeotransformModel

from pym3tools.save_models.attribute_models import StandardDatasetAttrs
from pym3tools.save_models.pipeline_cache_schema import Dataset


def cache_to_cubio(
    main_dset: Dataset[StandardDatasetAttrs],
    save_name: str,
    nodata_val: int = -999,
    measvals_overwrite: bool = False,
    bandlbls_overwrite: bool = False,
) -> tuple[CubeContext, CubeData]:
    data = main_dset.read()
    measvals: list[float] | Literal["default"]
    bandlbls: list[str] | Literal["default"]

    measvals = list(main_dset.attrs["measvals"])
    bandlbls = list(main_dset.attrs["bandlbls"])

    if measvals_overwrite:
        measvals = "default"

    if bandlbls_overwrite:
        bandlbls = "default"

    gtrans = GeotransformModel.fromgdal(main_dset.attrs["geotransform"])
    gtrans.force_northup()
    crs = main_dset.attrs["crs"]

    cc, cd = cube_from_numpy(
        data,
        "BIP",
        name=save_name,
        crs=crs,
        gtrans=gtrans,
        nodata=nodata_val,
        measvals=measvals,
        bandlbls=bandlbls,
    )

    return cc, cd
