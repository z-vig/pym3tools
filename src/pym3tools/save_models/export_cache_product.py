from pathlib import Path
from typing import Literal

import cubio as cb
import h5py
from cubio.geotools.models import GeotransformModel

from pym3tools.m3catalog import DataIDString
from pym3tools.save_models import PipelineCache
from pym3tools.save_models.attribute_models import StandardDatasetAttrs
from pym3tools.save_models.pipeline_cache_schema import Dataset


def cache_to_cubio(
    main_dset: Dataset[StandardDatasetAttrs],
    save_name: str,
    nodata_val: int = -999,
    measvals_overwrite: bool = False,
    bandlbls_overwrite: bool = False,
) -> tuple[cb.CubeContext, cb.CubeData]:
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

    cc, cd = cb.cube_from_numpy(
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


def save_suite(
    cache_fp: Path | str,
    save_dir: Path | str,
    m3id: DataIDString,
    format: Literal["BIP", "BSQ", "BIL"] = "BIL",
) -> None:
    """
    Saves suite of image data given a processed .hdf5 pipeline cache.
    """
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    with h5py.File(str(cache_fp)) as f:
        c = PipelineCache(f)
        save_dict: dict[str, Dataset[StandardDatasetAttrs]] = {
            f"{m3id}_georef": c.georeferenced.cube,
            f"{m3id}_obs": c.georeferenced.obs,
            f"{m3id}_photo": c.photometric_corrected.photometry_backplane,
            f"{m3id}_rfl": c.photometric_corrected.cube,
        }

        for k, v in save_dict.items():
            ctxt, cub = cache_to_cubio(v, k)
            cb.write_envi(ctxt, cub, format, Path(save_dir))
