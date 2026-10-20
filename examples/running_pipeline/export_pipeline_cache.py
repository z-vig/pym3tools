from pathlib import Path

import h5py
from cubio import CubeContext, CubeData, write_envi

from pym3tools.m3catalog import DataIDString, M3DataID
from pym3tools.save_models.export_cache_product import cache_to_cubio
from pym3tools.save_models.pipeline_cache_schema import PipelineCache

sample_ids: list[DataIDString] = [
    "M3T20090418T020644",
    "M3T20090418T020848",
    "M3G20090208T160125",
    "M3G20090208T175211",
    "M3G20090208T194335",
]


for id in sample_ids:
    if id.find("M3G") > -1:
        continue
    cache_fp = Path(
        f"D:/moon_data/m3/Gruithuisen_Region/{id}/pipeline_cache.hdf5"
    )
    products_directory = cache_fp.parent / "products"
    if not products_directory.exists():
        products_directory.mkdir()

    m3id = M3DataID.from_string(id)
    print(m3id.string, m3id.op)

    items_to_save: list[tuple[CubeContext, CubeData]] = []
    with h5py.File(str(cache_fp)) as f:
        cache = PipelineCache(f)

        items_to_save.append(
            cache_to_cubio(
                cache.photometric_corrected.photometry_backplane, f"{id}_photo"
            )
        )

        items_to_save.append(
            cache_to_cubio(cache.photometric_corrected.cube, f"{id}_rfl")
        )

        items_to_save.append(
            cache_to_cubio(cache.georeferenced.obs, f"{id}_obs")
        )

    for ctxt, cub in items_to_save:
        write_envi(
            ctxt,
            cub,
            "BIL",
            products_directory,
        )
