from pym3tools.save_models.export_cache_product import cache_to_cubio
from pym3tools.save_models.pipeline_cache_schema import PipelineCache
from pym3tools.m3catalog import DataIDString
from pym3tools.m3catalog import M3DataID

# from cubio import write_envi  # , write_zarr

import h5py

sample_ids: list[DataIDString] = [
    "M3T20090418T020644",
    "M3T20090418T020848",
    "M3G20090208T160125",
    "M3G20090208T175211",
    "M3G20090208T194335",
]


for id in sample_ids:

    cache_fp = f"D:/moon_data/m3/Gruithuisen_Region/{id}/pipeline_cache.hdf5"
    savename = f"{id}_PHOTOMETRY"

    m3id = M3DataID.from_string(id)
    print(m3id.string, m3id.op)

    # with h5py.File(cache_fp) as f:
    #     cache = PipelineCache(f)
    #     ctxt, cub = cache_to_cubio(
    #         cache.photometric_corrected.photometry_backplane, savename
    #     )

    # write_zarr(ctxt, cub, f"D:/moon_data/m3/Gruithuisen_Region/{savename}")
    # write_envi(
    #     ctxt,
    #     cub,
    #     "BIL",
    #     "D:/moon_data/m3/Gruithuisen_Region/mosaic_components_photometry/_",
    # )
