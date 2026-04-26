from pathlib import Path

from cubio import CubeContext, CubeData
from cubio.geotools.models import GeotransformModel
from cubio.types import NumpyDType
import numpy as np
import xarray as xr

from pym3tools.save_models.pipeline_cache_schema import Dataset
from pym3tools.save_models.attribute_models import StandardDatasetAttrs


def cache_to_cubio(
    main_dset: Dataset[StandardDatasetAttrs],
    save_name: str,
    nodata_val: int = -999,
) -> tuple[CubeContext, CubeData]:
    data = main_dset.read()
    measvals = main_dset.attrs["measvals"]
    bandlbls = main_dset.attrs["bandlbls"]
    gtrans = GeotransformModel.fromgdal(main_dset.attrs["geotransform"])
    gtrans.force_northup()
    crs = main_dset.attrs["crs"]

    data[np.isnan(data)] = -999

    cc = CubeContext.from_builder(
        {
            "data_filename": Path(save_name),
            "name": "georef_nogcps",
            "description": "Georeferenced Image with  no GCPs",
            "ncols": data.shape[1],
            "nrows": data.shape[0],
            "nbands": data.shape[2],
            "data_type": NumpyDType(str(data.dtype)),
            "crs": crs,
            "geotransform": gtrans,
            "nodata": nodata_val,
            "measurement_values": measvals,
            "band_names": bandlbls,
            "measurement_units": "nm",
        }
    )
    cd = CubeData(cc.name, "BIP")

    lons, lats = gtrans.generate_coords(
        height=data.shape[0], width=data.shape[1]
    )

    cd.array = xr.DataArray(
        data,
        coords={
            "latitude": lats,
            "longitude": lons,
            "wavelength": measvals,
        },
        dims=["latitude", "longitude", "wavelength"],
    )

    return cc, cd
