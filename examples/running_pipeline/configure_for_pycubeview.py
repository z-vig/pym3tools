# flake8: noqa

from pym3tools.data_retrieval.data_directory import M3DataPaths
import spectralio as sio
from cubio import cubedata_from_json_file, cubedata_from_geotiff, write_envi
from cubio.envi_hdr_tools import extract_hdr_bbl
import numpy as np
import xarray as xr


F32 = np.finfo(np.float32)


def tif_to_envi():
    cat = M3DataPaths("D:/moon_data/m3/Gruithuisen_Region/M3T20090418T020644/")
    mosaic_fp = "D:/moon_data/m3/Gruithuisen_Region/M3T_GRUIT_MOSAIC/M3T_GRUIT_PHOTOMETRY.tif"
    sample, _ = cubedata_from_json_file(
        "D:/moon_data/m3/Gruithuisen_Region/mosaic_components_photometry/M3T20090418T020644_PHOTOMETRY.json"
    )
    print(sample.band_names)
    bbl = extract_hdr_bbl(cat.rfl.hdr)
    if bbl == "No BBL Found":
        raise ValueError()

    ctxt, cub = cubedata_from_geotiff(
        mosaic_fp,
        "M3T_GRUIT_PHOTOMETRY",
        "Reflectance Mosaic for Targeted Mode",
        "photometry",
        "na",
        sample.band_names,
        sample.measurement_values,
    )

    new_data = cub.array.values
    print(new_data.max())
    print(new_data.size)
    print(np.count_nonzero(new_data == new_data.max()))
    new_data[new_data == new_data.max()] = -999
    cub.array = xr.DataArray(new_data)

    write_envi(ctxt, cub, "BIL", mosaic_fp)


def envi_to_geospcub():
    ctxt, cub = cubedata_from_json_file(
        "D:/moon_data/m3/Gruithuisen_Region/M3T_GRUIT_MOSAIC/M3T_GRUIT_PHOTOMETRY.json"
    )
    geodat = sio.BaseGeolocationModel(
        crs=ctxt.crs,
        geotransform=sio.geospatial_models.GeotransformModel(
            upperleft=sio.geospatial_models.PointModel(
                x=ctxt.geotransform.upperleft.x,
                y=ctxt.geotransform.upperleft.y,
            ),
            xres=ctxt.geotransform.xres,
            yres=ctxt.geotransform.yres,
            col_rotation=ctxt.geotransform.col_rotation,
            row_rotation=ctxt.geotransform.row_rotation,
        ),
    )
    geo_json = geodat.model_dump_json(indent=2)

    geospcub = sio.GeoSpectrum3D(
        name="m3t_photometry",
        wavelength=sio.WvlModel(
            values=ctxt.measurement_values,
            unit="nm",
            bbl=[bool(i) for i in ctxt.bad_bands],
        ),
        nrows=ctxt.shape.nrows,
        ncols=ctxt.shape.ncolumns,
        nbands=ctxt.shape.nbands,
        raster_fp="D:/moon_data/m3/Gruithuisen_Region/M3T_GRUIT_MOSAIC/M3T_GRUIT_PHOTOMETRY.bil",
        geodata=geodat,
    )
    dat_json = geospcub.model_dump_json(indent=2)

    with open(
        "D:/moon_data/m3/Gruithuisen_Region/M3T_GRUIT_MOSAIC/M3T_GRUIT_PHOTOMETRY.geodata",
        "w",
    ) as f:
        f.write(geo_json)

    with open(
        "D:/moon_data/m3/Gruithuisen_Region/M3T_GRUIT_MOSAIC/M3T_GRUIT_PHOTOMETRY.geospcub",
        "w",
    ) as f:
        f.write(dat_json)


tif_to_envi()
envi_to_geospcub()
