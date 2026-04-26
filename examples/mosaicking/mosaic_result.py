# flake8: noqa
from pathlib import Path
from pym3tools.geo_ops import mosaic_arrays
from pym3tools.geo_ops.data_transfer_classes import ResampledMosaic
from cubio import cubedata_from_json_file, cube_from_numpy, write_envi
from mosaic_config import MosaicConfig


def mosaic_from_config(config_fp: str | Path):
    config = MosaicConfig.from_json(config_fp)
    rfl_mosaic = mosaic_arrays(
        config.rfl_fp_list, "Mean", config.inc_fp_list, photometry_cube=True
    )
    inc_mosaic = mosaic_arrays(config.inc_fp_list, "Mean")

    mosaic_dict: dict[str, tuple[ResampledMosaic, str]] = {
        "M3G_GRUIT_RFL": (rfl_mosaic, config.rfl_fp_list[0]),
        "M3G_GRUIT_INC": (inc_mosaic, config.inc_fp_list[0]),
    }

    for name, (mosaic, sample) in mosaic_dict.items():
        sample_cc, _ = cubedata_from_json_file(sample)
        mosaic_cc, mosaic_cd = cube_from_numpy(
            mosaic.data,
            name,
            mosaic.crs,
            mosaic.gtrans,
            measvals=sample_cc.measurement_values,
            bandlbls=sample_cc.band_names,
        )

        write_envi(
            mosaic_cc,
            mosaic_cd,
            "BIL",
            "D:/moon_data/m3/Gruithuisen_Region/M3G_GDOMES_MOSAIC/_",
        )


if __name__ == "__main__":
    mosaic_from_config(Path(__file__).parent / "global_config.json")
