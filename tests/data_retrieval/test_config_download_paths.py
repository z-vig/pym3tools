from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import TestCase
from unittest.mock import call, patch

from pym3tools.data_retrieval import config_download_paths
from pym3tools.data_retrieval.data_directory import M3DataPaths
from pym3tools.data_retrieval.download_models import TabDownload, URLPath
from pym3tools.m3catalog import DataIDString

DATA_ID = cast(DataIDString, "M3G20090101T000000")


def make_save_paths() -> M3DataPaths:
    return cast(
        M3DataPaths,
        SimpleNamespace(
            L0_lbl=Path("dataset/L0/product.lbl"),
            raw=SimpleNamespace(base=Path("dataset/L0/raw")),
            L1_lbl=Path("dataset/L1/product.lbl"),
            rdn=SimpleNamespace(base=Path("dataset/L1/radiance")),
            loc=SimpleNamespace(base=Path("dataset/L1/location")),
            obs=SimpleNamespace(base=Path("dataset/L1/geometry")),
            L2_lbl=Path("dataset/L2/product.lbl"),
            rfl=SimpleNamespace(base=Path("dataset/L2/reflectance")),
            sup=SimpleNamespace(base=Path("dataset/L2/supplemental")),
            solar_spectrum=SimpleNamespace(base=Path("dataset/calibration/solar")),
            statistical_polish=SimpleNamespace(
                base=Path("dataset/calibration/statistical_polish")
            ),
            ground_truth=SimpleNamespace(base=Path("dataset/calibration/ground_truth")),
            photometry_correction=SimpleNamespace(
                base=Path("dataset/calibration/photometry")
            ),
            global_bandpass=SimpleNamespace(
                base=Path("dataset/calibration/global_bandpass")
            ),
            ssc_adj=SimpleNamespace(base=Path("dataset/calibration/ssc_adj")),
        ),
    )


class TestConfigDownloadPaths(TestCase):
    def test_config_l0_builds_label_and_image_downloads(self):
        save = make_save_paths()
        metadata = {
            "VOLUME_ID": "M3_0001",
            "FILE_SPECIFICATION_NAME": "DATA/RAW/L0_PRODUCT.LBL",
            "PRODUCT_ID": "L0_PRODUCT",
        }

        with patch.object(
            config_download_paths,
            "get_l0_metadata",
            side_effect=lambda _data_id, column: metadata[column],
        ) as get_metadata:
            downloads = config_download_paths.config_l0(DATA_ID, save)

        self.assertEqual(
            [download.to_save() for download in downloads],
            [
                {
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_0001/DATA/RAW/L0_PRODUCT.LBL": Path("dataset/L0/product.lbl")
                },
                {
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_0001/DATA/RAW/L0_PRODUCT.HDR": Path("dataset/L0/raw.hdr"),
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_0001/DATA/RAW/L0_PRODUCT.IMG": Path("dataset/L0/raw.img"),
                },
            ],
        )
        self.assertEqual(
            get_metadata.call_args_list,
            [
                call(DATA_ID, "VOLUME_ID"),
                call(DATA_ID, "FILE_SPECIFICATION_NAME"),
                call(DATA_ID, "FILE_SPECIFICATION_NAME"),
                call(DATA_ID, "PRODUCT_ID"),
            ],
        )

    def test_config_l1_builds_label_and_three_image_downloads(self):
        save = make_save_paths()
        metadata = {
            "VOLUME_ID": "M3_0002",
            "FILE_SPECIFICATION_NAME": "DATA/RDN/L1_PRODUCT.LBL",
            "PRODUCT_ID": "RADIANCE",
            "LOC_FILE": "LOCATION",
            "OBS_FILE": "GEOMETRY",
        }

        with patch.object(
            config_download_paths,
            "get_l1_metadata",
            side_effect=lambda _data_id, column: metadata[column],
        ):
            downloads = config_download_paths.config_l1(DATA_ID, save)

        source_base = "https://planetarydata.jpl.nasa.gov/img/data/m3/M3_0002/DATA/RDN/"
        self.assertEqual(
            [download.to_save() for download in downloads],
            [
                {source_base + "L1_PRODUCT.LBL": Path("dataset/L1/product.lbl")},
                {
                    source_base + "RADIANCE.HDR": Path("dataset/L1/radiance.hdr"),
                    source_base + "RADIANCE.IMG": Path("dataset/L1/radiance.img"),
                },
                {
                    source_base + "LOCATION.HDR": Path("dataset/L1/location.hdr"),
                    source_base + "LOCATION.IMG": Path("dataset/L1/location.img"),
                },
                {
                    source_base + "GEOMETRY.HDR": Path("dataset/L1/geometry.hdr"),
                    source_base + "GEOMETRY.IMG": Path("dataset/L1/geometry.img"),
                },
            ],
        )

    def test_config_l2_builds_label_and_two_image_downloads(self):
        save = make_save_paths()
        metadata = {
            "VOLUME_ID": "M3_0003",
            "FILE_SPECIFICATION_NAME": "DATA/RFL/L2_PRODUCT.LBL",
            "RFL_IMAGE_FILE_NAME": "REFLECTANCE",
            "SUP_IMAGE_FILE_NAME": "SUPPLEMENTAL",
        }

        with patch.object(
            config_download_paths,
            "get_l2_metadata",
            side_effect=lambda _data_id, column: metadata[column],
        ):
            downloads = config_download_paths.config_l2(DATA_ID, save)

        source_base = "https://planetarydata.jpl.nasa.gov/img/data/m3/M3_0003/DATA/RFL/"
        self.assertEqual(
            [download.to_save() for download in downloads],
            [
                {source_base + "L2_PRODUCT.LBL": Path("dataset/L2/product.lbl")},
                {
                    source_base + "REFLECTANCE.HDR": Path("dataset/L2/reflectance.hdr"),
                    source_base + "REFLECTANCE.IMG": Path("dataset/L2/reflectance.img"),
                },
                {
                    source_base + "SUPPLEMENTAL.HDR": Path(
                        "dataset/L2/supplemental.hdr"
                    ),
                    source_base + "SUPPLEMENTAL.IMG": Path(
                        "dataset/L2/supplemental.img"
                    ),
                },
            ],
        )

    def test_config_calibration_builds_calibration_downloads(self):
        save = make_save_paths()
        l2_metadata = {
            "VOLUME_ID": "M3_L2",
            "CH1:SOLAR_SPECTRUM_FILE_NAME": "SOLAR_SPECTRUM",
            "CH1:STATISTICAL_POLISHER_FILE_NAME": "STATISTICAL_POLISH",
            "CH1:PHOTOMETRY_CORR_FILE_NAME": "PHOTOMETRY_CORRECTION",
            "CH1:GLOBAL_BANDPASS_FILE_NAME": "GLOBAL_BANDPASS",
        }
        l1_metadata = {"VOLUME_ID": "M3_L1"}
        ground_truth = TabDownload.from_base(
            URLPath("https://example.test/ground_truth"),
            save.ground_truth.base,
        )

        with (
            patch.object(
                config_download_paths,
                "get_l2_metadata",
                side_effect=lambda _data_id, column: l2_metadata[column],
            ) as get_l2_metadata,
            patch.object(
                config_download_paths,
                "get_l1_metadata",
                side_effect=lambda _data_id, column: l1_metadata[column],
            ) as get_l1_metadata,
            patch.object(
                config_download_paths,
                "get_ground_truth_file",
                return_value=ground_truth,
            ) as get_ground_truth_file,
        ):
            downloads = config_download_paths.config_calibration(DATA_ID, save)

        self.assertEqual(
            [download.to_save() for download in downloads],
            [
                {
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_L2/CALIB/SOLAR_SPECTRUM.TAB": Path(
                        "dataset/calibration/solar.tab"
                    ),
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_L2/CALIB/SOLAR_SPECTRUM.LBL": Path(
                        "dataset/calibration/solar.lbl"
                    ),
                },
                {
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_L2/CALIB/STATISTICAL_POLISH.TAB": Path(
                        "dataset/calibration/statistical_polish.tab"
                    ),
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_L2/CALIB/STATISTICAL_POLISH.LBL": Path(
                        "dataset/calibration/statistical_polish.lbl"
                    ),
                },
                ground_truth.to_save(),
                {
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_L2/CALIB/PHOTOMETRY_CORRECTION.TAB": Path(
                        "dataset/calibration/photometry.tab"
                    ),
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_L2/CALIB/PHOTOMETRY_CORRECTION.LBL": Path(
                        "dataset/calibration/photometry.lbl"
                    ),
                },
                {
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_L1/CALIB/GLOBAL_BANDPASS.HDR": Path(
                        "dataset/calibration/global_bandpass.hdr"
                    ),
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_L1/CALIB/GLOBAL_BANDPASS.IMG": Path(
                        "dataset/calibration/global_bandpass.img"
                    ),
                },
            ],
        )
        self.assertEqual(
            get_l2_metadata.call_args_list,
            [
                call(DATA_ID, "VOLUME_ID"),
                call(DATA_ID, "CH1:GLOBAL_BANDPASS_FILE_NAME"),
                call(DATA_ID, "CH1:SOLAR_SPECTRUM_FILE_NAME"),
                call(DATA_ID, "CH1:STATISTICAL_POLISHER_FILE_NAME"),
                call(DATA_ID, "CH1:PHOTOMETRY_CORR_FILE_NAME"),
            ],
        )
        self.assertEqual(get_l1_metadata.call_args_list, [call(DATA_ID, "VOLUME_ID")])
        ground_truth_args = get_ground_truth_file.call_args.args
        self.assertEqual(
            str(ground_truth_args[1]),
            "https://planetarydata.jpl.nasa.gov/img/data/m3/M3_L2/CALIB",
        )
        self.assertIs(ground_truth_args[2], save)

    def test_config_ssc_adj_downloads_tab_without_label_source(self):
        save = make_save_paths()
        with patch.object(
            config_download_paths,
            "get_l1_metadata",
            return_value="M3_L1",
        ) as get_l1_metadata:
            downloads = config_download_paths.config_ssc_adj(DATA_ID, save)

        self.assertEqual(
            [download.to_save() for download in downloads],
            [
                {
                    "https://planetarydata.jpl.nasa.gov/img/data/m3/"
                    "M3_L1/EXTRAS/SSCADJ/"
                    f"{DATA_ID}_SSCADJ.TAB": Path("dataset/calibration/ssc_adj.tab"),
                    "": Path("dataset/calibration/ssc_adj.lbl"),
                }
            ],
        )
        get_l1_metadata.assert_called_once_with(DATA_ID, "VOLUME_ID")
