from pathlib import Path
from functools import partial

from pym3tools2.m3catalog import (
    DataIDString,
    get_l0_metadata,
    get_l1_metadata,
    get_l2_metadata,
)

from .download_models import (
    Downloadable,
    URLPath,
    TabDownload,
    ImageDownload,
    DownloadPath,
)
from .data_directory import M3DataPaths
from .get_ground_truth_file import get_ground_truth_file


JPL = URLPath("https://planetarydata.jpl.nasa.gov/img/data/m3/")


def config_l0(data_id: DataIDString, save: M3DataPaths):
    get_l0 = partial(get_l0_metadata, data_id)
    l0_path = (
        JPL
        / get_l0("VOLUME_ID")
        / Path(get_l0("FILE_SPECIFICATION_NAME")).parent
    )

    # ==== L0 Downloads ====
    l0_lbl = DownloadPath(
        str(l0_path / Path(get_l0("FILE_SPECIFICATION_NAME")).name),
        save.L0_lbl,
    )
    raw = ImageDownload.from_base(
        l0_path / get_l0("PRODUCT_ID"), save.raw.base
    )

    return [l0_lbl, raw]


def config_l1(data_id: DataIDString, save: M3DataPaths) -> list[Downloadable]:
    get_l1 = partial(get_l1_metadata, data_id)
    l1_path = JPL.joinpath(
        get_l1("VOLUME_ID"), Path(get_l1("FILE_SPECIFICATION_NAME")).parent
    )
    # ==== L1 Downloads ====
    l1_lbl = DownloadPath(
        str(l1_path / Path(get_l1("FILE_SPECIFICATION_NAME")).name),
        save.L1_lbl,
    )
    rdn = ImageDownload.from_base(
        l1_path / get_l1("PRODUCT_ID"), save.rdn.base
    )
    loc = ImageDownload.from_base(l1_path / get_l1("LOC_FILE"), save.loc.base)
    obs = ImageDownload.from_base(l1_path / get_l1("OBS_FILE"), save.obs.base)

    return [l1_lbl, rdn, loc, obs]


def config_l2(data_id: DataIDString, save: M3DataPaths) -> list[Downloadable]:
    get_l2 = partial(get_l2_metadata, data_id)
    l2_path = (
        JPL
        / get_l2("VOLUME_ID")
        / Path(get_l2("FILE_SPECIFICATION_NAME")).parent
    )
    # ==== L2 Downloads ====
    l2_lbl = DownloadPath(
        str(l2_path / Path(get_l2("FILE_SPECIFICATION_NAME")).name),
        save.L2_lbl,
    )
    rfl = ImageDownload.from_base(
        l2_path / get_l2("RFL_IMAGE_FILE_NAME"), save.rfl.base
    )
    sup = ImageDownload.from_base(
        l2_path / get_l2("SUP_IMAGE_FILE_NAME"), save.sup.base
    )

    return [l2_lbl, rfl, sup]


def config_calibration(
    data_id: DataIDString, save: M3DataPaths
) -> list[Downloadable]:
    get_l2 = partial(get_l2_metadata, data_id)
    cal_path = JPL / get_l2("VOLUME_ID") / "CALIB"
    # ==== Calibration Downloads ====
    solspec = TabDownload.from_base(
        cal_path / get_l2("CH1:SOLAR_SPECTRUM_FILE_NAME"),
        save.solar_spectrum.base,
    )
    statpol = TabDownload.from_base(
        cal_path / get_l2("CH1:STATISTICAL_POLISHER_FILE_NAME"),
        save.statistical_polish.base,
    )
    grndtru = get_ground_truth_file(get_l2, cal_path, save)

    falpha = TabDownload.from_base(
        cal_path / get_l2("CH1:PHOTOMETRY_CORR_FILE_NAME"),
        save.photometry_correction.base,
    )

    return [solspec, statpol, grndtru, falpha]
