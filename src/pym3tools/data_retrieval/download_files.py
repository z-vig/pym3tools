from pathlib import Path

from pym3tools2.m3catalog import DataIDString
from .data_directory import M3DataPaths, init_data_dir
from .download_models import Downloadable
from .config_download_paths import (
    config_l0,
    config_l1,
    config_l2,
    config_calibration,
)
from .retrieve_urls import retrieve_urls
from .add_json_files import add_json_to_level


def gather_downloads(downloads: list[Downloadable]) -> dict[str, Path]:
    master_download_inventory: dict[str, Path] = {}
    for i in downloads:
        master_download_inventory = {
            **master_download_inventory,
            **i.to_save(),
        }
    return master_download_inventory


def gather_files(
    dataset_dir: Path, data_id: DataIDString
) -> tuple[M3DataPaths, dict[str, Path]]:
    m3data = M3DataPaths(dataset_dir)
    if m3data.check_status():
        print(f"All files for {data_id} are already downloaded.")
        return m3data, {}
    l0_files = config_l0(data_id, m3data)
    l1_files = config_l1(data_id, m3data)
    l2_files = config_l2(data_id, m3data)
    cal_files = config_calibration(data_id, m3data)

    download_dict = gather_downloads(
        [*l0_files, *l1_files, *l2_files, *cal_files]
    )

    return m3data, download_dict


def download_m3_dataset(
    base_dir: Path,
    data_id: DataIDString,
    overwrite_existing: bool = False,
    write_json: bool = True,
) -> dict[str, Path]:
    dataset_dir = init_data_dir(base_dir, data_id, overwrite_existing)
    m3data, download_dict = gather_files(dataset_dir, data_id)
    retrieve_urls(download_dict)
    if write_json:
        add_json_to_level(m3data, "L0")
        add_json_to_level(m3data, "L1B")
        add_json_to_level(m3data, "L2")

    return download_dict
