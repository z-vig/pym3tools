from typing import Literal, Any
from collections.abc import Callable
from datetime import datetime

from pym3tools.types import is_valid_acquisition_mode, acq_mode_to_char

from .download_models import TabDownload, URLPath
from .data_directory import M3DataPaths


def determine_warm_cold(start_time: datetime) -> Literal["Warm", "Cold"]:
    if (
        (
            start_time >= datetime(2009, 1, 19, 0, 0, 0)
            and start_time < datetime(2009, 2, 15, 0, 0, 0)
        )
        or (
            start_time >= datetime(2009, 4, 15, 0, 0, 0)
            and start_time < datetime(2009, 4, 28, 0, 0, 0)
        )
        or (
            start_time >= datetime(2009, 7, 12, 0, 0, 0)
            and start_time < datetime(2009, 8, 17, 0, 0, 0)
        )
    ):
        return "Cold"
    elif (
        (
            start_time >= datetime(2008, 11, 18, 0, 0, 0)
            and start_time < datetime(2009, 1, 19, 0, 0, 0)
        )
        or (
            start_time >= datetime(2009, 5, 13, 0, 0, 0)
            and start_time < datetime(2009, 5, 17, 0, 0, 0)
        )
        or (
            start_time >= datetime(2009, 5, 20, 0, 0, 0)
            and start_time < datetime(2009, 7, 10, 0, 0, 0)
        )
    ):
        return "Warm"
    else:
        raise ValueError("Invalid Start Time")


def get_ground_truth_file(
    l0_search_function: Callable[[Any], str],
    l2_path: URLPath,
    save: M3DataPaths,
) -> TabDownload:
    start = datetime.strptime(
        l0_search_function("START_TIME"), "%Y-%m-%dT%H:%M:%S"
    )
    temp_status = determine_warm_cold(start)
    mode = l0_search_function("INSTRUMENT_MODE_ID").lower().capitalize()
    if not is_valid_acquisition_mode(mode):
        raise ValueError(f"Invalid Acquisition Mode: {mode}")
    acq = acq_mode_to_char[mode]

    if temp_status == "Cold":
        grndtru = TabDownload.from_base(
            l2_path / f"M3{acq}20111117_RFL_GRND_TRU_1",
            save.ground_truth.base,
        )
    elif temp_status == "Warm":
        grndtru = TabDownload.from_base(
            l2_path / f"M3{acq}20111117_RFL_GRND_TRU_2",
            save.ground_truth.base,
        )
    else:
        raise ValueError("Invalid Ground Truth")

    return grndtru
