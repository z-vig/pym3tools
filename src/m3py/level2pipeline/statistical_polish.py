# Standard Libraries
from datetime import datetime
from typing import Mapping, Tuple, Optional
import os
from pathlib import Path
import re

# Dependencies
import numpy as np
import h5py as h5  # type: ignore

# Relative Imports
from .step import Step, PipelineState, StepCompletionState

# Top-Level Imports
from m3py.PDSretrieval.file_manager import M3FileManager
from m3py.io.read_m3 import get_wavelengths

TimeRange = Tuple[datetime, datetime]
PathLike = str | os.PathLike | Path


class InstrumentTemperatureCheckError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class StatisticalPolishFileError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


time_ranges: Mapping[str, TimeRange] = {
    "cold1": (datetime(2009, 1, 19, 0, 0, 0), datetime(2009, 2, 15, 0, 0, 0)),
    "cold2": (datetime(2009, 4, 15, 0, 0, 0), datetime(2009, 4, 28, 0, 0, 0)),
    "cold3": (datetime(2009, 7, 12, 0, 0, 0), datetime(2009, 8, 17, 0, 0, 0)),
    "warm1": (datetime(2008, 11, 18, 0, 0, 0), datetime(2009, 1, 19, 0, 0, 0)),
    "warm2": (datetime(2009, 5, 13, 0, 0, 0), datetime(2009, 5, 17, 0, 0, 0)),
    "warm3": (datetime(2009, 5, 20, 0, 0, 0), datetime(2009, 7, 10, 0, 0, 0)),
}


def check_instrument_temp(
    acq_time: datetime, manager: M3FileManager
) -> PathLike:
    temperature_check: Optional[str] = None
    for k, v in time_ranges.items():
        if v[0] <= acq_time < v[1]:
            temperature_check = k

    if temperature_check is None:
        raise InstrumentTemperatureCheckError(
            f"{acq_time} is not wihtin a known time range of the M3 mission."
        )

    if temperature_check.find("cold") > -1:
        return manager.cal_dir.statistical_polish1
    elif temperature_check.find("warm") > -1:
        return manager.cal_dir.statistical_polish2
    else:
        raise ValueError(f"{temperature_check} is an invalid key.")


class StatisticalPolish(Step):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

    def run(self, state: PipelineState) -> PipelineState:

        acq_time_pattern = re.compile(
            r"START_TIME\s*=\s(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
        )
        m3_date = "%Y-%m-%dT%H:%M:%S"
        with open(self.manager.pds_dir.l1.lbl) as f:
            acq_time_str = re.findall(acq_time_pattern, f.read())[0]
            acq_time = datetime.strptime(acq_time_str, m3_date)

        stat_pol_path = Path(check_instrument_temp(acq_time, self.manager))

        acq_time_check_pattern = re.compile(
            r"CH1:STATISTICAL_POLISHER_FILE_NAME\s*=\s\"(M3(?:G|T)\d{8}_\w{3}"
            r"_\w{4}_\w{3}_\d.TAB)\""
        )
        with open(self.manager.pds_dir.l2.lbl) as f:
            stat_pol_file_check = re.findall(acq_time_check_pattern, f.read())[
                0
            ]

        if stat_pol_path.name != stat_pol_file_check:
            raise StatisticalPolishFileError(
                f"{stat_pol_path} does not match the file name specified in"
                f"the L2 label file: {stat_pol_file_check}."
            )

        stat_pol_tab_pattern = re.compile(
            r"\s*\d{1,3}\s*(\d{2,4}.\d{3})\s{2}(\d.\d{6})\s{3}\d.\d{6}\n"
        )

        _, bbl = get_wavelengths(self.manager)

        with open(stat_pol_path) as f:
            data_array = np.array(
                re.findall(stat_pol_tab_pattern, f.read()), dtype=np.float32
            )
            stat_pol_wvl = data_array[bbl, 0]
            self.stat_pol_coefs = data_array[bbl, 1]

        if not np.allclose(stat_pol_wvl, state.wvl):
            raise StatisticalPolishFileError(
                "The wavelengths that were read from the statistical polishing"
                "file do not match the data wavelengths."
            )

        statisical_polish_applied = (
            state.data * self.stat_pol_coefs[None, None, :]
        )

        new_flags = state.flags
        new_flags.statistical_polishing_applied = StepCompletionState.Complete

        new_state = PipelineState(
            data=statisical_polish_applied,
            wvl=state.wvl,
            obs=state.obs,
            georef=state.georef,
            flags=new_flags,
        )

        return new_state

    def save(self, output: PipelineState):
        super().save(output)
        with h5.File(self.manager.cache, "r+") as f:
            g = f[self.name]
            assert isinstance(g, h5.Group)
            g.attrs["statistical_polish_coefs"] = self.stat_pol_coefs
