# Standard Libraries
import re
from typing import Tuple

# Dependencies
import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type: ignore

# Top-Level Imports
from m3py.io.read_m3 import get_wavelengths
from m3py.PDSretrieval.file_manager import M3FileManager


class SolarSpectrumReadError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def get_solar_correction_values(
    manager: M3FileManager,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns solar spectrum, solar wavelengths and solar distance.
    """
    solspec_parse = re.compile(r"\s*(\d{2,4}.\d{6})")
    wvl, bbl = get_wavelengths(manager)
    with open(manager.cal_dir.solar_spectrum) as f:
        data_array = np.array(
            [re.findall(solspec_parse, i) for i in f.readlines()],
            dtype=np.float32,
        )
        solar_wvl = data_array[bbl, 0]
        solar_spec = data_array[bbl, 1]

    solar_distance_pattern = re.compile(r"SOLAR_DISTANCE\s*=\s(\d.\d*)\s<AU>")
    with open(manager.pds_dir.l1.lbl) as f:
        solar_distance = float(re.findall(solar_distance_pattern, f.read())[0])

    if not np.allclose(solar_wvl, wvl[bbl]):
        raise SolarSpectrumReadError(
            "The solar spectrum wavelength values do not match the data"
            "wavelength values."
        )

    return solar_spec, solar_wvl, solar_distance


def get_phase_function_rgi(manager) -> RegularGridInterpolator:
    pattern = re.compile(r"\s\d.\d{9}")
    _, bbl = get_wavelengths(manager)
    with open(manager.cal_dir.phase_function) as f:
        phase_function_lookup = np.array(
            [re.findall(pattern, i) for i in f.readlines()[1:]],
            dtype=np.float32,
        )
        phase_function_lookup = phase_function_lookup[:100, bbl]

    x = np.arange(phase_function_lookup.shape[0])
    y = np.arange(phase_function_lookup.shape[1])
    return RegularGridInterpolator((x, y), phase_function_lookup)
