# Built-Ins
from pathlib import Path

# Dependencies
import rasterio as rio  # type: ignore
import numpy as np


def read_m3_rio(data_fp: str | Path) -> tuple[np.ndarray, dict]:
    """
    Reads any rasterio-compatible M3 dataset.

    Parameters
    ----------
    data_fp: str | Path
        Filepath to M3 data.

    Returns
    -------
    arr: np.ndarray
        Array of data in the file.
    prf: dict
        Profile of metadata returned from rasterio.
    """
    with rio.open(data_fp) as f:
        arr: np.ndarray = f.read()  # type: ignore
        prf: dict = f.profile

    return arr, prf
