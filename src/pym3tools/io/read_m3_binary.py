# Standard Libraries
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional, Tuple, Mapping
import re

# Top-Level Imports
from pym3tools.PDSretrieval import M3FileManager
from pym3tools.formats.m3_data_format import M3DataFormat

# Dependencies
import numpy as np

type pathlike = str | os.PathLike


@dataclass
class Window:
    """
    Class for keeping track of window information for image viewing.
    The user will specify the bottom left row (X) and column (Y) of the window
    as well as the width and height as shown below:
    """

    X: int
    Y: int
    W: int
    H: int


def read_m3(
    img_path: str | os.PathLike,
    data_format: M3DataFormat,
    acq_type: str,
    window: Optional[Window] = None,
):
    """
    Reads binary M3 data from the PDS.

    Parameters
    ----------
    img_path: str | os.PathLike
        Filepath to M3 binary data file.
    data_format: M3DataFormat
        Format of the M3 data.
    acq_type:
        Acquisition type, global or targeted.
    window: rasterio.Window, optional
        Window to read the data from.
    """
    img_path = Path(img_path)

    nbands = getattr(data_format, acq_type).nbands
    ncols = getattr(data_format, acq_type).ncols
    dtype = getattr(data_format, acq_type).dtype
    hdrlen = getattr(data_format, acq_type).header_length

    dtype_dict: Mapping[str, Tuple[type, int]] = {
        "<d": (np.float64, 64 // 8),
        "<f": (np.float32, 32 // 8),
        "<h": (np.int16, 16 // 8),
    }

    numpy_dtype, nbytes = dtype_dict.get(dtype, (None, None))
    if (numpy_dtype is None) or (nbytes is None):
        raise ValueError(f"{dtype} is an invalid data type.")

    full_col_bytes = hdrlen + (ncols * nbands * nbytes)

    total_rows = os.path.getsize(img_path) // full_col_bytes

    if window is None:
        window = Window(0, 0, ncols, total_rows)

    start_row = window.Y
    col_offset = hdrlen + (window.X * nbands * nbytes)
    start_byte = start_row * full_col_bytes
    col_end_buffer = (ncols - (window.X + window.W)) * nbytes

    # Validating Window
    xbounds_chk = (window.Y + window.H) > total_rows
    ybounds_chk = (window.X + window.W) > ncols
    if xbounds_chk and not ybounds_chk:
        raise ValueError("Window does not fit within X bounds.")
    elif ybounds_chk and not xbounds_chk:
        raise ValueError("Window does not fit within Y bounds.")
    elif xbounds_chk and ybounds_chk:
        raise ValueError("Window does not fit within either X or Y bounds.")

    window_data: np.ndarray = np.empty(
        [window.H, window.W, nbands], dtype=numpy_dtype
    )

    with open(img_path, "rb") as f:
        byte_index = 0
        f.seek(start_byte)
        byte_index = f.tell()
        for i in range(0, window.H):
            f.seek(col_offset + byte_index)
            for j in range(0, nbands):
                bindat = f.read(window.W * nbytes)
                byte_index = f.tell()
                f.seek(byte_index + col_end_buffer)
                row = np.frombuffer(bindat, dtype=dtype)
                window_data[i, :, j] = row
                byte_index = f.tell()

    if window_data.shape[1] == 320:
        window_data = window_data[:, ::-1, :]

    return window_data


def get_wavelengths(
    file_config: M3FileManager | None = None,
    rfl_hdr: Optional[pathlike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a list of wavelengths from reflectance header file.

    Parameters
    ----------
    file_config: M3FileManager
        M3 File Config object.
    rfl_hdr: str | os.PathLike
        Path to reflectance header file.

    Returns
    -------
    all_wavelengths: np.ndarray
        List of all raw M3 wavelengths for the data level
    bbl: np.ndarray
        Bad band list. Index of good bands.

    Examples
    --------
    ```
    >>> mngr = m3.M3FileManager(root, data_id)
    >>> wvl, bbl = m3.io.get_wavelengths(mngr)
    >>> good_wvl = wvl[bbl]
    ```
    """
    if file_config is not None:
        rfl_hdr_path = file_config.pds_dir.l2.rfl_hdr
    elif rfl_hdr is not None:
        rfl_hdr_path = Path(rfl_hdr)
    else:
        raise ValueError("Either `file_config` or `rfl_hdr` must be provided.")

    wvl_key = re.compile(r"wavelength\s*=\s*{([\s\S]*?)}")  # wavelength list
    bbl_key = re.compile(r"bbl\s*=\s*{([\s\S]*?)}")  # Band Bands List

    def parse_list(file_read: str, pattern: re.Pattern) -> list[str]:
        result = re.search(pattern, file_read)
        if result is None:
            raise ValueError(f"Invalid HDR Format at: {rfl_hdr_path}")
        list_str = result.groups()[0]
        num_list = re.split(r",\s*\n*", list_str)
        return num_list

    with open(rfl_hdr_path, "r") as f:
        fread = f.read()
        wavelengths = [float(i) for i in parse_list(fread, wvl_key)]
        bbl = [int(i) for i in parse_list(fread, bbl_key)]
    return np.array(wavelengths, dtype=np.float32), np.array(bbl, dtype=bool)
