from math import ceil

import numpy as np
from cubio.cube_context.envi_hdr_tools import (
    extract_hdr_bbl,
    extract_hdr_wavelengths,
)
from scipy.ndimage import uniform_filter1d

from pym3tools import M3DataPaths
from pym3tools.convert_targeted_to_global import (
    GLOBAL_RESAMPLING,
    N_GLOBAL_CHANNELS,
    slice_target_to_global,
)

glb = M3DataPaths("D:/moon_data/m3/Gruithuisen_Region/M3G20090208T194335")
trg = M3DataPaths("D:/moon_data/m3/Gruithuisen_Region/M3T20090418T020644/")

glbwvl = np.array(extract_hdr_wavelengths(glb.rdn.hdr))
trgwvl = np.array(extract_hdr_wavelengths(trg.rdn.hdr))

trgbbl = np.array(extract_hdr_bbl(trg.rfl.hdr))


def apply_filter_1d(data: np.ndarray, filter_width: int, add_offset: bool = True):
    filter_offset = -filter_width // 2
    if not add_offset:
        filter_offset = 0
    filtered_arr = uniform_filter1d(
        data,
        filter_width,
        mode="nearest",
        origin=filter_offset,
    )
    return filtered_arr


def targeted_to_global_wavelengths(
    targeted_wavelengths: np.ndarray | list[float],
):
    trgwvl = np.array(targeted_wavelengths)
    position = 0
    glb_wvl_recreation = np.zeros(N_GLOBAL_CHANNELS)
    for slc, resamp in GLOBAL_RESAMPLING.items():
        piece = trgwvl[slc]
        newslc, length = slice_target_to_global(slc, resamp, position)
        position += length
        filtered = apply_filter_1d(piece, resamp)
        downsamp_slice = slice(0, slc.stop, resamp)
        glb_wvl_recreation[newslc] = filtered[downsamp_slice]

    return glb_wvl_recreation


trg_float_bbl = [float(i) for i in trgbbl]
test_bbl = targeted_to_global_wavelengths(trg_float_bbl)
print([ceil(i) for i in test_bbl])
# glb_wvl_recreation = targeted_to_global_wavelengths(trgwvl)
# for i, j in zip(glbwvl, glb_wvl_recreation):
#     print(f"Actual: {i:.2f}, Recreated: {j:.2f}, Difference: {i-j:.2f}")
