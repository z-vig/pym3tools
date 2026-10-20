import numpy as np


def column_mean_correction(arr: np.ndarray) -> np.ndarray:
    """
    Subtracts correction factors from columns so that the means of all the
    columns are equal. Array must be in BIP format.
    """

    col_means = np.nanmean(arr, axis=(0, 2))
    correction_offsets = col_means - col_means.mean()

    corr_arr = arr - correction_offsets[None, :, None]

    return corr_arr
