import numpy as np


def boolean_to_slice(mask: np.ndarray) -> slice:
    """
    Convert a 1D boolean array into a slice, ensuring all True values
    are contiguous.

    Parameters
    ----------
    mask : np.ndarray
        1D boolean array

    Returns
    -------
    slice
        Slice covering the contiguous True region

    Raises
    ------
    ValueError
        If mask is not 1D, not boolean, has no True values, or
        True values are not contiguous.
    """
    if mask.ndim != 1:
        raise ValueError("mask must be 1D")

    if mask.dtype != bool:
        raise ValueError("mask must be of boolean dtype")

    true_idx = np.flatnonzero(mask)

    if true_idx.size == 0:
        raise ValueError("mask contains no True values")

    # Check contiguity: differences must all be 1
    if not np.all(np.diff(true_idx) == 1):
        raise ValueError("True values are not contiguous")

    start = true_idx[0]
    stop = true_idx[-1] + 1  # slice stop is exclusive

    return slice(start, stop)
