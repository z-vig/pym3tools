import cubio as cb
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def gaussian_column_filter(arr: np.ndarray) -> np.ndarray:
    return gaussian_filter1d(arr, 1, axis=1)


cc, cd = cb.cube_from_json(
    "D:/moon_data/m3/Gruithuisen_Region/M3T20090418T020644/pds_data/L1/M3T20090418T020644_V03_RDN.json"
)
cd.transpose_to("BIP")

arr = cd.array.values
arr_corr = gaussian_column_filter(arr)

plt.figure()
plt.imshow(arr[:, :, 10], cmap="Grays_r")
plt.title("Original")

plt.figure()
plt.imshow(arr_corr[:, :, 10], cmap="Grays_r")
plt.title("Column Correction")

plt.figure()
plt.plot(np.mean(arr[:, :, 10], axis=0), color="k", label="Original")
plt.plot(
    np.mean(arr_corr[:, :, 10], axis=0), color="r", label="Column Correction"
)
plt.legend()

plt.figure()
plt.plot(cc.measurement_values, arr[30, 30, :], color="k", label="Original")
plt.plot(
    cc.measurement_values,
    arr_corr[30, 30, :],
    color="r",
    label="Column Correction",
)
plt.legend()

plt.show()
