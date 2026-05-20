# %%
from cubio import cube_from_json
import matplotlib.pyplot as plt
import numpy as np
from pym3tools.convert_targeted_to_global import targeted_to_global_spectral

TRG_RFL = (
    "D:/moon_data/m3/Gruithuisen_Region/M3T_GDOMES_MOSAIC/M3T_GRUIT_RFL.json"
)
GLB_RFL = (
    "D:/moon_data/m3/Gruithuisen_Region/M3G_GDOMES_MOSAIC/M3G_GRUIT_RFL.json"
)
cc, cd = cube_from_json(TRG_RFL)
cd.mask.add_to_zmask(cc.get_bbl_mask())
cd.transpose_to("BIP")
glb_cc, glb_cd = cube_from_json(GLB_RFL)
glb_cd.transpose_to("BIP")
global_mode = targeted_to_global_spectral(cd.array.values)

rng = np.random.default_rng()

# %%
for _ in range(5):
    X = rng.choice(np.arange(0, cc.shape.ncolumns))
    Y = rng.choice(np.arange(0, cc.shape.nrows))
    PT = (Y, X)

    plt.figure()
    plt.plot(cc.measurement_values, cd.array[*PT, :])
    plt.plot(glb_cc.measurement_values, global_mode[*PT, :])
    # plt.plot(cc.measurement_values, filtered[*PT, :], marker=".")
    # plt.vlines(cc.measurement_values[slice(0, 28, 4)], 0, 0.18)
    # plt.vlines(cc.measurement_values[slice(28, 112, 2)], 0, 0.18)
    # plt.vlines(cc.measurement_values[slice(112, 256, 4)], 0, 0.18)

    # for n in np.arange(0, 28, 4):
    #     spec = cd.array[*PT, :]
    #     wvl = cc.measurement_values
    #     slc = slice(n, n + 4)
    #     yval = np.mean(spec[slc])
    #     yerr = np.std(spec[slc])
    #     xval = np.mean(wvl[slc])
    #     plt.errorbar(xval, yval, yerr, color="k", capsize=4, marker=".")

    # for n in np.arange(28, 112, 2):
    #     spec = cd.array[*PT, :]
    #     wvl = cc.measurement_values
    #     slc = slice(n, n + 2)
    #     yval = np.mean(spec[slc])
    #     yerr = np.std(spec[slc])
    #     xval = np.mean(wvl[slc])
    #     plt.errorbar(xval, yval, yerr, color="k", capsize=4, marker=".")

    # for n in np.arange(112, 256, 4):
    #     spec = cd.array[*PT, :]
    #     wvl = cc.measurement_values
    #     slc = slice(n, n + 4)
    #     yval = np.mean(spec[slc])
    #     yerr = np.std(spec[slc])
    #     xval = np.mean(wvl[slc])
    #     plt.errorbar(xval, yval, yerr, color="k", capsize=4, marker=".")
plt.show()
