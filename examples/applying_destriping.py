from pathlib import Path

import cubio as cb
import matplotlib.pyplot as plt
import numpy as np

from pym3tools.destriping import column_mean_correction


def apply_destriping(rfl_json: str | Path) -> None:
    rfl_json = Path(rfl_json)

    original_cc, original_cd = cb.cube_from_json(rfl_json)

    destriped = column_mean_correction(original_cd.array.values)

    destriped_cc, destriped_cd = cb.cube_from_numpy(
        destriped,
        original_cd.fmt,
        original_cc,
        name=f"{rfl_json.stem}_destriped",
    )

    cb.write_envi(destriped_cc, destriped_cd, "BIL", rfl_json.parent)


def compare_destriped_to_original(
    original_json: str | Path, destriped_json: str | Path
) -> None:
    _original_cc, original_cd = cb.cube_from_json(original_json)
    _destriped_cc, destriped_cd = cb.cube_from_json(destriped_json)

    original_cube = original_cd.array.values
    destriped_cube = destriped_cd.array.values

    for i in [original_cube, destriped_cube]:
        i[i < 0] = np.nan

    plt.figure()
    plt.imshow(original_cd.array.values[:, 19, :], cmap="Grays_r")
    plt.title("Original")

    plt.figure()
    plt.imshow(destriped_cd.array.values[:, 19, :], cmap="Grays_r")
    plt.title("Destriped")

    plt.show()


if __name__ == "__main__":
    # apply_destriping(
    #     "D:/moon_data/m3/Gruithuisen_Region/M3T20090418T020644/products/M3T20090418T020644_rfl.json"
    # )
    compare_destriped_to_original(
        "D:/moon_data/m3/Gruithuisen_Region/M3T20090418T020644/products_withstriping/M3T20090418T020644_rfl.json",
        "D:/moon_data/m3/Gruithuisen_Region/M3T20090418T020644/products/M3T20090418T020644_rfl.json",
    )
