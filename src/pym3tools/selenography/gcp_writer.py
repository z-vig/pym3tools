# Standard Libraries
from uuid import uuid4
import shutil

# Dependencies
import numpy as np

# Top-Level Imports
from pym3tools.types import PathLike


def write_gcp_file_from_loc(
    loc_image: np.ndarray,
    save_path: PathLike,
    source_image: PathLike,
    row_offset: int,
    col_offset: int,
    height: int,
    width: int,
    band: int,
    ngcps: int = 50,
    save_gcps_copy: bool = True,
):
    """
    Writes a *.gcps file using the four corners plus 50 random points from the
    M3 backplane. These control points will contain a known offset.
    """

    with open(save_path, "w") as f:
        f.write(f"Source for Target Image: {source_image}\n")
        f.write(f"Row Offset: {row_offset}\n")
        f.write(f"Column Offset: {col_offset}\n")
        f.write(f"Target Image Height: {height}\n")
        f.write(f"Target Image Width: {width}\n")
        f.write(f"Target Image Band Used: {band}\n")
        f.write("index, pixel_row, pixel_col, map_x, map_y, ID\n")

        rng = np.random.default_rng()
        rand_row = rng.choice(np.arange(0, height), ngcps, replace=False)
        rand_col = rng.choice(np.arange(0, width), ngcps, replace=False)
        gcps = loc_image[rand_row + row_offset, rand_col + col_offset, :2]
        for n in range(ngcps):
            f.write(
                f"{n}, {rand_row[n]}, {rand_col[n]}, {gcps[n, 0]},"
                f" {gcps[n, 1]}, {uuid4()}\n"
            )
        corner_pts = [
            (0, 0),
            (height - 1, 0),
            (0, width - 1),
            (height - 1, width - 1),
        ]
        for n, c in enumerate(corner_pts):
            i, j = c
            f.write(
                f"{ngcps + n}, {i}, {j}, "
                f"{loc_image[i+row_offset, j+col_offset, 0]},"
                f" {loc_image[i+row_offset, j+col_offset, 1]}, {uuid4()}\n"
            )

    if save_gcps_copy:
        shutil.copy(save_path, "./debug_gcps.txt")
