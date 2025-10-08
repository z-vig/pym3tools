# Standard Libraries
import os
from pathlib import Path
from typing import Callable, Mapping, Optional, List
from dataclasses import dataclass

# Dependencies
import numpy as np
import rasterio as rio  # type: ignore
from rasterio.control import GroundControlPoint  # type: ignore

# Type Aliases
PathLike = str | os.PathLike | Path
MapGCPFunction = Callable[[PathLike, PathLike], List[GroundControlPoint]]
PixelGCPFunction = Callable[[PathLike], List[GroundControlPoint]]


def gcps_from_arcgis(
    points_path: PathLike, src_path: PathLike
) -> list[GroundControlPoint]:
    """
    Returns a list of GCP objects from reading a .points file returned from
    a hand-georeference in ArcGIS.
    """
    src_path = Path(src_path)
    points_path = Path(points_path)

    gcps_in = np.loadtxt(points_path)
    gcps_out = []
    with rio.open(src_path) as ds:
        for i in range(gcps_in.shape[0]):
            xpixel, ypixel = ds.index(gcps_in[i, 0], gcps_in[i, 1])
            gcps_out.append(
                GroundControlPoint(
                    row=xpixel, col=ypixel, x=gcps_in[i, 2], y=gcps_in[i, 3]
                )
            )

    convert_gcps(points_path, src_path)

    return gcps_out


def gcps_from_qgis(
    points_path: PathLike, src_path: PathLike
) -> list[GroundControlPoint]:
    """
    Returns a list of GCP objects from reading a .points file returned from
    a hand-georeference in ArcGIS.
    """
    src_path = Path(src_path)
    points_path = Path(points_path)

    with open(points_path) as f:
        lines = f.readlines()

    nrow = len(lines) - 2
    gcps_in = np.empty((nrow, 4), dtype=np.float32)
    for n, i in enumerate(lines[2:]):
        row = np.array([float(j) for j in i.split(",")][:4])
        gcps_in[n, :] = row

    gcps_out = []
    with rio.open(src_path) as ds:
        print(f"WIDTH: {ds.width}")
        print(f"HEIGHT: {ds.height}")
        origin_x, pixel_width, _, origin_y, _, pixel_height = (
            ds.transform.to_gdal()
        )

        for j in range(gcps_in.shape[0]):
            xpixel = (gcps_in[int(j), 2] - origin_x) / pixel_width
            ypixel = (gcps_in[int(j), 3] - origin_y) / pixel_height
            # xpixel, ypixel = ds.index(gcps_in[i, 0], gcps_in[i, 1])
            gcps_out.append(
                GroundControlPoint(
                    row=ypixel,
                    col=xpixel,
                    x=gcps_in[int(j), 0],
                    y=gcps_in[int(j), 1],
                )
            )

    convert_gcps(points_path, src_path)

    return gcps_out


def read_gcps(
    gcps_path: PathLike, header_length: int = 7
) -> list[GroundControlPoint]:
    """
    Reads the Ground Control Points found in a *.gcps file.
    """
    arr = np.loadtxt(
        gcps_path,
        skiprows=header_length,
        delimiter=",",
        usecols=[i for i in range(1, 5)],
    )
    ids = np.loadtxt(gcps_path, usecols=5, skiprows=header_length, dtype=str)
    gcp_list = []
    for i in range(arr.shape[0]):
        gcp = GroundControlPoint(*arr[i, :], id=str(ids[i]))
        gcp_list.append(gcp)
    return gcp_list


@dataclass
class GCPSMetadata:
    src_path: str
    row_offset: int
    col_offset: int
    height: int
    width: int
    band: int


def read_gcps_header(
    gcps_path: PathLike, header_length: int = 7
) -> GCPSMetadata:
    """
    Reads the header and returns the metadata found in a *.gcps file.
    """
    with open(gcps_path, "r") as f:
        lines = [line.strip() for line in f.readlines()[: header_length - 1]]

    # Extract values after the first colon and space
    values = [line.split(": ", 1)[1] for line in lines]
    src_path = values[0]
    row_offset, col_offset, height, width, band = map(int, values[1:])
    return GCPSMetadata(src_path, row_offset, col_offset, height, width, band)


def convert_gcps(
    gcps_path: PathLike, src_path: PathLike, dst_path: PathLike | None = None
) -> None:
    """
    Converts a list of Ground Control Points from a purely map-based format
    as is output from ArcGIS or QGIS to a pixel to map format with the
    file extension *.gcps.

    Parameters
    ----------
    src_path: PathLike
        Source Image path. This is the image that was used to perform a
        georeferencing operation in ArcGIS or QGIS.
    gcps_path: PathLike
        Path to export Ground Control Points. Either .txt or .points format.
    dst_path: PathLike, optional
        Path to save the new *.gcps file to. If None (default), the file
        takes the same name as the gcps_path just with a new suffix.
    """
    src_path = Path(src_path)
    gcps_path = Path(gcps_path)
    if dst_path is not None:
        dst_path = Path(dst_path).with_suffix(".gcps")
    else:
        dst_path = Path(src_path.parent, src_path.with_suffix(".gcps"))

    if gcps_path.suffix == ".points":
        gcp_list = gcps_from_qgis(src_path, gcps_path)
    elif gcps_path.suffix == ".txt":
        gcp_list = gcps_from_arcgis(src_path, gcps_path)
    elif gcps_path.suffix == ".gcps":
        print("GCPs already in the *.gcps format.")
        return
    else:
        raise ValueError(f"{gcps_path} is not in a recognized GCP format.")

    with open(dst_path, "w") as f:
        f.write(f"Source Image located at: {src_path}\n")
        f.write("index, pixel_row, pixel_col, map_x, map_y, id\n")
        for n, i in enumerate(gcp_list):
            f.write(f"{n}, {i.row}, {i.col}, {i.x}, {i.y}, {i.id}\n")


MAP_GCP_LOADERS: Mapping[str, MapGCPFunction] = {
    ".points": gcps_from_qgis,
    ".txt": gcps_from_arcgis,
}

PIXEL_GCP_LOADERS: Mapping[str, PixelGCPFunction] = {".gcps": read_gcps}


def load_gcps(gcps_path: PathLike, src_path: Optional[PathLike] = None):
    """
    Loads Ground Control Points from a file. Known file tpyes include:
    - `.points`
    - `.txt`
    - `.gcps`

    Parameters
    ----------
    gcps_path: PathLike
        Path to Ground Control Point File.
    src_path: PathLike
        Path to source image.
    """
    gcps_path = Path(gcps_path)
    file_ext = gcps_path.suffix

    if file_ext in MAP_GCP_LOADERS:
        if src_path is None:
            raise ValueError(f"{gcps_path} requires a source image to load.")
        return MAP_GCP_LOADERS[file_ext](gcps_path, src_path)
    if file_ext in PIXEL_GCP_LOADERS:
        return PIXEL_GCP_LOADERS[file_ext](gcps_path)

    raise ValueError(f"{file_ext} is not a supported GCP file.")
