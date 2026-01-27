# Built-Ins
from pathlib import Path
import re


class HDRFormatError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def add_wavelength_field(band_lbls: list[float] | list[str], dst_path: Path):
    """Adds wavelength field to ENVI HDR file."""
    hdr_lines = [
        "wavelength units = nm",
        "wavelength = {" + ",\n".join(map(str, band_lbls)) + "}",
    ]
    with open(Path(dst_path).with_suffix(".hdr"), "a") as f:
        f.write("\n".join(hdr_lines))


def add_bbl_field(bbl: list[bool], dst_path: Path):
    bbl_int: list[int] = [int(i) for i in bbl]
    bbl_hdr_line = "\nbbl = {" + ",\n".join(map(str, bbl_int)) + "}\n"
    with open(Path(dst_path).with_suffix(".hdr"), "a") as f:
        f.write(bbl_hdr_line)


def replace_band_labels_field(
    band_lbls: list[float] | list[str], dst_path: Path
):
    """
    Replaces the rasterio-generated Band Labels field in the ENVI HDR file.

    Raises
    ------
    HDRFormatError
        If ENVI HDR file does not already contain a "band names" field.
    """
    patt = re.compile(r"band names = {[\d\D]+}\n")

    new_band_names = "band names = {\n"
    for lbl in band_lbls:
        new_band_names += f"{lbl},\n"
    new_band_names += "}\n"

    with open(Path(dst_path).with_suffix(".hdr"), "r") as f:
        fread = f.read()

    search = re.search(patt, fread)
    if search is not None:
        match_slice = slice(*search.span(), 1)

        new_fread = fread.replace(fread[match_slice], new_band_names)

        with open(Path(dst_path).with_suffix(".hdr"), "w") as f:
            f.write(new_fread)
    else:
        raise HDRFormatError(".hdr file is in an incompatible format.")
