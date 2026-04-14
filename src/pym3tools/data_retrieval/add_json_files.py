from cubio import cubedata_from_envi_file
from pathlib import Path
from .data_directory import M3DataPaths
from pym3tools2.types import DataLevel


def _write_json(fp: Path, name: str):
    raw_ctxt, raw_cube = cubedata_from_envi_file(fp, name)
    raw_ctxt.write_envi_hdr(fp)


# Configuration mapping: (attribute_name, data_name) pairs
DATA_LEVEL_CONFIG: dict[DataLevel, list[tuple[str, str]]] = {
    "L0": [("raw", "RawData")],
    "L1B": [("rdn", "Radiance"), ("obs", "Geometry"), ("loc", "Location")],
    "L2": [("rfl", "Reflectance"), ("sup", "Supplemental")],
}


def add_json_to_level(file_catalog: M3DataPaths, level: DataLevel) -> None:
    """Process JSON files for a given data level."""
    if level not in DATA_LEVEL_CONFIG:
        raise ValueError(f"Unknown data level: {level}")

    for attr_name, data_name in DATA_LEVEL_CONFIG[level]:
        img_path = getattr(file_catalog, attr_name).img
        _write_json(img_path, data_name)


# Backward compatibility (optional, can be removed)
def add_json_to_l0(file_catalog: M3DataPaths) -> None:
    add_json_to_level(file_catalog, "L0")


def add_json_to_l1(file_catalog: M3DataPaths) -> None:
    add_json_to_level(file_catalog, "L1B")


def add_json_to_l2(file_catalog: M3DataPaths) -> None:
    add_json_to_level(file_catalog, "L2")
