# flake8: noqa
from pathlib import Path
from typing import Self

from pydantic import BaseModel
from pym3tools.types import MosaicMethod


class MosaicConfig(BaseModel):
    rfl_fp_list: list[str]
    method: MosaicMethod
    inc_fp_list: list[str]
    photometry_cube: bool

    @classmethod
    def from_json(cls, fp: str | Path) -> Self:
        with open(fp, "r") as f:
            cfg = cls.model_validate_json(f.read())
        return cfg


global_config = MosaicConfig(
    rfl_fp_list=[
        "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T160125/products/M3G20090208T160125_rfl.json",
        "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T175211/products/M3G20090208T175211_rfl.json",
        "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T194335/products/M3G20090208T194335_rfl.json",
    ],
    method="Mean",
    inc_fp_list=[
        "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T160125/products/M3G20090208T160125_photo.json",
        "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T175211/products/M3G20090208T175211_photo.json",
        "D:/moon_data/m3/Gruithuisen_Region/M3G20090208T194335/products/M3G20090208T194335_photo.json",
    ],
    photometry_cube=True,
)

targeted_config = MosaicConfig(
    rfl_fp_list=[
        "D:/moon_data/m3/Gruithuisen_Region/M3T20090418T020644/products/M3T20090418T020644_rfl.json",
        "D:/moon_data/m3/Gruithuisen_Region/M3T20090418T020848/products/M3T20090418T020848_rfl.json",
    ],
    method="Mean",
    inc_fp_list=[
        "D:/moon_data/m3/Gruithuisen_Region/M3T20090418T020644/products/M3T20090418T020644_photo.json",
        "D:/moon_data/m3/Gruithuisen_Region/M3T20090418T020848/products/M3T20090418T020848_photo.json",
    ],
    photometry_cube=True,
)


if __name__ == "__main__":
    with open(Path(__file__).parent / "global_config.json", "w") as f:
        f.write(global_config.model_dump_json(indent=2))
    with open(Path(__file__).parent / "targeted_config.json", "w") as f:
        f.write(targeted_config.model_dump_json(indent=2))
