from dataclasses import dataclass
from datetime import datetime
from typing import Self
import re

from pym3tools2.types import (
    AcquisitionMode,
    acq_char_to_mode,
    acq_mode_to_char,
    M3ImageType,
    is_valid_m3_image_type,
    OpticalPeriod,
)

from .data_ids import DataIDString, is_valid_data_id


@dataclass
class M3DataID:
    acquistion_mode: AcquisitionMode
    acquisition_datetime: datetime
    image_type: M3ImageType | None
    version: int | None

    @classmethod
    def from_string(cls, id_str: str) -> Self:
        data_id_pattern = re.compile(
            r"""
            ^
            M3
            (?P<acquisition_mode>[G|T])
            (?P<year>\d{4})
            (?P<month>\d{2})
            (?P<day>\d{2})
            T
            (?P<hour>\d{2})
            (?P<minute>\d{2})
            (?P<second>\d{2})
            (?:_V[0]+?(?P<version>\d+))?
            (?:_(?P<image_type>\w+))?
            $
            """,
            re.VERBOSE,
        )
        matching = re.match(data_id_pattern, id_str)
        if matching is None:
            raise ValueError("Invalid Data ID")
        result = matching.groupdict()
        acq_mode = result.get("acquisition_mode")
        img_type = result.get("image_type")
        version = result.get("version")

        if not isinstance(acq_mode, str):
            raise ValueError("Invalid Data ID 1")
        acq_mode = acq_char_to_mode.get(acq_mode)
        if acq_mode is None:
            raise ValueError("Invalid Data ID 2")

        if img_type is not None:
            if not is_valid_m3_image_type(img_type):
                raise ValueError("Invalid Data ID 3")

        acq_dt = datetime(
            int(result["year"]),
            int(result["month"]),
            int(result["day"]),
            int(result["hour"]),
            int(result["minute"]),
            int(result["second"]),
        )

        if version is not None:
            version = int(version)

        return cls(acq_mode, acq_dt, img_type, version)

    @property
    def string(self) -> DataIDString:
        id_string = (
            f"M3{acq_mode_to_char[self.acquistion_mode]}"
            f"{self.acquisition_datetime.year:02}"
            f"{self.acquisition_datetime.month:02}"
            f"{self.acquisition_datetime.day:02}T"
            f"{self.acquisition_datetime.hour:02}"
            f"{self.acquisition_datetime.minute:02}"
            f"{self.acquisition_datetime.second:02}"
        )
        if not is_valid_data_id(id_string):
            raise ValueError("Invalid Data ID")
        return id_string

    @property
    def op(self) -> OpticalPeriod:
        if (
            datetime(2008, 10, 18)
            < self.acquisition_datetime
            <= datetime(2009, 1, 9)
        ):
            return "OP1A"
        elif (
            datetime(2009, 1, 9)
            < self.acquisition_datetime
            <= datetime(2009, 2, 15)
        ):
            return "OP1B"
        elif (
            datetime(2009, 4, 15)
            < self.acquisition_datetime
            <= datetime(2009, 4, 28)
        ):
            return "OP2A"
        elif (
            datetime(2009, 5, 13)
            < self.acquisition_datetime
            <= datetime(2009, 5, 17)
        ):
            return "OP2B"
        elif (
            datetime(2009, 5, 20)
            < self.acquisition_datetime
            <= datetime(2009, 8, 17)
        ):
            return "OP2C"

        raise ValueError(
            f"{self.acquisition_datetime} is not in the acceptable mission "
            "time range."
        )
