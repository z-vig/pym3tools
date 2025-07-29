# Standard Libraries
from dataclasses import dataclass
import re

M3_PDS_ROOT = r"https://planetarydata.jpl.nasa.gov/img/data/m3/"

DATA_ID_PATTERN = r"M3\w\d{8}T\d{6}"


def get_m3_id(input_string: str) -> list | str | None:
    pattern = re.compile(r".*(M3\w\d{8}T\d{6}).*")
    located_ids = re.findall(pattern, input_string)
    if len(located_ids) == 0:
        return None
    elif len(located_ids) == 1:
        return located_ids[0]
    else:
        print("Multilpe M3 Data IDs found.")
        return located_ids


@dataclass
class FileRetrievalPatterns():
    level0: re.Pattern = re.compile(
        f"{M3_PDS_ROOT}"
        r"CH1M3_0001/DATA/\d{8}_\d{8}/\d{6}/L0/"
        rf"{DATA_ID_PATTERN}_V01_L0.(?:HDR|IMG|LBL)"
    )
    level1: re.Pattern = re.compile(
        f"{M3_PDS_ROOT}"
        r"CH1M3_0003/DATA/\d{8}_\d{8}/\d{6}/L1B/"
        rf"{DATA_ID_PATTERN}_V03_(?:L1B|LOC|OBS|RDN).(?:HDR|IMG|LBL)"
    )
    level2: re.Pattern = re.compile(
        f"{M3_PDS_ROOT}"
        r"CH1M3_0004/DATA/\d{8}_\d{8}/\d{6}/L2/"
        rf"{DATA_ID_PATTERN}_V01_(?:L2|RFL|SUP).(?:HDR|IMG|LBL)"
    )
    global_caldata: re.Pattern = re.compile(
        f"{M3_PDS_ROOT}"
        r"CH1M3_0004/CALIB/M3G\d{8}_RFL_.*"
    )
    targeted_caldata: re.Pattern = re.compile(
        f"{M3_PDS_ROOT}"
        r"CH1M3_0004/CALIB/M3T\d{8}_RFL_.*"
    )
    acq_type: re.Pattern = re.compile(r"M3(G|T)\d{8}T\d{6}")
    short_id: re.Pattern = re.compile(r"M3\w\d{8}T(\d{6})")
