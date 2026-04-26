from pym3tools.m3catalog import DataIDString
from pym3tools.data_retrieval import download_m3_dataset

from pathlib import Path

sample_ids: list[DataIDString] = [
    "M3T20090418T020644",
    "M3T20090418T020848",
    "M3G20090208T160125",
    "M3G20090208T175211",
    "M3G20090208T194335",
]

for i in sample_ids:
    download_m3_dataset(Path("D:/moon_data/m3/Gruithuisen_Region/"), i)
