# Standard Libraries
from importlib.resources import files
from dataclasses import dataclass
import re
from pathlib import Path

# Relative Imports
from .l1bindex import L1BIndex
from .l2index import L2Index
from .l0index import L0Index

# Top-Level Imports
from pym3tools.types import PathLike


class DataIDNotFoundError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class IndexFile:
    column_names: list[str]


@dataclass
class ColumnMetadata:
    index: int
    name: str
    data_type: str
    start_byte: int
    nbytes: int
    format: str
    description: str

    @classmethod
    def from_index(
        cls, index: L1BIndex | L2Index | L0Index
    ) -> "ColumnMetadata":
        if isinstance(index, L0Index):
            idx_file = files("pym3tools.PDSretrieval.data").joinpath(
                "L0_INDEX_OP1.LBL"
            )
        elif isinstance(index, L1BIndex):
            idx_file = files("pym3tools.PDSretrieval.data").joinpath(
                "L1B_INDEX.LBL"
            )
        elif isinstance(index, L2Index):
            idx_file = files("pym3tools.PDSretrieval.data").joinpath(
                "L2_INDEX.LBL"
            )

        col_obj_header = re.compile(r"\sOBJECT\s*=\sCOLUMN(?:.|\n)*?")
        with idx_file.open() as f:
            fread = f.read()
            match = re.finditer(col_obj_header, fread)
        start_bytes = [i.start() for i in match]
        start_bytes.append(len(fread))
        col_str = fread[
            start_bytes[index.value] : start_bytes[index.value + 1]  # noqa
        ]
        idx_pattern = re.compile(r"COLUMN_NUMBER\s+=\s(\d+)\s+\n")
        name_pattern = re.compile(r"NAME\s+=\s\"?([\w:]+)\"?\s+\n")
        dtype_pattern = re.compile(r"DATA_TYPE\s+=\s\"?(\w+)\"?\s+\n")
        start_byte_pattern = re.compile(r"START_BYTE\s+=\s+(\d+)\s+\n")
        nbytes_pattern = re.compile(r"BYTES\s+=\s+(\d+)\s+\n")
        format_pattern = re.compile(r"FORMAT\s+=\s\"?(\w+)\"?\s+\n")
        desc_pattern = re.compile(r"DESCRIPTION\s+=\s\"([^\"]+)\"\s+\n")

        # print(
        #     index,
        #     re.findall(idx_pattern, col_str),
        #     re.findall(name_pattern, col_str),
        #     re.findall(dtype_pattern, col_str),
        #     re.findall(start_byte_pattern, col_str),
        #     re.findall(nbytes_pattern, col_str),
        # )

        format_find = re.findall(format_pattern, col_str)
        if len(format_find) == 0:
            format = "N/A"
        else:
            format = format_find[0]

        desc = re.sub(
            re.compile(r"\s+"), " ", re.findall(desc_pattern, col_str)[0]
        )

        return cls(
            int(re.findall(idx_pattern, col_str)[0]),
            re.findall(name_pattern, col_str)[0],
            re.findall(dtype_pattern, col_str)[0],
            int(re.findall(start_byte_pattern, col_str)[0]) - 1,
            int(re.findall(nbytes_pattern, col_str)[0]),
            format,
            desc,
        )

    def get_entry(self, index_line: str):
        return index_line[
            self.start_byte : self.start_byte + self.nbytes  # noqa
        ].replace(" ", "")


def _generate_l1bindex_enum(savepath: PathLike):
    l1B_idx = files("pym3tools.PDSretrieval.data").joinpath("L1B_INDEX.LBL")
    col_obj_header = r"\sOBJECT\s*=\sCOLUMN(?:.|\n)*?"
    col_name = re.compile(col_obj_header + r"NAME\s*=\s\"?(\w+(?::\w+)?)")
    with l1B_idx.open() as f:
        fread = f.read()
        col_names = [i.replace(":", "_") for i in re.findall(col_name, fread)]
    with open(savepath, "w") as f:
        f.write("from enum import Enum\n\n\n")
        f.write("class L1BIndex(Enum):\n")
        for n, i in enumerate(col_names):
            f.write(f"    {i} = {n}\n")


def _generate_l2index_enum(savepath: PathLike):
    l2_idx = files("pym3tools.PDSretrieval.data").joinpath("L2_INDEX.LBL")
    col_obj_header = r"\sOBJECT\s*=\sCOLUMN(?:.|\n)*?"
    col_name = re.compile(col_obj_header + r"NAME\s*=\s\"?(\w+(?::\w+)?)")
    with l2_idx.open() as f:
        fread = f.read()
        col_names = [i.replace(":", "_") for i in re.findall(col_name, fread)]
    with open(savepath, "w") as f:
        f.write("from enum import Enum\n\n\n")
        f.write("class L2Index(Enum):\n")
        for n, i in enumerate(col_names):
            f.write(f"    {i} = {n}\n")


def _generate_l0index_enum(savepath: PathLike):
    l2_idx = files("pym3tools.PDSretrieval.data").joinpath("L0_INDEX_OP1.LBL")
    col_obj_header = r"\sOBJECT\s*=\sCOLUMN(?:.|\n)*?"
    col_name = re.compile(col_obj_header + r"NAME\s*=\s\"?(\w+(?::\w+)?)")
    with l2_idx.open() as f:
        fread = f.read()
        col_names = [i.replace(":", "_") for i in re.findall(col_name, fread)]
    with open(savepath, "w") as f:
        f.write("from enum import Enum\n\n\n")
        f.write("class L0Index(Enum):\n")
        for n, i in enumerate(col_names):
            f.write(f"    {i} = {n}\n")


def create_urls_file(data_id: str, savedir: PathLike) -> None:
    l0_idx_op1 = files("pym3tools.PDSretrieval.data").joinpath(
        "L0_INDEX_OP1.TAB"
    )
    l0_idx_op2 = files("pym3tools.PDSretrieval.data").joinpath(
        "L0_INDEX_OP2.TAB"
    )
    l1B_idx = files("pym3tools.PDSretrieval.data").joinpath("L1B_INDEX.TAB")
    l2_idx = files("pym3tools.PDSretrieval.data").joinpath("L2_INDEX.TAB")
    jpl_url = Path("https://planetarydata.jpl.nasa.gov/img/data/m3")

    l0_index_line = "None"
    with l0_idx_op1.open() as f:
        for line in f.readlines():
            if data_id in line:
                print(f"{data_id} is from OP1")
                l0_index_line = line
    with l0_idx_op2.open() as f:
        for line in f.readlines():
            if data_id in line:
                print(f"{data_id} is from OP2")
                l0_index_line = line

    with l1B_idx.open() as f:
        l1b_index_line = "None"
        for line in f.readlines():
            if data_id in line:
                l1b_index_line = line

    with l2_idx.open() as f:
        l2_index_line = "None"
        for line in f.readlines():
            if data_id in line:
                l2_index_line = line

    if l0_index_line == "None":
        raise DataIDNotFoundError(f"{data_id} is not in the L0 file index.")
    if l1b_index_line == "None":
        raise DataIDNotFoundError(f"{data_id} is not in the L1B file index.")
    if l2_index_line == "None":
        raise DataIDNotFoundError(f"{data_id} is not in the L2 file index.")

    l0_data_vol = ColumnMetadata.from_index(L0Index.VOLUME_ID)
    l0_lbl_f = ColumnMetadata.from_index(L0Index.FILE_SPECIFICATION_NAME)
    l0_data_f = ColumnMetadata.from_index(L0Index.PRODUCT_ID)

    l0_lbl_path = Path(
        jpl_url,
        l0_data_vol.get_entry(l0_index_line),
        l0_lbl_f.get_entry(l0_index_line),
    ).with_suffix(".LBL")
    l0_root = l0_lbl_path.parent
    raw_data_path = Path(l0_root, l0_data_f.get_entry(l0_index_line))

    l0_paths = [
        raw_data_path.with_suffix(".IMG"),
        raw_data_path.with_suffix(".HDR"),
    ]

    l1_data_vol = ColumnMetadata.from_index(L1BIndex.VOLUME_ID)
    l1_lbl_f = ColumnMetadata.from_index(L1BIndex.FILE_SPECIFICATION_NAME)
    loc_f = ColumnMetadata.from_index(L1BIndex.LOC_FILE)
    obs_f = ColumnMetadata.from_index(L1BIndex.OBS_FILE)
    rdn_f = ColumnMetadata.from_index(L1BIndex.PRODUCT_ID)

    l1B_lbl_path = Path(
        jpl_url,
        l1_data_vol.get_entry(l1b_index_line),
        l1_lbl_f.get_entry(l1b_index_line),
    ).with_suffix(".LBL")

    l1B_root = l1B_lbl_path.parent

    loc_path = Path(l1B_root, loc_f.get_entry(l1b_index_line))
    obs_path = Path(l1B_root, obs_f.get_entry(l1b_index_line))
    rdn_path = Path(l1B_root, rdn_f.get_entry(l1b_index_line))

    l1_paths = []
    for i in [loc_path, obs_path, rdn_path]:
        l1_paths.append(i.with_suffix(".IMG"))
        l1_paths.append(i.with_suffix(".HDR"))

    l2_data_vol = ColumnMetadata.from_index(L2Index.VOLUME_ID)
    l2_lbl_f = ColumnMetadata.from_index(L2Index.FILE_SPECIFICATION_NAME)
    rfl_f = ColumnMetadata.from_index(L2Index.RFL_IMAGE_FILE_NAME)
    sup_f = ColumnMetadata.from_index(L2Index.SUP_IMAGE_FILE_NAME)
    solspec_f = ColumnMetadata.from_index(L2Index.CH1_SOLAR_SPECTRUM_FILE_NAME)
    falpha_f = ColumnMetadata.from_index(L2Index.CH1_PHOTOMETRY_CORR_FILE_NAME)

    l2_lbl_path = Path(
        jpl_url,
        l2_data_vol.get_entry(l2_index_line),
        l2_lbl_f.get_entry(l2_index_line),
    ).with_suffix(".LBL")

    l2_root = l2_lbl_path.parent

    l2_cal_root_parts = list(l2_root.parts)[:-3]
    l2_cal_root_parts[-1] = "CALIB"
    l2_cal_root = Path(*l2_cal_root_parts)

    rfl_path = Path(l2_root, rfl_f.get_entry(l2_index_line))
    sup_path = Path(l2_root, sup_f.get_entry(l2_index_line))
    solspec_path = Path(l2_cal_root, solspec_f.get_entry(l2_index_line))
    falpha_path = Path(l2_cal_root, falpha_f.get_entry(l2_index_line))

    statpol_ids = {
        "G": [
            Path(l2_cal_root, "M3G20110830_RFL_STAT_POL_1"),
            Path(l2_cal_root, "M3G20110830_RFL_STAT_POL_2"),
        ],
        "T": [
            Path(l2_cal_root, "M3T20111020_RFL_STAT_POL_1"),
            Path(l2_cal_root, "M3T20111020_RFL_STAT_POL_2"),
        ],
    }

    gndtru_ids = [
        Path(l2_cal_root, f"{data_id[:3]}20111117_RFL_GRND_TRU_1"),
        Path(l2_cal_root, f"{data_id[:3]}20111117_RFL_GRND_TRU_2"),
    ]

    statpol_paths = []
    gndtru_paths = []
    for i in statpol_ids[data_id[2]]:
        statpol_paths.append(i.with_suffix(".TAB"))
        statpol_paths.append(i.with_suffix(".LBL"))
    for i in gndtru_ids:
        gndtru_paths.append(i.with_suffix(".TAB"))
        gndtru_paths.append(i.with_suffix(".LBL"))

    l2_paths = []
    cal_paths = [*statpol_paths, *gndtru_paths]
    for i in [rfl_path, sup_path]:
        l2_paths.append(i.with_suffix(".IMG"))
        l2_paths.append(i.with_suffix(".HDR"))
    for i in [solspec_path, falpha_path]:
        cal_paths.append(i.with_suffix(".LBL"))
        cal_paths.append(i.with_suffix(".TAB"))

    url_list: list[PathLike] = [
        l0_lbl_path,
        *l0_paths,
        l1B_lbl_path,
        *l1_paths,
        l2_lbl_path,
        *l2_paths,
        *cal_paths,
    ]

    save_file = Path(savedir, f"{data_id}_urls").with_suffix(".txt")
    with open(save_file, "w") as f:
        print("Saved to: ", save_file)
        for url in url_list:
            f.write(
                f"{str(url).
                   replace("https:\\", "https://").
                   replace("\\", "/")}\n"
            )
