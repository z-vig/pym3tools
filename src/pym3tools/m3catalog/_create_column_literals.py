"""
Generates index data files as python files.
"""

import re
from importlib.resources import files
from pathlib import Path


def _generate_l0enum(savepath: Path):
    l2_idx = files("pym3tools.m3catalog.index_data").joinpath("L0_INDEX_OP1.LBL")
    col_obj_header = r"\sOBJECT\s*=\sCOLUMN(?:.|\n)*?"
    col_name = re.compile(col_obj_header + r"NAME\s*=\s\"?(\w+(?::\w+)?)")
    with l2_idx.open() as f:
        fread = f.read()
        col_names = [i for i in re.findall(col_name, fread)]
    with open(savepath, "w") as f:
        f.write("from typing import Literal\n\n")
        f.write("type L0ColumnName = Literal[\n")
        f.writelines(f'    "{i}",\n' for i in col_names)
        f.write("]\n\n")
        f.write("l0_column_names: list[L0ColumnName] = [\n")
        f.writelines(f'    "{i}",\n' for i in col_names)
        f.write("]")
        f.write("\n")


def _generate_l1benum(savepath: Path):
    l1B_idx = files("pym3tools.m3catalog.index_data").joinpath("L1B_INDEX.LBL")
    col_obj_header = r"\sOBJECT\s*=\sCOLUMN(?:.|\n)*?"
    col_name = re.compile(col_obj_header + r"NAME\s*=\s\"?(\w+(?::\w+)?)")
    with l1B_idx.open() as f:
        fread = f.read()
        col_names = [i for i in re.findall(col_name, fread)]
    with open(savepath, "w") as f:
        f.write("from typing import Literal\n\n")
        f.write("type L1BColumnName = Literal[\n")
        f.writelines(f'    "{i}",\n' for i in col_names)
        f.write("]\n\n")
        f.write("l1b_column_names: list[L1BColumnName] = [\n")
        f.writelines(f'    "{i}",\n' for i in col_names)
        f.write("]")
        f.write("\n")


def _generate_l2enum(savepath: Path):
    l2_idx = files("pym3tools.m3catalog.index_data").joinpath("L2_INDEX.LBL")
    col_obj_header = r"\sOBJECT\s*=\sCOLUMN(?:.|\n)*?"
    col_name = re.compile(col_obj_header + r"NAME\s*=\s\"?(\w+(?::\w+)?)")
    with l2_idx.open() as f:
        fread = f.read()
        col_names = [i for i in re.findall(col_name, fread)]
    with open(savepath, "w") as f:
        f.write("from typing import Literal\n\n")
        f.write("type L2ColumnName = Literal[\n")
        f.writelines(f'    "{i}",\n' for i in col_names)
        f.write("]\n\n")
        f.write("l2_column_names: list[L2ColumnName] = [\n")
        f.writelines(f'    "{i}",\n' for i in col_names)
        f.write("]")
        f.write("\n")


def generate_all_enums(savepath: Path):
    _generate_l0enum(savepath / "l0_columns_names.py")
    _generate_l1benum(savepath / "l1b_columns_names.py")
    _generate_l2enum(savepath / "l2_columns_names.py")


if __name__ == "__main__":
    cwd = Path(__file__).parent.resolve()
    generate_all_enums(cwd)
