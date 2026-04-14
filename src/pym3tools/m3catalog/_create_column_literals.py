from pathlib import Path
from importlib.resources import files
import re


def _generate_l0enum(savepath: Path):
    l2_idx = files("pym3tools2.m3catalog.index_data").joinpath(
        "L0_INDEX_OP1.LBL"
    )
    col_obj_header = r"\sOBJECT\s*=\sCOLUMN(?:.|\n)*?"
    col_name = re.compile(col_obj_header + r"NAME\s*=\s\"?(\w+(?::\w+)?)")
    with l2_idx.open() as f:
        fread = f.read()
        col_names = [i for i in re.findall(col_name, fread)]
    with open(savepath, "w") as f:
        f.write("from typing import Literal, TypeAlias\n\n\n")
        f.write("L0ColumnName: TypeAlias = Literal[\n")
        for i in col_names:
            f.write(f'    "{i}",\n')
        f.write("]\n\n")
        f.write("l0_column_names: list[L0ColumnName] = [\n")
        for i in col_names:
            f.write(f'    "{i}",\n')
        f.write("]")
        f.write("\n")


def _generate_l1benum(savepath: Path):
    l1B_idx = files("pym3tools2.m3catalog.index_data").joinpath(
        "L1B_INDEX.LBL"
    )
    col_obj_header = r"\sOBJECT\s*=\sCOLUMN(?:.|\n)*?"
    col_name = re.compile(col_obj_header + r"NAME\s*=\s\"?(\w+(?::\w+)?)")
    with l1B_idx.open() as f:
        fread = f.read()
        col_names = [i for i in re.findall(col_name, fread)]
    with open(savepath, "w") as f:
        f.write("from typing import Literal, TypeAlias\n\n\n")
        f.write("L1BColumnName: TypeAlias = Literal[\n")
        for i in col_names:
            f.write(f'    "{i}",\n')
        f.write("]\n\n")
        f.write("l1b_column_names: list[L1BColumnName] = [\n")
        for i in col_names:
            f.write(f'    "{i}",\n')
        f.write("]")
        f.write("\n")


def _generate_l2enum(savepath: Path):
    l2_idx = files("pym3tools2.m3catalog.index_data").joinpath("L2_INDEX.LBL")
    col_obj_header = r"\sOBJECT\s*=\sCOLUMN(?:.|\n)*?"
    col_name = re.compile(col_obj_header + r"NAME\s*=\s\"?(\w+(?::\w+)?)")
    with l2_idx.open() as f:
        fread = f.read()
        col_names = [i for i in re.findall(col_name, fread)]
    with open(savepath, "w") as f:
        f.write("from typing import Literal, TypeAlias\n\n\n")
        f.write("L2ColumnName: TypeAlias = Literal[\n")
        for i in col_names:
            f.write(f'    "{i}",\n')
        f.write("]\n\n")
        f.write("l2_column_names: list[L2ColumnName] = [\n")
        for i in col_names:
            f.write(f'    "{i}",\n')
        f.write("]")
        f.write("\n")


def generate_all_enums(savepath: Path):
    _generate_l0enum(savepath / "l0_columns_names.py")
    _generate_l1benum(savepath / "l1b_columns_names.py")
    _generate_l2enum(savepath / "l2_columns_names.py")


if __name__ == "__main__":
    cwd = Path(__file__).parent.resolve()
    print(cwd)
    generate_all_enums(cwd)
