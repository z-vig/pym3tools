# Standard Libraries
from pathlib import Path
import os
import re
from dataclasses import dataclass, field

# Relative Imports
from .retrieve_urls import retrieve_urls
from .file_retrieval_patterns import FileRetrievalPatterns

type pathlike = str | os.PathLike


@dataclass
class PDSDataFiles:
    root: pathlike
    lbl: pathlike
    level: int = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.lbl = Path(self.lbl)
        if not self.root.is_dir():
            print(f"Creating L{self.level} directory.")
            self.root.mkdir()
        if self.level not in (0, 1, 2):
            raise ValueError(f"{self.level} is not a valid processing level.")

    def __str__(self) -> str:
        tree_string = f"{Path(self.root).name}\n"
        n = 1
        for k, v in vars(self).items():
            n += 1
            if (k not in ("root", "level")) and (n != len(vars(self))):
                tree_string += (
                    "\u2502   \u2502   \u251c\u2500\u2500\u2500" f"{v.name}\n"
                )
            elif (k not in ("root", "level")) and (n == len(vars(self))):
                tree_string += (
                    "\u2502   \u2502   \u2514\u2500\u2500\u2500" f"{v.name}"
                )
        return tree_string


@dataclass
class L0Files(PDSDataFiles):
    hdr: os.PathLike
    img: os.PathLike

    def __post_init__(self):
        self.level = 0
        super().__post_init__()


@dataclass
class L1Files(PDSDataFiles):
    loc_img: os.PathLike
    loc_hdr: os.PathLike
    obs_img: os.PathLike
    obs_hdr: os.PathLike
    rdn_img: os.PathLike
    rdn_hdr: os.PathLike

    def __post_init__(self):
        self.level = 1
        super().__post_init__()


@dataclass
class L2Files(PDSDataFiles):
    rfl_hdr: os.PathLike
    rfl_img: os.PathLike
    sup_hdr: os.PathLike
    sup_img: os.PathLike

    def __post_init__(self):
        self.level = 2
        super().__post_init__()


class PDSDir:
    root: os.PathLike
    retrieval: os.PathLike
    l0: L0Files
    l1: L1Files
    l2: L2Files

    def __init__(
        self, parent: os.PathLike, data_id: str, verbose: bool = False
    ):
        self.root = Path(parent)
        self.retrieval = Path(self.root, f"{data_id}_urls.txt")

        with open(self.retrieval, "r") as f:
            fread = f.read()
            l0_urls = re.findall(FileRetrievalPatterns.level0, fread)
            l1_urls = re.findall(FileRetrievalPatterns.level1_v3, fread)
            l2_urls = re.findall(FileRetrievalPatterns.level2, fread)

        for urls, lbl, objtype in zip(
            [l0_urls, l1_urls, l2_urls],
            ["Level 0", "Level 1", "Level 2"],
            [L0Files, L1Files, L2Files],
        ):
            save_dir = Path(self.root, f"L{lbl[-1]}")
            # This contains all files with PDSDataFiles kwargs as the keys.
            constructor_dict = {"root": save_dir}
            for i in urls:
                base_kw = Path(i).suffix[1:].lower()
                cube_type = (
                    Path(i)
                    .stem[Path(i).stem.find("_", 19) + 1 :]  # noqa
                    .lower()
                )
                if cube_type in ("l0", "l1b", "l2"):
                    cube_type = ""
                else:
                    cube_type = cube_type + "_"
                kw = cube_type + base_kw
                constructor_dict[kw] = Path(save_dir, Path(i).name)
            setattr(self, f"l{lbl[-1]}", objtype(**constructor_dict))

            file_path_dict = {
                i: Path(save_dir, Path(i).name)
                for i in urls
                if not Path(save_dir, Path(i).name).is_file()
            }  # This contains only files that do not exist yet.

            if verbose:
                print(f"{len(file_path_dict)} {lbl} Files will be downloaded.")
            retrieve_urls(file_path_dict)

    def __str__(self):
        tree_string = f"{Path(self.root).name}\n"
        for k, v in vars(self).items():
            if k not in ("root"):
                tree_string += "\u2502   \u251c\u2500\u2500\u2500" f"{v}\n"
        return tree_string
