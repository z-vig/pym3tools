from dataclasses import dataclass
from typing import Self, Union, TypedDict
from importlib.resources import files
from importlib.resources.abc import Traversable
import re

import numpy as np

from .l0_columns_names import L0ColumnName
from .l1b_columns_names import L1BColumnName
from .l2_columns_names import L2ColumnName


class ColumnMetadataConstructor(TypedDict):
    index: int
    name: str
    data_type: str
    start_byte: int
    nbytes: int
    format: str
    description: str


def _parse_index_label(
    idx_file: Traversable,
    column_name: Union[L0ColumnName, L1BColumnName, L2ColumnName],
) -> ColumnMetadataConstructor:
    column_entry_pattern = re.compile(
        r"COLUMN_NUMBER\s+=\s(\d+)\s+\n"
        rf"\s+NAME\s+=\s\"?{column_name}\"?\s+\n"
        r"\s+DATA_TYPE\s+=\s\"?(\w+)\"?\s+\n"
        r"\s+START_BYTE\s+=\s+(\d+)\s+\n"
        r"\s+BYTES\s+=\s+(\d+)\s+\n"
        r"(?:\s+FORMAT\s+=\s\"?([A-Za-z0-9.]+)\"?\s+\n)?"
        r"\s+DESCRIPTION\s+=\s\"([^\"]+)\"\s+END_OBJECT"
    )

    with open(str(idx_file)) as f:
        result = re.search(column_entry_pattern, f.read())

    if result is None:
        raise ValueError("Column Name not valid.")
    grps = result.groups()

    constr: ColumnMetadataConstructor = {
        "index": int(grps[0]),
        "name": column_name,
        "data_type": grps[1],
        "start_byte": int(grps[2]) - 1,
        "nbytes": int(grps[3]),
        "format": grps[4],
        "description": re.sub(" +", " ", grps[5]).replace("\n ", ""),
    }

    return constr


@dataclass
class ColumnMetadata:
    """
    Class for storing metadata for a column in the M3 index files.


    Returns
    -------
    index: int
        The column number in the index file.
    name: str
        The name of the column.
    data_type: str
        The data type of the column.
    start_byte: int
        The starting byte of the column in the index file.
    nbytes: int
        The number of bytes of the column in the index file.
    format: str
        The format of the column in the index file.
    description: str
        The description of the column in the index file.
    tab_file: Traversable
        The path to the index file containing the column.
    """

    index: int
    name: str
    data_type: str
    start_byte: int
    nbytes: int
    format: str
    description: str
    tab_file: Traversable

    @classmethod
    def _from_index(
        cls,
        col_name: Union[L0ColumnName, L1BColumnName, L2ColumnName],
        index_name: str,
    ) -> Self:
        """Create ColumnMetadata from an index file."""
        idx_data_dir = files("pym3tools2.m3catalog.index_data")
        idx_lbl = idx_data_dir.joinpath(f"{index_name}.LBL")
        idx_tab = idx_data_dir.joinpath(f"{index_name}.TAB")
        return cls(**_parse_index_label(idx_lbl, col_name), tab_file=idx_tab)

    @classmethod
    def from_l0_op1(cls, col_name: L0ColumnName) -> Self:
        """
        Creates a metadata object for a column in the L0 OP1 index file.

        Parameters
        ----------
        col_name: L0ColumnName
            The name of the column to create the metadata for.
        """
        return cls._from_index(col_name, "L0_INDEX_OP1")

    @classmethod
    def from_l0_op2(cls, col_name: L0ColumnName) -> Self:
        """
        Creates a metadata object for a column in the L0 OP2 index file.

        Parameters
        ----------
        col_name: L0ColumnName
            The name of the column to create the metadata for.
        """
        return cls._from_index(col_name, "L0_INDEX_OP2")

    @classmethod
    def from_l1b(cls, col_name: L1BColumnName) -> Self:
        """
        Creates a metadata object for a column in the L1B index file.

        Parameters
        ----------
        col_name: L0ColumnName
            The name of the column to create the metadata for.
        """
        return cls._from_index(col_name, "L1B_INDEX")

    @classmethod
    def from_l2(cls, col_name: L2ColumnName) -> Self:
        """
        Creates a metadata object for a column in the L2 index file.

        Parameters
        ----------
        col_name: L0ColumnName
            The name of the column to create the metadata for.
        """
        return cls._from_index(col_name, "L2_INDEX")

    @property
    def entries(self) -> np.ndarray:
        all_entries = []
        with open(str(self.tab_file)) as f:
            for index_line in f.readlines():
                col_slice = slice(
                    self.start_byte, self.start_byte + self.nbytes
                )
                col_entry = index_line[col_slice].replace(" ", "")
                all_entries.append(col_entry)
        return np.array(all_entries)
