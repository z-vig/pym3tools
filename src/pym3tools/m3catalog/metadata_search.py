from typing import overload
from pym3tools2.m3catalog import DataIDString
from pym3tools2.m3catalog import ColumnMetadata
from pym3tools2.m3catalog import M3DataID
from pym3tools2.m3catalog.l0_columns_names import L0ColumnName
from pym3tools2.m3catalog.l1b_columns_names import L1BColumnName
from pym3tools2.m3catalog.l2_columns_names import L2ColumnName


def _get_metadata(
    product_ids: list[str],
    target_product_id: DataIDString | list[DataIDString],
    target_column_entries: list[str],
) -> str | list[str] | None:
    if isinstance(target_product_id, list):
        result_list = []
        for targ_id in target_product_id:
            for i, j in zip(product_ids, target_column_entries):
                m3id = M3DataID.from_string(i)
                if targ_id == m3id.as_string:
                    result_list.append(str(j))
        return result_list
    else:
        for i, j in zip(product_ids, target_column_entries):
            m3id = M3DataID.from_string(i)
            if target_product_id == m3id.as_string:
                return j
        return None


@overload
def get_l0_metadata(
    data_id: DataIDString, column_name: L0ColumnName
) -> str: ...


@overload
def get_l0_metadata(
    data_id: list[DataIDString], column_name: L0ColumnName
) -> list[str]: ...


def get_l0_metadata(
    data_id: DataIDString | list[DataIDString], column_name: L0ColumnName
) -> str | list[str]:
    id_col1 = ColumnMetadata.from_l0_op1("PRODUCT_ID")
    id_col2 = ColumnMetadata.from_l0_op2("PRODUCT_ID")
    id_col_entries = [*id_col1.entries, *id_col2.entries]

    target_col1 = ColumnMetadata.from_l0_op1(column_name)
    target_col2 = ColumnMetadata.from_l0_op2(column_name)
    target_col_entries = [*target_col1.entries, *target_col2.entries]

    result = _get_metadata(id_col_entries, data_id, target_col_entries)
    if result is not None:
        return result
    raise ValueError("Invalid Data ID")


@overload
def get_l1_metadata(
    data_id: DataIDString, column_name: L1BColumnName
) -> str: ...


@overload
def get_l1_metadata(
    data_id: list[DataIDString], column_name: L1BColumnName
) -> list[str]: ...


def get_l1_metadata(
    data_id: DataIDString | list[DataIDString], column_name: L1BColumnName
) -> str | list[str]:
    id_col = ColumnMetadata.from_l1b("PRODUCT_ID").entries

    target_col = ColumnMetadata.from_l1b(column_name).entries

    result = _get_metadata(id_col.tolist(), data_id, target_col.tolist())
    if result is not None:
        return result
    raise ValueError("Invalid Data ID")


@overload
def get_l2_metadata(
    data_id: DataIDString, column_name: L2ColumnName
) -> str: ...


@overload
def get_l2_metadata(
    data_id: list[DataIDString], column_name: L2ColumnName
) -> list[str]: ...


def get_l2_metadata(
    data_id: DataIDString | list[DataIDString], column_name: L2ColumnName
) -> str | list[str]:
    id_col = ColumnMetadata.from_l2("PRODUCT_ID").entries

    target_col = ColumnMetadata.from_l2(column_name).entries

    result = _get_metadata(id_col.tolist(), data_id, target_col.tolist())
    if result is not None:
        return result
    raise ValueError("Invalid Data ID")
