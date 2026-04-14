from pathlib import Path

from pym3tools.m3catalog.column_metadata import ColumnMetadata
from pym3tools.m3catalog.m3_data_id import M3DataID
from pym3tools.types import OpticalPeriod

l0_op1 = (
    Path(__file__).parent
    / "src"
    / "pym3tools"
    / "m3catalog"
    / "index_data"
    / "L0_INDEX_OP1.TAB"
)

l0_op2 = (
    Path(__file__).parent
    / "src"
    / "pym3tools"
    / "m3catalog"
    / "index_data"
    / "L0_INDEX_OP2.TAB"
)

l1b = (
    Path(__file__).parent
    / "src"
    / "pym3tools"
    / "m3catalog"
    / "index_data"
    / "L1B_INDEX.TAB"
)

l2 = (
    Path(__file__).parent
    / "src"
    / "pym3tools"
    / "m3catalog"
    / "index_data"
    / "L2_INDEX.TAB"
)

l0ids_op1 = ColumnMetadata.from_l0_op1("PRODUCT_ID")
l0ids_op2 = ColumnMetadata.from_l0_op2("PRODUCT_ID")
l1ids = ColumnMetadata.from_l1b("PRODUCT_ID")
l2ids = ColumnMetadata.from_l2("PRODUCT_ID")

all_m3_products_str = [
    *l0ids_op1.entries,
    *l0ids_op2.entries,
    *l1ids.entries,
    *l2ids.entries,
]

ALL_M3_PRODUCTS = [M3DataID.from_string(i) for i in all_m3_products_str]


def list_by_op(op: OpticalPeriod):
    filler = "="
    main_string = f"{op} Products"
    print(f"{main_string:{filler}^18}")
    for m3id in ALL_M3_PRODUCTS:
        if m3id.op == op:
            print(m3id.as_string, m3id.image_type)
