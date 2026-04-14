from importlib.resources import files
from pathlib import Path

from pym3tools.file_downloader.l0index import L0Index
from pym3tools.file_downloader.create_urls_file import ColumnMetadata

l0_idx_op1 = files("pym3tools.file_downloader.data").joinpath(
    "L0_INDEX_OP1.TAB"
)
l0_idx_op2 = files("pym3tools.file_downloader.data").joinpath(
    "L0_INDEX_OP2.TAB"
)
l1B_idx = files("pym3tools.file_downloader.data").joinpath("L1B_INDEX.TAB")
l2_idx = files("pym3tools.file_downloader.data").joinpath("L2_INDEX.TAB")

prod_ID_locator = ColumnMetadata.from_index(L0Index.PRODUCT_ID)

with open(Path(str(l1B_idx))) as f:
    for i in f.readlines():
        id = prod_ID_locator.get_entry(i)
        print(id)
