# flake8: noqa
import numpy as np
from pym3tools.m3catalog.list_m3_products import ALL_M3_PRODUCTS

l0_product_ids = np.zeros(0, dtype="|S18")
l1_product_ids = np.zeros(0, dtype="|S18")
l2_product_ids = np.zeros(0, dtype="|S18")
for i in ALL_M3_PRODUCTS:
    if i.image_type == "L0":
        l0_product_ids = np.append(l0_product_ids, i.as_string)
    if i.image_type == "RDN":
        l1_product_ids = np.append(l1_product_ids, i.as_string)
    if i.image_type == "RFL":
        l2_product_ids = np.append(l2_product_ids, i.as_string)

with open("l0_data_ids.py", "w") as f:
    f.write("from typing import Literal, TypeAlias\n\n\n")
    f.write("L0DataIDString: TypeAlias = Literal[\n")
    for i in l0_product_ids:
        f.write(f'    "{i}",\n')
    f.write("]\n\n")
    f.write("l0_data_id_strings: list[L0DataIDString] = [\n")
    for i in l0_product_ids:
        f.write(f'    "{i}",\n')
    f.write("]")
    f.write("\n")

with open("data_ids.py", "w") as f:
    f.write("from typing import Literal, TypeAlias, TypeGuard\n\n\n")
    f.write("DataIDString: TypeAlias = Literal[\n")
    for i in l1_product_ids:
        f.write(f'    "{i}",\n')
    f.write("]\n\n")
    f.write("data_id_strings: list[DataIDString] = [\n")
    for i in l1_product_ids:
        f.write(f'    "{i}",\n')
    f.write("]\n\n\n")
    f.write("def is_valid_data_id(value: str) -> TypeGuard[DataIDString]:\n")
    f.write("    return value in data_id_strings")
    f.write("\n")
