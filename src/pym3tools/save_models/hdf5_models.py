from dataclasses import dataclass, field
from typing import Self

import h5py
import numpy as np

from pym3tools2.types import DatasetStatus


def _get_and_validate_h5_dataset(
    group: h5py.Group, dset_name: str
) -> tuple[h5py.Dataset, DatasetStatus]:
    dset_exists = group.get(dset_name)
    status: DatasetStatus
    if dset_exists is None:
        dset = group.create_dataset(dset_name, (0, 0), dtype=np.float32)
        status = "NotSet"
    else:
        _dset = group[dset_name]
        if not isinstance(_dset, h5py.Dataset):
            raise ValueError(
                f"Invalid dataset ({dset_name}) in group: {group.name}"
            )
        dset = _dset
        status = "Set"
    return dset, status


def _get_and_validate_h5group(
    base: h5py.Group | h5py.File, name: str
) -> h5py.Group:
    grp = base[name]
    if not isinstance(grp, h5py.Group):
        raise ValueError(f"Invalid group ('{name}') in file: {base.name}")
    return grp


@dataclass(frozen=True)
class GroupSpec:
    name: str
    groups: dict[str, Self] = field(default_factory=dict)

    def initialize(self, parent: h5py.File | h5py.Group):
        new_parent = parent.create_group(self.name)
        for g in self.groups.values():
            g.initialize(new_parent)
