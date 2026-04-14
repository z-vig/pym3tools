from dataclasses import dataclass
from typing import Generic, TypeVar, Mapping
import h5py
import numpy as np

from pym3tools2.types import DatasetStatus
from pym3tools2.save_models import attribute_models as attr_models
from .hdf5_models import (
    _get_and_validate_h5_dataset,
    _get_and_validate_h5group,
)
from .hdf5_models import GroupSpec


PIPELINE_SCHEMA = GroupSpec(
    name="pipeline_data",
    groups={
        "cropped": GroupSpec(name="cropped"),
        "georeferenced": GroupSpec(name="georeferenced"),
        "solar_removed": GroupSpec(name="solar_removed"),
        "stat_polished": GroupSpec(name="stat_polished"),
        "thermal_corrected": GroupSpec(name="thermal_corrected"),
        "photometric_corrected": GroupSpec(name="photometric_corrected"),
    },
)


A = TypeVar("A", bound=Mapping[str, object])


@dataclass
class Dataset(Generic[A]):
    """
    HDF5 Dataset representation for `pym3tools` pipeline cache.
    """

    name: str
    _dset: h5py.Dataset
    _status: DatasetStatus

    def __post_init__(self) -> None:
        self._parent: h5py.Group = self._dset.parent  # type: ignore

    def read(self) -> np.ndarray:
        data = self._dset[()]
        if self._status == "NotSet":
            raise FileNotFoundError(f"{self.name} does not exist.")
        return data

    def write(self, data: np.ndarray) -> None:
        del self._parent[self.name]
        self._dset = self._parent.create_dataset(
            self.name, data=data, dtype=data.dtype
        )

    @property
    def attrs(self) -> A:
        return self._dset.attrs  # type: ignore

    def set_attrs(self, attrs: A):
        if self._status == "NotSet":
            raise ValueError("Write data before setting attributes.")
        for k, v in attrs.items():
            self._dset.attrs[k] = v


@dataclass
class BaseGroup(Generic[A]):
    _group: h5py.Group

    @property
    def cube(self) -> Dataset[attr_models.StandardDatasetAttrs]:
        dset, status = _get_and_validate_h5_dataset(self._group, "cube")
        return Dataset("cube", dset, status)

    @property
    def attrs(self) -> A:
        return self._group.attrs  # type: ignore

    def set_attrs(self, attrs: A) -> None:
        for k, v in attrs.items():
            self._group.attrs[k] = v


@dataclass
class CroppedGroup(BaseGroup[attr_models.CroppedAttrs]):
    @property
    def latlong(self) -> Dataset[attr_models.StandardDatasetAttrs]:
        latlong, status = _get_and_validate_h5_dataset(self._group, "latlong")
        return Dataset("latlong", latlong, status)


@dataclass
class GeoreferencedGroup(BaseGroup[attr_models.GeoreferencedAttrs]):
    @property
    def latlong(self) -> Dataset[attr_models.StandardDatasetAttrs]:
        latlong, status = _get_and_validate_h5_dataset(self._group, "latlong")
        return Dataset("latlong", latlong, status)

    @property
    def obs(self) -> Dataset[attr_models.StandardDatasetAttrs]:
        obs, status = _get_and_validate_h5_dataset(self._group, "obs")
        return Dataset("obs", obs, status)


@dataclass
class SolarRemovedGroup(BaseGroup[attr_models.SolarRemovedAttrs]):
    @property
    def solarspectrum(self) -> Dataset[attr_models.OneDimensionDatasetAttrs]:
        solspec, status = _get_and_validate_h5_dataset(
            self._group, "solarspectrum"
        )
        return Dataset("solarspectrum", solspec, status)


@dataclass
class StatPolishedGroup(BaseGroup[attr_models.StatPolishedAttrs]):
    @property
    def statpol_coefficients(
        self,
    ) -> Dataset[attr_models.OneDimensionDatasetAttrs]:
        statpol, status = _get_and_validate_h5_dataset(
            self._group, "statpol_coefficients"
        )
        return Dataset("statpol_coefficients", statpol, status)


@dataclass
class ThermalCorrectedGroup(BaseGroup[attr_models.ThermalCorrectedAttrs]):
    @property
    def temperature_map(self) -> Dataset[attr_models.StandardDatasetAttrs]:
        temp, status = _get_and_validate_h5_dataset(
            self._group, "temperature_map"
        )
        return Dataset("temperature_map", temp, status)


@dataclass
class PhotometricCorrectedGroup(
    BaseGroup[attr_models.PhotometricCorrectedAttrs]
):
    @property
    def photometric_coefficients(
        self,
    ) -> Dataset[attr_models.StandardDatasetAttrs]:
        coefs, status = _get_and_validate_h5_dataset(
            self._group, "photometric_coefficients"
        )
        return Dataset("photometric_coefficients", coefs, status)

    @property
    def photometry_backplane(
        self,
    ) -> Dataset[attr_models.StandardDatasetAttrs]:
        photom, status = _get_and_validate_h5_dataset(
            self._group, "photometry_backplane"
        )
        return Dataset("photometry_backplane", photom, status)


@dataclass
class PipelineCache:
    _file: h5py.File

    def __post_init__(self) -> None:
        _base = self._file["pipeline_data"]
        if not isinstance(_base, h5py.Group):
            raise ValueError("Invalid Base Group")
        self._base: h5py.Group = _base

    @property
    def cropped(self) -> CroppedGroup:
        grp = _get_and_validate_h5group(self._base, "cropped")
        return CroppedGroup(grp)

    @property
    def georeferenced(self) -> GeoreferencedGroup:
        grp = _get_and_validate_h5group(self._base, "georeferenced")
        return GeoreferencedGroup(grp)

    @property
    def solar_removed(self) -> SolarRemovedGroup:
        grp = _get_and_validate_h5group(self._base, "solar_removed")
        return SolarRemovedGroup(grp)

    @property
    def stat_polished(self) -> StatPolishedGroup:
        grp = _get_and_validate_h5group(self._base, "stat_polished")
        return StatPolishedGroup(grp)

    @property
    def thermal_corrected(self) -> ThermalCorrectedGroup:
        grp = _get_and_validate_h5group(self._base, "thermal_corrected")
        return ThermalCorrectedGroup(grp)

    @property
    def photometric_corrected(self) -> PhotometricCorrectedGroup:
        grp = _get_and_validate_h5group(self._base, "photometric_corrected")
        return PhotometricCorrectedGroup(grp)
