# Standard Libraries
from dataclasses import dataclass

# Dependencies
import numpy as np


class AngularUnitsError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class M3Geometry:
    """
    Stores M<sup>3</sup> Geometry data in a convenient dataclass.
    """
    solaz: np.ndarray
    solze: np.ndarray
    m3az: np.ndarray
    m3ze: np.ndarray
    phase: np.ndarray
    solen: np.ndarray
    m3len: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    cosi: np.ndarray
    radians: bool

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "M3Geometry":
        """
        Builds an `M3Geometry` instance from an observation backplane.
        """
        obs[obs == -999] = np.nan
        keys = list(cls.__dataclass_fields__.keys())
        values = {k: obs[:, :, i] for i, k in enumerate(keys[:-1])}
        return cls(**values, radians=False)

    def convert_to_rad(self):
        keys = list(self.__dataclass_fields__.keys())[:-2]
        deg_to_rad = np.pi/180
        [setattr(self, k, deg_to_rad * getattr(self, k)) for k in keys]
        self.radians = True


def calc_i(geo: M3Geometry) -> np.ndarray:
    if not geo.radians:
        raise AngularUnitsError(
            "Before calculating incidence angle, geo must be in Radians. Try"
            "running geo.convert_to_rad()"
        )
    arg = np.cos(geo.solze) * np.cos(geo.slope) +\
        np.sin(geo.solze) * np.sin(geo.slope) * np.cos(geo.solaz - geo.aspect)
    incidence_angle = (180/np.pi) * np.acos(arg)
    return incidence_angle


def calc_e(geo: M3Geometry) -> np.ndarray:
    if not geo.radians:
        raise AngularUnitsError(
            "Before calculating emission angle, geo must be in Radians. Try"
            "running geo.convert_to_rad()"
        )
    arg = np.cos(geo.m3ze) * np.cos(geo.slope) +\
        np.sin(geo.m3ze) * np.sin(geo.slope) * np.cos(geo.m3az - geo.aspect)
    emission_angle = (180/np.pi) * np.acos(arg)
    return emission_angle


def calc_g(geo: M3Geometry) -> np.ndarray:
    if not geo.radians:
        raise AngularUnitsError(
            "Before calculating phase angle, geo must be in Radians. Try"
            "running geo.convert_to_rad()"
        )
    arg = np.cos(geo.m3ze) * np.cos(geo.solze) +\
        np.sin(geo.m3ze) * np.sin(geo.solze) * np.cos(geo.solaz - geo.m3az)
    phase_angle = (180/np.pi) * np.acos(arg)
    return phase_angle
