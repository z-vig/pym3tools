# Standard Libraries
from importlib.resources import files

# Dependencies
from pydantic import BaseModel
import numpy as np
from rasterio import Affine  # type: ignore

prj_file = files("m3py.selenography.data").joinpath("cgs_moon_2000.prj")
with prj_file.open() as f:
    DEFAULT_CRS = f.read()


class AffineDict(BaseModel):
    """
    Little wrapper around Affine() object to use the affine coefficients like
    a dictionary.
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def to_affine(self) -> Affine:
        return Affine(
            a=self.a, b=self.b, c=self.c, d=self.d, e=self.e, f=self.f
        )


class GeorefData(BaseModel):
    row_offset: int
    col_offset: int
    height: int
    width: int
    geotransform: AffineDict
    crs: str
    nodata: int

    @classmethod
    def empty(cls) -> "GeorefData":
        return cls(
            row_offset=0,
            col_offset=0,
            height=0,
            width=0,
            geotransform=AffineDict(a=0, b=1, c=0, d=0, e=0, f=1),
            crs=DEFAULT_CRS,
            nodata=-999,
        )

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "GeorefData":
        return cls(
            row_offset=0,
            col_offset=0,
            height=arr.shape[0],
            width=arr.shape[1],
            geotransform=AffineDict(a=0, b=1, c=0, d=0, e=0, f=1),
            crs=DEFAULT_CRS,
            nodata=-999,
        )

    def to_list(self):
        return [self.row_offset, self.col_offset, self.height, self.width]
