from dataclasses import dataclass

import numpy as np
from cubio.geotools.models import GeotransformModel, BoundingBoxModel


@dataclass
class ResampledList:
    data: list[np.ndarray]
    gtrans: GeotransformModel
    bounds: BoundingBoxModel
    crs: str


@dataclass
class ResampledMosaic:
    data: np.ndarray
    gtrans: GeotransformModel
    bounds: BoundingBoxModel
    crs: str
