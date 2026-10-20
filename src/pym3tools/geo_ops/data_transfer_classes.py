from dataclasses import dataclass

import numpy as np
from cubio.geotools.models import BoundingBoxModel, GeotransformModel


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
