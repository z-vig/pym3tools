# flake8: noqa
import numpy as np
from typing import Final

MOON_RADIUS: float = 1737400.0
MOON_GCS_PRJ: str = (
    'GEOGCRS["GCS_Moon_2000", DATUM["D_Moon_2000", ELLIPSOID["Moon_2000_IAU_IAG",1737400,0, LENGTHUNIT["metre",1]]], PRIMEM["Reference_Meridian",0, ANGLEUNIT["degree",0.0174532925199433]], CS[ellipsoidal,2], AXIS["geodetic latitude (Lat)",north, ORDER[1], ANGLEUNIT["degree",0.0174532925199433]], AXIS["geodetic longitude (Lon)",east, ORDER[2], ANGLEUNIT["degree",0.0174532925199433]], USAGE[ SCOPE["Not known."], AREA["World."], BBOX[-90,-180,90,180]], ID["ESRI",104903]]'
)
TIME_FMT: str = "%Y%m%dT%H%M%S"
DEG2RAD = np.pi / 180

h: Final[float] = 6.626 * 10**-34  # J*s, planck's constant
k_b: Final[float] = 1.381 * 10**-23  # J/K, boltzmann's constant
c: Final[float] = 2.998 * 10**8  # m/s, speed of lig
