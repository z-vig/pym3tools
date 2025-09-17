# Dependencies
import numpy as np
import numpy.typing as npt
from scipy.special import legendre_p_all, legendre_p
from scipy.optimize import least_squares

E = 0.21  # +/- 0.02, from Warrell, 2004
C = 0.7  # +/- 0.1, from Warrell, 2004
DEG_TO_RAD = np.pi / 180


def gamma(w):
    """
    Utility quantity "gamma" from Hapke, 1984.

    Parameters
    ----------
    w: float
        Single Scattering Albedo
    """
    return np.sqrt(1 - w)


def r0(w):
    """
    Utility value for the linear approximation of H, defined in Hapke, 2002.

    Parameters
    ----------
    w: float
        Single Scattering Albedo
    """
    return (1 - gamma(w)) / (1 + gamma(w))


def H(w: np.ndarray, x: np.ndarray):
    """
    Improved Approximation of Chandresekhar's H-function.
    """
    return (
        1
        - w[:, None]
        * x[None, :]
        * (
            r0(w)[:, None]
            + ((1 - 2 * r0(w)[:, None] * x[None, :]) / 2)
            * np.log((1 + x) / x)[None, :]
        )
    ) ** -1


def A(n: int):
    """
    Legendre scattering function coefficients

    Parameters
    ----------
    n: int
        Order of legendre polynomial
    """
    if n % 2 == 0:
        return 0

    prefix = ((-1) ** ((n + 1) / 2)) / n
    series = np.prod(np.arange(1, n + 1, 2)) / np.prod(np.arange(2, n + 2, 2))
    return prefix * series


def p_hg(g: npt.NDArray[np.float32], max_n: int = 35):
    """
    Henyey_Greenstein phase function.

    Parameters
    ----------
    g: float or np.ndarray
        Phase angle in degrees.
    """
    even_terms = np.zeros((max_n, *g.shape))
    odd_terms = np.zeros((max_n, *g.shape))

    g_rads = (np.pi / 180) * g  # Converting to radians
    for i in range(1, max_n):
        if i % 2 == 0:
            b_n = (2 * i + 1) * (E) ** i
            even_terms[i, :] = (
                b_n * legendre_p(i, np.cos(g_rads))[0, :]  # type: ignore
            )
        else:
            b_n = C * (2 * i + 1) * (E) ** i
            odd_terms[i, :] = (
                b_n * legendre_p(i, np.cos(g_rads))[0, :]  # type: ignore
            )

    return 1 + np.sum(even_terms, axis=0) - np.sum(odd_terms, axis=0)


def P(x: np.ndarray, max_n: int = 35):
    """
    Hemispherical radiance that is scattered by a single Henyey-Greenstein
    particle. If the incidence angle is used, the lower hemisphere is
    calculated. If the emission angle is usedm the upper hemisphere is
    calculated. Coefficients of the Henyey-Greenstein function are given by
    Warrell, 2004.

    Parameters
    ----------
    x: np.ndarray
        Map of either incidence angle (for P(mu_0)) or emission angle (for
        P(mu)). An appropriate input for these quantities would be:
        P(np.cos(i)), for example.
    max_n: int, optional
        Maximum number order to use to estimate the legendre polynomial term.
    """
    lpolys = legendre_p_all(max_n, x)[0, 1:, :]  # Excluding n=0
    nvals = np.arange(1, max_n + 1, dtype=int)
    A_vals = np.empty(nvals.size)
    for n in nvals:
        A_vals[n - 1] = A(n)

    coefs = A_vals * (C * (2 * nvals + 1) * E**nvals)
    return 1 + np.sum(coefs[:, None] * lpolys, axis=0)


def Rho(max_n: int = 35):
    """
    Average radiance scattered back into the lower hemisphere while being
    uniformly illuminated by the lower hemishpere.

    Parameters
    ----------
    max_n: int, optional
        Maximum number order to use to estimate the legendre polynomial term.
    """
    nvals = np.arange(1, max_n + 1, dtype=int)
    A_vals = np.empty(nvals.size)
    for n in nvals:
        A_vals[n - 1] = A(n)
    return 1 - np.sum((A_vals**2) * (C * (2 * nvals + 1) * E**nvals))


def M(w, i, e):
    """
    Approximated multiple scattering term from Hapke, 2002.

    Parameters
    ----------
    w: np.ndarray
        Single scattering albedo
    i: np.ndarray
        Incidence angle in degrees
    e: np.ndarray
        Emission angle in degrees
    """
    mu = np.cos(DEG_TO_RAD * e)
    mu0 = np.cos(DEG_TO_RAD * i)
    return (
        (P(mu0)[None, :] * (H(w, mu) - 1))
        + (P(mu)[None, :] * (H(w, mu0) - 1))
        + (Rho() * (H(w, mu) - 1) * (H(w, mu0) - 1))
    )


def IMSA_r(w, i, e):
    """
    Returns IMSA Relative Reflectance
    """
    mu0 = np.cos(DEG_TO_RAD * i)[None, :]
    mu = np.cos(DEG_TO_RAD * e)[None, :]
    return (1 - gamma(w)[:, None] ** 2) / (
        (1 + 2 * gamma(w)[:, None] * mu0) * (1 + 2 * gamma(w)[:, None] * mu)
    )


def IMSA(data: np.ndarray, i: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Converts a relative reflectance (i.e. the reflectance of a surface with
    respect to a standard geometry) to a single scattering albedo using the
    isotropic multiple scattering approximation (IMSA) of Hapke, 1993.

    Parameters
    ----------
    data: np.ndarray
        Spectral data cube.
    i, e: np.ndarray
        The incidence and emission backplane images, respectively, in units of
        degrees.

    Returns
    -------
    ssa: np.ndarray
        Conversion of data to SSA.

    Notes
    -----
    The assumptions for this calculation are as follows:

    - Spectra are already normalized to a standard geometry.
    - Phase angle is outside the opposition peak
    - Scattering coefficients (Phase function) of the material is isotropic
    - Thermal emission is negligible or removed
    - Uses the two-stream approximation of Chandresekhar's H-function

    The equation used can be found in Hapke, 1993, pg. 291, eq. 11.6.
    """
    mu_0 = np.cos(i[:, :, None] * DEG_TO_RAD)
    mu = np.cos(e[:, :, None] * DEG_TO_RAD)
    gamma = (
        np.sqrt(
            data**2 * (mu_0 + mu) ** 2
            + (1 + 4 * mu_0 * mu * data) * (1 - data)
        )
        - (mu_0 + mu) * data
    ) / (1 + 4 * mu_0 * mu * data)

    return 1 - gamma**2


def B_sh(g):
    """
    Shadow Hiding Opposition Effect (SHOE). The B0 and h values are set from
    Warrell, 2004.
    """
    B0 = 3.1  # +/- 0.5 from Warrell, 2004
    h = 0.11  # +/- 0.03 from Warrell, 2004
    return B0 / (1 + ((1 / h) * np.tan(DEG_TO_RAD * g / 2)))


def AMSA(
    w: np.ndarray,
    i: np.ndarray,
    e: np.ndarray,
    g: np.ndarray,
    include_SHOE: bool = True,
) -> np.ndarray:
    """
    Full AMSA Model from Hapke, 2002

    Parameters
    ----------
    w: float
        single scattering albedo
    i: float
        incidence angle in degrees
    e: float
        emission angle in degrees
    g: float
        phase angle in degrees
    """
    mu0 = np.cos(DEG_TO_RAD * i)
    mu = np.cos(DEG_TO_RAD * e)
    if include_SHOE:
        return (
            (w[:, None] / (4 * np.pi))
            * (mu0 / (mu0 + mu))[None, :]
            * ((p_hg(g) + B_sh(g))[None, :] + M(w, mu0, mu))
        )
    else:
        return (
            (w[:, None] / (4 * np.pi))
            * (mu0 / (mu0 + mu))[None, :]
            * (p_hg(g)[None, :] + M(w, mu0, mu))
        )


def fit_to_AMSA(r: np.ndarray, i: float, e: float, g: float):
    """
    Fits a reflectance spectrum to Hapke's Anisotropic Multiple Scattering
    Approximation model in a least squares sense.

    Parameters
    ----------
    r: np.ndarray
        Reflectance Spectrum.

    Returns
    -------
    w: np.ndarray
        Fitted single scattering albedo values.
    """

    def residuals(w):
        return (
            r
            - AMSA(
                w,
                i * np.ones(w.size),
                e * np.ones(w.size),
                g * np.ones(w.size),
            )[
                :, 0
            ]  # second axis is just repeating i, e, g, so take the first index
        )

    result = least_squares(  # type: ignore
        residuals,
        np.mean(r) * np.ones(r.size),
    )

    return result.x
