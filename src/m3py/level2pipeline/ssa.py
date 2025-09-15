# Dependencies
import numpy as np

# Relative Imports
from .step import Step, PipelineState, StepCompletionState


def isotropic_multiple_scattering_approximation(
    data: np.ndarray, i: np.ndarray, e: np.ndarray
) -> np.ndarray:
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
    deg_to_rad = np.pi / 180
    mu_0 = np.cos(i[:, :, None] * deg_to_rad)
    mu = np.cos(e[:, :, None] * deg_to_rad)
    gamma = (
        np.sqrt(
            data**2 * (mu_0 + mu) ** 2
            + (1 + 4 * mu_0 * mu * data) * (1 - data)
        )
        - (mu_0 + mu) * data
    ) / (1 + 4 * mu_0 * mu * data)

    return 1 - gamma**2


class ConvertToSSA(Step):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def run(self, state: PipelineState) -> PipelineState:
        ssa = isotropic_multiple_scattering_approximation(
            state.data, state.obs[:, :, 0], state.obs[:, :, 1]
        )

        new_flags = state.flags
        new_flags.converted_to_ssa = StepCompletionState.Complete

        new_state = PipelineState(
            data=ssa,
            wvl=state.wvl,
            obs=state.obs,
            georef=state.georef,
            flags=new_flags,
        )

        return new_state
