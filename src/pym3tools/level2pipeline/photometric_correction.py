# Dependencies
import h5py as h5  # type: ignore

# Relative Imports
from .step import Step, PipelineState, StepCompletionState
from .utils.photometric_correction_utils import (
    compute_limb_darkening,
    compute_f_alpha,
)
from .utils.data_fetching_utils import get_phase_function_rgi


class PhotometricCorrection(Step):
    def run(self, state: PipelineState) -> PipelineState:
        ldf, ldf_norm = compute_limb_darkening(state.obs)
        rgi = get_phase_function_rgi(self.manager)
        f_alpha, f_alpha_norm = compute_f_alpha(
            state.obs[:, :, 2], rgi, state.wvl.size
        )

        self.photometric_coefficients = (ldf_norm / ldf)[:, :, None] * (
            f_alpha_norm / f_alpha
        )

        new_flags = state.flags
        new_flags.photometrically_corrected = StepCompletionState.Complete
        new_state = PipelineState(
            data=self.photometric_coefficients * state.data,
            wvl=state.wvl,
            obs=state.obs,
            georef=state.georef,
            flags=new_flags,
        )
        return new_state

    def save(self, output: PipelineState) -> None:
        super().save(output)
        with h5.File(self.manager.cache, "r+") as f:
            g = f[self.name]
            assert isinstance(g, h5.Group)
            g.create_dataset(
                "photometric_coefficients",
                data=self.photometric_coefficients,
                dtype="f4",
            )
