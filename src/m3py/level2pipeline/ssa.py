# Dependencies
# import numpy as np

# Relative Imports
from .step import Step, PipelineState, StepCompletionState
from .utils.ssa_utils import IMSA


def AMSA():
    """ """


class ConvertToSSA(Step):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def run(self, state: PipelineState) -> PipelineState:
        ssa = IMSA(state.data, state.obs[:, :, 0], state.obs[:, :, 1])

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
