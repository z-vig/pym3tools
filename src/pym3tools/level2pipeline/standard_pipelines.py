# Standard Libraries
from pathlib import Path
from enum import Enum

# Dependencies
import yaml
from pydantic import ValidationError

# Top-Level Imports
from pym3tools.types import PathLike

# Relative Imports
from .standard_pipeline_runners import (
    BaseConfig,
    RegionalConfig,
    _run_regional,
    _run_regional_pds_terrain,
)


class InvalidConfigError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PipelineMode(Enum):
    REGIONAL = "regional"
    REGIONAL_PDSTERR = "regional_pds_terrain"
    GLOBAL = "global"


PIPELINES = {
    PipelineMode.REGIONAL: (RegionalConfig, _run_regional),
    PipelineMode.REGIONAL_PDSTERR: (RegionalConfig, _run_regional_pds_terrain),
}


def run_pipeline(config_file: PathLike):
    if not Path(config_file).suffix.__str__() in [".yaml", ".yml"]:
        raise InvalidConfigError("The config file is not a .yaml file.")

    cfg_dict = yaml.safe_load(open(Path(config_file)))
    cfg = BaseConfig(**cfg_dict)

    try:
        pipeline_mode = PipelineMode(cfg.pipeline_type)
    except ValueError:
        raise InvalidConfigError(
            f"Unknown pipeline type: {cfg.pipeline_type!r}"
            f"Valid Options: {[m.value for m in PipelineMode]}"
        )

    print(f"\n\n-----Running pipeline in mode: {pipeline_mode}-----")
    schema, runner = PIPELINES[pipeline_mode]

    try:
        params = schema.from_base(cfg)
    except ValidationError as e:
        raise InvalidConfigError(
            f"Invalid parameters for {pipeline_mode.value}:\n{e}"
        )

    runner(params)

    # return pipeline_mode
    # # pipeline_mode.run()
