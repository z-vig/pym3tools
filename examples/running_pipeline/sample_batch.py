from pathlib import Path

from pydantic import BaseModel

from pym3tools.rdn2rfl import pipeline_steps as r2r
from pym3tools.rdn2rfl import M3Level2Pipeline
from pym3tools.rdn2rfl import Step


class RegionalPipelineConfig(BaseModel):
    name: str
    gcps: Path
    slope: Path
    aspect: Path
    catalog: Path
    cache: Path


class RegionalPipelineBatch(BaseModel):
    batch_name: str
    config_list: list[RegionalPipelineConfig]


with open(Path(__file__).parent / "batch_config.json") as f:
    batch = RegionalPipelineBatch.model_validate_json(f.read())

for config in batch.config_list:
    print(f"Running Pipeline for ID: {config.name}")
    steps: list[Step] = [
        r2r.Georeference(
            gcps_fp=config.gcps,
            custom_slope=config.slope,
            custom_aspect=config.aspect,
        ),
        r2r.SolarRemoval(),
        r2r.StatisticalPolish(),
        r2r.ThermalCorrection("Clark_Modified"),
        r2r.PhotometricCorrection("Lommel-Seeliger"),
    ]

    pipeline = M3Level2Pipeline(
        steps,
        config.catalog,
        config.cache,
        overwrite_cache=True,
    )

    pipeline.run()
