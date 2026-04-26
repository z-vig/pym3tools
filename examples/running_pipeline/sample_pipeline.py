from pathlib import Path

import h5py
from cubio import write_envi

from pym3tools.rdn2rfl import pipeline_steps as r2r
from pym3tools.rdn2rfl import M3Level2Pipeline
from pym3tools.rdn2rfl import Step
from pym3tools.save_models import cache_to_cubio, PipelineCache
from pym3tools.save_models.pipeline_cache_schema import Dataset
from pym3tools.save_models.attribute_models import StandardDatasetAttrs

catalog = Path("D:/moon_data/m3/Gruithuisen_Region/M3G20090208T160125/")
gcps = "D:/moon_data/m3/Gruithuisen_Region/gcps_files/M3G20090208T160125.gcps"
cache = catalog / "pipeline_cache.hdf5"
slope = "D:/moon_data/derived_products/gruithuisen_region/Gruit_Slope.json"
aspect = "D:/moon_data/derived_products/gruithuisen_region/Gruit_Aspect.json"

print(f"Running Pipeline for ID: {catalog.name}")
steps: list[Step] = [
    r2r.Georeference(gcps_fp=gcps, custom_slope=slope, custom_aspect=aspect),
    r2r.SolarRemoval(),
    r2r.StatisticalPolish(),
    r2r.ThermalCorrection("Clark_Modified"),
    r2r.PhotometricCorrection("Lommel-Seeliger"),
]

pipeline = M3Level2Pipeline(
    steps,
    catalog,
    cache,
    overwrite_cache=True,
)

pipeline.run()

with h5py.File(str(cache)) as f:
    c = PipelineCache(f)
    save_dict: dict[str, Dataset[StandardDatasetAttrs]] = {
        "test_georef": c.georeferenced.cube,
        "test_obs": c.georeferenced.obs,
        "test_photo": c.photometric_corrected.photometry_backplane,
        "test_rfl": c.photometric_corrected.cube,
    }

    for k, v in save_dict.items():
        ctxt, cub = cache_to_cubio(v, k)
        write_envi(ctxt, cub, "BIL", "D:/moon_data/m3/Gruithuisen_Region/_")
