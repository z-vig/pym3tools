`pym3tools`
---

  >*A modern standardized toolkit for processing Moon Mineralogy Mapper (M<sup>3</sup>) Data in a pure Python 3.13 environment.*

---
[![Unit Tests](https://github.com/z-vig/pym3tools/actions/workflows/unit_testing.yml/badge.svg)](https://github.com/z-vig/pym3tools/actions/workflows/unit_testing.yml) [![DOI](https://zenodo.org/badge/1028534300.svg)](https://doi.org/10.5281/zenodo.23510051)
## What is `pym3tools`?

`pym3tools` is a modular data pipeline designed to make fetching, processing and analyzing M<sup>3</sup> a breeze in Python. `pym3tools` incorporates many different processing methods from current planetary science literature, which allows it to serve as a state-of-the-art, standardized processing method, which will help improve the repeatability of geologic results obtained via M<sup>3</sup>.

---

## Available Modules

| Module| Description |
|-------------|--------------|
|`data_retrieval`| Provides utilities for **downloading** M<sup>3</sup> data and placing it into user-friendly data structures|
|`m3catalog`| Lays the framework for data structures that are compatible with **M<sup>3</sup> metadata**.|
|`geo_ops`| Geographic operations such as **georeferencing**. |
|`rfl2rfl`| Contains the main logic and computational functions for processing level 1 (**Radiance**) M<sup>3</sup> data **into** level 2 (**Reflectance**) data.|
|`save_models`| Provides the framework for saving steps of the `rdn2rfl` pipeline to easy-to-use HDF5 files.|
---

## Quick Start

### Installation

`pym3tools` is currently only available through PyPI (pip).
```bash
pip install pym3tools
```

### File Retrieval from PDS
```python
import pym3tools as m3
from pym3tools.rdn2rfl import pipeline_steps as r2r

# Data covering the Compton Belk'ovich Volcanic Complex
m3_data_ids = [
    "M3G20090531T215442",
    "M3G20090601T021112",
    "M3G20090601T104211",
    "M3G20090601T145212",
    "M3G20090627T201242",
]

# Retrieving Data
root_dir = "./cbvc_data"
for i in m3_data_ids:
    m3.data_retrieval.download_m3_dataset(i)  # Creates data directories
```

### Building a Processing Pipeline
```python
import pym3tools as m3
from pym3tools.level2pipeline import (
    Crop, Georeferemce, TerrainModel, SolarSpectrumRemoval, StatisticalPolish,
    Photometric Correction
)

# Listing steps of the processing pipeline
steps = [
    r2r.Georeference(
        enabled=True,
        gcps_fp="./path_to/ground_control_points.gcps",
        custom_slope="./path_to/slope_data.bil",
        custom_aspect="./path_to/aspect_data.bil"
    ),
    r2r.SolarRemoval(enabled=True),
    r2r.StatisticalPolish(enabled=True),
    r2r.ThermalCorrection(enabled=True, method="Clark_Modified"),
    r2r.PhotometricCorrection(enabled=True, method="Lommel-Seeliger"),
]

# Initializing pipeline
pipeline = m3.rdn2rfl.M3Level2Pipeline(
    steps, "path_to/data_directory", "path_to/save_file.hdf5"
)

# Running pipeline
pipeline.run()
```
