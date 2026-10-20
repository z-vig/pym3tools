`pym3tools`
---

  >*A modern standardized toolkit for processing Moon Mineralogy Mapper (M<sup>3</sup>) Data in a pure Python 3.13 environment.*

---
[![Unit Tests](https://github.com/z-vig/pym3tools/actions/workflows/unit_testing.yml/badge.svg)](https://github.com/z-vig/pym3tools/actions/workflows/unit_testing.yml)

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

`pym3tools` is currently only available through PyPI.
```bash
pip install pym3tools
```

### File Retrieval from PDS
```python
import pym3tools as m3

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
    if Path(root_dir, i).exists():
        _ = m3.M3FileManager(root_dir, i)  # Downloads Data
    else:
        m3.file_downloader.create_urls_file(root_dir, i)  # Creates url files
```

### Building a Processing Pipeline
```python
import pym3tools as m3
from pym3tools.level2pipeline import (
    Crop, Georeferemce, TerrainModel, SolarSpectrumRemoval, StatisticalPolish,
    Photometric Correction
)
from rasterio import BoundingBox

# Initializing File Manager
root_dir = "./cbvc_data"
file_manager = m3.M3FileManager(root_dir, "M3G20090531T215442")

# Defining Regional Scope
bbox = BoundingBox(left=87, bottom=48, right=114, top=71)

# Listing steps of the processing pipeline
steps = [
    Crop("cropped", bbox=bbox, save_output=True),
    Georeference("georeferenced", save_output=True),
    TerrainModel("terrain_model", save_output=True),
    SolarSpectrumRemoval("solar_spec_removed", save_output=True),
    StatisticalPolish("statistical_polish", save_output=True),
    ClarkThermalCorrection("thermal_correction", use_pds_temperatures=True, save_output=True),
    PhotometricCorrection("photometric_correction", save_output=True),
]

# Initializing pipeline
pipeline = m3.M3Level2Pipeline(steps, file_manager)

# Running pipeline
pipeline.run()
```
