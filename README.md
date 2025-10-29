# 🛰️ **pym3tools**

  >*A modern standardized toolkit for processing Moon Mineralogy Mapper (M<sup>3</sup>) Data in Python.*

---

## What is `pym3tools`?

`pym3tools` is a modular data pipeline designed to make fetching, processing and analyzing M<sup>3</sup> a breeze in Python. `pym3tools` incorporates many different processing methods from current planetary science literature, which allows it to serve as a state-of-the-art, standardized processing method, which will help improve the repeatability of geologic results obtained via M<sup>3</sup>.

---

## Available Modules

| Module| Description |
|-------------|--------------|
|ℹ️`io`| Provides easy-to-use and flexible input/output functions for **reading and writing** M<sup>3</sup> data |
|📬`PDSretrieval`    | Allows the user to **download files** directly from NASA's planetary data system (PDS) into a user-friendly format in the form of a file manager class. |
|⚙️`level2pipeline`   | Contains the main logic and computional functions for processing level 1v3 (**Radiance**) M<sup>3</sup> data **into** level 2 (**Reflectance**) data|
|🌗 `selenography`     | Wraps `gdal` and `rasterio` to provide **geospatial utilities tailored to use on the Moon**, including built-in lunar coordinate systems |
