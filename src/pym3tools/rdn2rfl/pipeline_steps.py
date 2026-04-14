from pathlib import Path
from typing import Literal

from cubio.geotools.models import BoundingBoxModel, GCPGroup
import numpy as np

from pym3tools2.constants import MOON_GCS_PRJ
from pym3tools2.save_models import attribute_models as attrs
from pym3tools2.data_retrieval.get_ground_truth_file import determine_warm_cold
from pym3tools2.types import ThermalCorrectionMethod, TopoCorrectionMethod

from .pipeline_services import (
    crop,
    georeference,
    solar_removal,
    statistical_polish,
    thermal_correction,
    photometric_correction,
)
from .step_model import Step
from .pipeline_state import PipelineState


class Crop(Step):
    """
    Croping Step
    ------------
    Cropping step for the M<sup>3</sup> L1 to L2 pipeline. This step crops raw
    M3 image strips to a particular bounding box. This is distinct from the
    cropping that may occur in the georeferencing step due to a limited extent
    of Ground Control Points.

    Parameters
    ----------
    bounding_box : BoundingBoxModel
        Bounding box for cropping the data.
    name : str, optional
        Name of the pipeline step. Default is "cropping".
    crs: str, optional
        Coordinate reference system of the bounding box. Automatically set to
        GCS Moon 2000.
    prj4_str : str, optional
        PROJ4 string for the target projection. Default is MOON_GCS_PRJ.
    enabled : bool, optional
        Whether the step is enabled.
    """

    def __init__(
        self,
        bounding_box: BoundingBoxModel,
        name: str = "cropped",
        crs: str = MOON_GCS_PRJ,
        enabled: bool = True,
        save_output: bool = True,
    ) -> None:
        super().__init__(name, enabled, save_output)
        self.bbox = bounding_box
        self.crs = crs

    def run(self, state: PipelineState) -> PipelineState:
        crop_res = crop.crop_data(state, self.bbox, self.catalog)
        self.cropped_loc = np.stack(
            [crop_res.longitudes, crop_res.latitudes], axis=-1
        )
        cropped_area = crop.get_cropped_area(state, self.crs, self.bbox)
        return crop.modify_state(state, crop_res.cropped_data, cropped_area)

    def save(self, output: PipelineState) -> None:
        crop.write_to_cache(
            self.cache, self._time(), output, self.cropped_loc, self.bbox
        )


class Georeference(Step):
    """
    Georeferencing Step
    ---
    Georeference step for the M<sup>3</sup> L1 to L2 pipeline. This step
    handles the georeferencing of the input data based on either GCPs or the
    LOC backplane. It also calculates a reusable pyresample geometry that can
    be used for subsequent steps in the pipeline.

    Parameters
    ----------
    bounding_box : BoundingBoxModel or Literal["auto"]
        Bounding box for cropping the data. If "auto", the bounding box will be
        calculated from the GCPs.
    gcps_fp : Path or str, optional
        File path to the GCPs file. Required if bounding_box is "auto".
    name : str
        Name of the pipeline step.
    prj4_str : str, optional
        PROJ4 string for the target projection. Default is MOON_GCS_PRJ.
    enabled : bool, optional
        Whether the step is enabled.
    """

    def __init__(
        self,
        gcps_fp: Path | str | None = None,
        bounding_box: BoundingBoxModel | None = None,
        name: str = "georeferenced",
        prj4_str: str = MOON_GCS_PRJ,
        enabled: bool = True,
        save_output: bool = True,
    ) -> None:
        super().__init__(name, enabled, save_output)
        self.prj = prj4_str

        self.gcps: GCPGroup | None = None
        if gcps_fp is not None:
            self.gcps = GCPGroup.from_gcps_file(gcps_fp)

        self._bbox = bounding_box
        self.attrs: attrs.GeoreferencedAttrs

    def run(self, state: PipelineState) -> PipelineState:
        # ==== Retrieving latitude/longitude ====
        latlong_result = georeference.retrieve_latlong(
            state, self.cache, self.catalog, self.gcps
        )
        bbox, self.ngcps = georeference.set_bbox(
            self._bbox, latlong_result, self.gcps
        )
        swath, area = georeference.make_resampling_geometries(
            latlong_result, self.prj, bbox
        )
        self.loc = georeference.get_new_loc(area)
        gtrans = georeference.get_new_geotransform(area)

        state = georeference.modify_state(
            state, latlong_result, gtrans, swath, area, self.prj
        )
        return state

    def save(self, output: PipelineState) -> None:
        georeference.write_to_cache(
            self._time(), output, self.loc, self.cache, self.prj, self.ngcps
        )


class SolarRemoval(Step):
    """
    Solar Removal Step
    ---
    Solar spectrum removal step for the M<sup>3</sup> L1 to L2 pipeline. This
    step retrieves the solar spectrum from PDS metadata, scales it by the
    distance and a factor of 1/π and divides the entire data cube by this
    spectrum.

    Parameters
    ----------
    name : str, optional
        Name of the pipeline step. Default is "solar_removed".
    enabled : bool, optional
        Whether the step is enabled. Default is True.
    save_output : bool, optional
        Whether to save the output of this step to the cache. Default is True.
    """

    def __init__(
        self,
        name: str = "solar_removed",
        enabled: bool = True,
        save_output: bool = True,
    ) -> None:
        super().__init__(name, enabled, save_output)

    def run(self, state: PipelineState) -> PipelineState:
        self.solar_data = solar_removal.retrieve_solar_data(self.catalog)
        self.solar_data.scale_solar_spectrum()
        state = solar_removal.modify_state(state, self.solar_data)
        return state

    def save(self, output: PipelineState) -> None:
        solar_removal.write_to_cache(
            self._time(), self.cache, output, self.solar_data
        )


class StatisticalPolish(Step):
    """
    Statistical Polish Step
    ---
    Statistical polish step for the M<sup>3</sup> L1 to L2 pipeline. This step
    retrieves the statistical polish coefficients from the M3DataPaths catalog,
    determines whether the acquisition was during the warm or cold phase of the
    mission, and applies the statistical polish correction to the data cube.

    Parameters
    ----------
    name : str, optional
        Name of the pipeline step. Default is "statistical_polish".
    enabled : bool, optional
        Whether the step is enabled. Default is True.
    save_output : bool, optional
        Whether to save the output of this step to the cache. Default is True.
    """

    def __init__(
        self,
        name: str = "statisical_polished",
        enabled: bool = True,
        save_output: bool = True,
    ) -> None:
        self.instr_state: Literal["Warm", "Cold"]
        super().__init__(name, enabled, save_output)

    def run(self, state: PipelineState) -> PipelineState:
        self.coefs = statistical_polish.retrieve_statpol_coefs(self.catalog)
        self.instr_state = determine_warm_cold(
            self.catalog.id.acquisition_datetime
        )
        state = statistical_polish.modify_state(state, self.coefs)
        return state

    def save(self, output: PipelineState) -> None:
        statistical_polish.write_to_cache(
            self.cache, output, self._time(), self.coefs, self.instr_state
        )


class ThermalCorrection(Step):
    """
    Thermal Correction Step
    ---
    Thermal correction step for the M<sup>3</sup> L1 to L2 pipeline. This step
    applies thermal correction to the data cube based on the selected method.

    Parameters
    ----------
    method : ThermalCorrectionMethod
        The thermal correction method to use.
    name : str, optional
        Name of the pipeline step. Default is "thermal_corrected".
    enabled : bool, optional
        Whether the step is enabled. Default is True.
    save_output : bool, optional
        Whether to save the output of this step to the cache. Default is True.
    """

    def __init__(
        self,
        method: ThermalCorrectionMethod,
        name: str = "thermal_corrected",
        enabled: bool = True,
        save_output: bool = True,
    ) -> None:
        self.method: ThermalCorrectionMethod = method
        super().__init__(name, enabled, save_output)

    def run(self, state: PipelineState) -> PipelineState:
        state, self.temp_map = thermal_correction.modify_state(
            state, self.catalog, self.method
        )
        return state

    def save(self, output: PipelineState) -> None:
        thermal_correction.write_to_cache(
            self.cache, output, self._time(), self.temp_map, self.method
        )


class PhotometricCorrection(Step):
    def __init__(
        self,
        method: TopoCorrectionMethod,
        name: str = "photometric_corrected",
        enabled: bool = True,
        save_output: bool = True,
    ) -> None:
        super().__init__(name, enabled, save_output)
        self.method: TopoCorrectionMethod = method

    def run(self, state: PipelineState) -> PipelineState:
        phase_rgi = photometric_correction.get_phase_function_rgi(self.catalog)
        self.photo_coefs = photometric_correction.get_photometric_coefficients(
            state.obs.photometry_cube,
            self.method,
            phase_rgi,
            state.wavelengths,
        )
        state = photometric_correction.modify_state(state, self.photo_coefs)
        return state

    def save(self, output: PipelineState) -> None:
        photometric_correction.write_to_cache(
            self.cache, output, self._time(), self.photo_coefs, self.method
        )
