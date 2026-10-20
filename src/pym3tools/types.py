from typing import Any, Literal, TypeGuard

from cubio import CubeContext, CubeData

type DataLevel = Literal["L0", "L1B", "L2"]


type AcquisitionMode = Literal["Global", "Target"]
acquisition_modes: list[AcquisitionMode] = ["Global", "Target"]
acq_char_to_mode: dict[str, AcquisitionMode] = {
    "G": "Global",
    "T": "Target",
}
acq_mode_to_char: dict[AcquisitionMode, str] = {
    v: k for k, v in acq_char_to_mode.items()
}


def is_valid_acquisition_mode(val: str) -> TypeGuard[AcquisitionMode]:
    return val in acquisition_modes


type M3ImageType = Literal["RDN", "RFL", "LOC", "OBS", "L0"]
m3_image_types: list[M3ImageType] = ["RDN", "RFL", "LOC", "OBS", "L0"]


def is_valid_m3_image_type(val: str) -> TypeGuard[M3ImageType]:
    return val in m3_image_types


type M3CalType = Literal[
    "SOLAR_SPECTRUM",
    "STATISTICAL_POLISH",
    "GROUND_TRUTH",
    "PHASE_ANGLE_CORRECTION",
]
m3_cal_types: list[M3CalType] = [
    "SOLAR_SPECTRUM",
    "STATISTICAL_POLISH",
    "GROUND_TRUTH",
    "PHASE_ANGLE_CORRECTION",
]

type OpticalPeriod = Literal["OP1A", "OP1B", "OP2A", "OP2B", "OP2C"]

type DatasetStatus = Literal["NotSet", "Set"]

"""
Thermal Correction Methods
--------------------------
Clark = Original M3 thermal correction derived by Clark et al., 2011
Clark_Modified = Uses the Clark et al., 2011 derived temeprature values from
    the PDS, but does not derived a thermal correction from the spectra.
Li_Milliken = Uses the Li and Milliken, 2016 method.
Shkuratov = Uses the Shkuratov et al., XXXX method.
Wohler_Grumpe = Uses the Wohler and Grumpe, 2022 method that fits a linear
    model to samples from the Apollo Collection.
"""
type ThermalCorrectionMethod = Literal[
    "Clark", "Clark_Modified", "Li_Milliken", "Shkuratov", "Wohler_Grumpe"
]


type TopoCorrectionMethod = Literal["Lommel-Seeliger", "Lunar Lambert"]

type CubioTuple = tuple[CubeContext, CubeData]


def is_cubio_tuple(val: Any) -> TypeGuard[CubioTuple]:
    return (
        isinstance(val, tuple)
        and isinstance(val[0], CubeData)
        and isinstance(val[1], CubeContext)
        and len(val) == 2
    )


type MosaicMethod = Literal["Mean", "MinimumIncidenceAngle", "MaxAlbedo"]

type DestripingMethod = Literal["UniformColumnMean", "GaussianFilter"]
