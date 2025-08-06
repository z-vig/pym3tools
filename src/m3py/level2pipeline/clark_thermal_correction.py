# Standard Libraries
from dataclasses import dataclass, field, MISSING
from typing import Tuple, Final
import re

# Dependencies
import numpy as np
from tqdm import tqdm  # type: ignore
import h5py as h5  # type: ignore

# Relative Imports
from .step import Step, PipelineState
from .photometric_correction import (
    photometric_correction_single_spectrum, get_rgi, compute_f_alpha
)

# Top-Level Imports
from m3py.io.read_m3 import get_wavelengths

# Some constants
h: Final[float] = 6.626 * 10**-34  # J*s, planck's constant
k_b: Final[float] = 1.381 * 10**-23  # J/K, boltzmann's constant
c: Final[float] = 2.998 * 10**8  # m/s, speed of light


def last_nonzero_val_3D(cube, return_index=False):
    """
    If you have an empty 3D image array with the first two dimensions being
    pixels and the third dimension of size N, and each pixel is filled in to a
    certain depth, M <= N, this function returns a 2D image array that picks
    out all the pixel values at position M.
    """
    nx, ny, _ = cube.shape
    result = np.empty((nx, ny), dtype=cube.dtype)
    for i in range(nx):
        for j in range(ny):
            vals = cube[i, j, :]
            nonzero_indices = np.nonzero(vals)[0]
            if return_index:
                result[i, j] = nonzero_indices[-1] if\
                    nonzero_indices.size > 0 else 0
            else:
                result[i, j] = vals[nonzero_indices[-1]] if\
                    nonzero_indices.size > 0 else 0
    return result


def last_nonzero_val_4D(cube):
    """
    An equivalent of `last_nonzero_val_3D` but for 4D starting arrays.
    """
    nx, ny, nz, _ = cube.shape
    last_idx = last_nonzero_val_3D(cube[:, :, 0, :], return_index=True)
    result = np.empty((nx, ny, nz), dtype=cube.dtype)
    for i in range(nx):
        for j in range(ny):
            idx = int(last_idx[i, j])
            result[i, j, :] = cube[i, j, :, idx]
    return result


def find_wvl(wvls: np.ndarray, targetwvl: float) -> Tuple[int, float]:
    """
        findλ(λ.targetλ)

    Given a list of wavelengths, `wvls`, find the index of a `targetwvl` and
    the actual wavelength closest to your target.

    Parameters
    ----------
    wvls: np.ndarray
        Wavelength array to search in.
    targetwvl:
        Wavelength to search for.

    Returns
    -------
    idx: int
        Index of the found wavelength.
    wvl: float
        Actual wavelength that is closest to the target wavelength (at idx).
    """

    idx = np.argmin(np.abs(wvls - targetwvl))
    return int(idx), wvls[idx]


def project_line(
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    x3: float
) -> float:
    """
    Given two points and third x-value, project line through the two points and
    return the projected Y value at the third point.

    Parameters
    ----------
    pt1: Tuple[float, float]
        First X, Y point to fit.
    pt2: Tuple[float, float]
        Second X, Y point to fit.
    x3: float
        X value to project line to.

    Returns
    -------
    y3: float
        Projected Y value at `x3`.
    """
    m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    y3 = m * (x3 - pt1[0]) + pt1[1]
    return y3


@dataclass
class RefWvl:
    target: int
    index: int
    actual: float


@dataclass
class RefWvlSet:
    A: RefWvl = field(default_factory=lambda: RefWvl(1550, 0, 0))
    B: RefWvl = field(default_factory=lambda: RefWvl(2350, 0, 0))
    C: RefWvl = field(default_factory=lambda: RefWvl(2700, 0, 0))
    D: RefWvl = field(default_factory=lambda: RefWvl(2280, 0, 0))
    E: RefWvl = field(default_factory=lambda: RefWvl(2590, 0, 0))

    @classmethod
    def from_data(cls, wvls: np.ndarray) -> "RefWvlSet":
        initializing_dict: dict[str, RefWvl] = {}
        for k, v in cls.__dataclass_fields__.items():
            if v.default_factory is not MISSING:
                target_wvl = v.default_factory().target
            else:
                raise ValueError("Error initializing RefWvlSet")
            wvl_index, actual_wvl = find_wvl(wvls, target_wvl)
            initializing_dict[k] = RefWvl(target_wvl, wvl_index, actual_wvl)
        return cls(**initializing_dict)


def get_temp(
    B: float,
    emiss: float,
    wvl: float,
    F: float
) -> float:
    return (h * c / (wvl * k_b)) *\
        (np.log(((2 * h * c**2 * emiss)/(F * B * wvl**5)) + 1))**-1


def get_temp_iter(
    B: float,
    emiss: float,
    wvl: float,
    F: float,
    phi: float
) -> float:
    return (h * c / (wvl * k_b)) *\
        (np.log(((2 * h * c**2 * emiss * phi) / (F * B * wvl**5)) + 1))**-1


def get_thermal_spectrum(
    wvl: np.ndarray,
    temperature: float,
    solar_spectrum: np.ndarray,
    solar_distance: float
) -> np.ndarray:
    if np.isfinite(temperature):
        B = ((2 * h * c**2) / (wvl ** 5)) *\
            (1 / (np.exp((h * c) / (wvl * k_b * temperature)) - 1))
        F = 10**6 * solar_spectrum / np.pi
        therm_spec = (B / F) * solar_distance**2
        return therm_spec
    elif np.isnan(temperature):
        return np.zeros(wvl.shape, dtype=np.float32)
    else:
        raise ValueError("Temperature is not NaN or finite float.")


def initial_temperature_estimate(
    spectrum: np.ndarray,
    refwvl: RefWvlSet,
    solar_wvl: np.ndarray,
    solar_spectrum: np.ndarray
) -> float:
    ptA = (refwvl.A.actual, spectrum[refwvl.A.index])
    ptB = (refwvl.B.actual, spectrum[refwvl.B.index])

    Yval_at_C = project_line(ptA, ptB, refwvl.C.actual)
    initial_thermal_component = spectrum[refwvl.C.index] - Yval_at_C

    if initial_thermal_component < 0:
        return np.nan

    initial_emissivity = 1 - spectrum[refwvl.A.index]

    Fidx = np.argmin(solar_wvl - refwvl.C.actual)  # Index of solar spectrum(F)

    initial_temp = get_temp(
        initial_thermal_component,
        initial_emissivity,
        refwvl.C.actual * 10**-9,
        10**6 * solar_spectrum[Fidx] / np.pi
    )

    return initial_temp


def iterative_temperature_estimate(
    spectrum: np.ndarray,
    refwvl: RefWvlSet,
    solar_wvl: np.ndarray,
    solar_spectrum: np.ndarray,
    photometric_coefs: np.ndarray
) -> float:
    wvl_dependent_emiss = 1 - spectrum

    ptA = (refwvl.D.actual, spectrum[refwvl.D.index])
    ptB = (refwvl.E.actual, spectrum[refwvl.E.index])

    Yval_at_C = project_line(ptA, ptB, refwvl.C.actual)
    next_thermal_component = spectrum[refwvl.C.index] - Yval_at_C

    if next_thermal_component < 0:
        return np.nan

    # Index of solar spectrum(F)
    Fidx = np.argmin(np.abs(solar_wvl - refwvl.C.actual))

    next_temp_estimate = get_temp_iter(
        next_thermal_component,
        wvl_dependent_emiss[refwvl.C.index],
        refwvl.C.actual * 10**-9,
        10**6 * solar_spectrum[Fidx] / np.pi,
        photometric_coefs[refwvl.C.index]
    )

    return next_temp_estimate


def remove_thermal(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    temperature: float,
    solar_spectrum: np.ndarray,
    solar_distance: float,
    return_thermal_spectrum: bool = False
) -> np.ndarray:
    thermal_spectrum = get_thermal_spectrum(
        wavelengths * 10**-9, temperature, solar_spectrum, solar_distance
    )

    if return_thermal_spectrum:
        return thermal_spectrum
    else:
        return spectrum - thermal_spectrum


class ClarkThermalCorrection(Step):
    def __init__(self, name, max_iterations: int = 12, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.max_iterations = max_iterations

    def run(self, state: PipelineState) -> PipelineState:
        temperature_maps = np.full(
            (*state.data.shape[:2], self.max_iterations),
            fill_value=np.nan,
            dtype=np.float32
        )
        correction_steps = np.full(
            (*state.data.shape, self.max_iterations),
            fill_value=np.nan,
            dtype=np.float32
        )

        reference_wvl = RefWvlSet.from_data(state.wvl)

        solspec_parse = re.compile(r"\s*(\d{2,4}.\d{6})")
        _, bbl = get_wavelengths(self.manager)
        with open(self.manager.cal_dir.solar_spectrum) as f:
            data_array = np.array(
                [re.findall(solspec_parse, i) for i in f.readlines()],
                dtype=np.float32
            )
            solar_wvl = data_array[bbl, 0]
            solar_spec = data_array[bbl, 1]

        solar_distance_pattern = re.compile(
            r"SOLAR_DISTANCE\s*=\s(\d.\d*)\s<AU>"
        )
        with open(self.manager.pds_dir.l1.lbl) as f:
            solar_distance = float(re.findall(
                solar_distance_pattern, f.read()
            )[0])

        pattern = re.compile(r"\s\d.\d{9}")
        with open(self.manager.cal_dir.phase_function) as f:
            phase_function_lookup = np.array(
                [re.findall(pattern, i) for i in f.readlines()[1:]],
                dtype=np.float32
            )
            phase_function_lookup = phase_function_lookup[:100, bbl]

        f_alpha_rgi = get_rgi(phase_function_lookup)
        wvl_size = state.wvl.size

        for i in tqdm(range(state.data.shape[0]),
                      desc="Iteratively solving pixels..."):
            for j in range(state.data.shape[1]):
                iter_counter = 0

                spectrum = state.data[i, j, :]

                if np.all(np.isnan(spectrum)):
                    continue

                # Removing solar distance correction
                spectrum *= solar_distance ** 2

                initial_temp = initial_temperature_estimate(
                    spectrum,
                    reference_wvl,
                    solar_wvl,
                    solar_spec
                )

                no_thermal = remove_thermal(
                    spectrum,
                    state.wvl,
                    initial_temp,
                    solar_spec,
                    solar_distance
                )

                i_val = state.obs[i, j, 0]
                e_val = state.obs[i, j, 1]
                g_val = state.obs[i, j, 2]

                f_alpha = compute_f_alpha(g_val, f_alpha_rgi, wvl_size)

                first_temp_photo_corrected, photo_coefs =\
                    photometric_correction_single_spectrum(
                        no_thermal, i_val, e_val, g_val, f_alpha,
                        limb_darkening="Lommel-Seeliger"
                    )

                temperature_maps[i, j, iter_counter] = initial_temp

                correction_steps[
                    i, j, :, iter_counter
                ] = first_temp_photo_corrected

                iter_counter += 1

                while True:
                    next_temp = iterative_temperature_estimate(
                        correction_steps[i, j, :, iter_counter],
                        reference_wvl,
                        solar_wvl,
                        solar_spec,
                        photo_coefs
                    )

                    next_no_thermal = remove_thermal(
                        spectrum,
                        state.wvl,
                        next_temp,
                        solar_spec,
                        solar_distance
                    )

                    next_temp_photo_corrected = photo_coefs * next_no_thermal
                    iter_counter += 1

                    if iter_counter < self.max_iterations:
                        temperature_maps[i, j, iter_counter] = next_temp

                        correction_steps[
                            i, j, :, iter_counter
                        ] = next_temp_photo_corrected
                    else:
                        break

                    if abs(
                        next_temp - temperature_maps[i, j, iter_counter-1]
                    ) < 2:
                        break

        self.final_temp_map = last_nonzero_val_3D(temperature_maps)
        final_temp_correction = last_nonzero_val_4D(correction_steps)

        new_state = PipelineState(
            data=final_temp_correction,
            wvl=state.wvl,
            obs=state.obs,
            georef=state.georef
        )

        return new_state

    def save(self, output: PipelineState) -> None:
        super().save(output)
        with h5.File(self.manager.cache, "r+") as f:
            g = f[self.name]
            assert isinstance(g, h5.Group)
            g.create_dataset("temperature_map",
                             data=self.final_temp_map, dtype="f4")
