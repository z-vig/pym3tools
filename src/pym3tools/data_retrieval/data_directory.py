from pathlib import Path
from dataclasses import dataclass
from typing import Self
from warnings import warn

from pym3tools.m3catalog import DataIDString, is_valid_data_id, M3DataID


def init_data_dir(
    parent_dir: Path, data_id: DataIDString, overwrite: bool = False
) -> Path:
    """
    Initializes directory structure for M3 data download.

    Parameters
    ----------
    parent_dir: Path
        Parent directory into which the new M3 dataset directory will be made.
    data_id: DataIDString
        A valid M3 dataset/product id.
    overwrite: bool, optional
        Overwrite the current data, if it exists. Default is False.

    Returns
    -------
    Path
        Path to the new M3 dataset directory.
    """
    if not parent_dir.is_dir():
        raise ValueError(f"Parent Directory at {parent_dir} does not exist.")

    base_dir = parent_dir / data_id

    data_dir = base_dir / "pds_data"
    l0dir = data_dir / "L0"
    l1dir = data_dir / "L1"
    l2dir = data_dir / "L2"

    cal_dir = base_dir / "cal_data"

    if base_dir.exists() and (overwrite is False):
        warn(
            f"{base_dir} already exists. If you want to overwrite this data,"
            " set overwrite=True"
        )

    for i in [base_dir, data_dir, cal_dir, l0dir, l1dir, l2dir]:
        i.mkdir(exist_ok=True)

    return base_dir


@dataclass(frozen=True)
class PDSImagePath:
    """
    Encodes the three main components of a PDS compatible, ENVI image into a
    user-friendly class. The .img, .hdr and .lbl parts are all exposed.
    """

    img: Path
    hdr: Path
    json: Path

    @classmethod
    def from_base(cls, base: Path) -> Self:
        return cls(
            img=base.with_suffix(".img"),
            hdr=base.with_suffix(".hdr"),
            json=base.with_suffix(".json"),
        )

    @property
    def base(self) -> Path:
        return Path(self.img.parent, self.img.stem)

    def exists(self) -> bool:
        return self.img.exists() and self.hdr.exists()


@dataclass(frozen=True)
class PDSTabDataPath:
    """Encodes tabulated data from the PDS into a convenient class."""

    lbl: Path
    tab: Path

    @classmethod
    def from_base(cls, base: Path) -> Self:
        return cls(
            lbl=base.with_suffix(".lbl"),
            tab=base.with_suffix(".tab"),
        )

    @property
    def base(self) -> Path:
        return Path(self.tab.parent, self.tab.stem)

    def exists(self) -> bool:
        return self.lbl.exists() and self.tab.exists()


@dataclass(frozen=True)
class M3DataPaths:
    """
    User interface for the directory structure of downloaded M3 data.

    All paths are derived from a single root that is set by the user. The root
    should be the M3 Data ID.
    """

    _root: Path | str

    @property
    def root(self) -> Path:
        return Path(self._root)

    @property
    def id(self) -> M3DataID:
        id = self.root.name
        m3id = M3DataID.from_string(id)
        if not is_valid_data_id(m3id.string):
            raise ValueError("Invalid Data ID")
        return m3id

    @property
    def _cal_dir(self) -> Path:
        return self.root / "cal_data"

    @property
    def _data_dir(self) -> Path:
        return self.root / "pds_data"

    @property
    def _l0(self) -> Path:
        return self._data_dir / "L0"

    @property
    def _l1(self) -> Path:
        return self._data_dir / "L1"

    @property
    def _l2(self) -> Path:
        return self._data_dir / "L2"

    # ==== L0 Properties ====
    @property
    def raw(self) -> PDSImagePath:
        return PDSImagePath.from_base(self._l0 / f"{self.id.string}_V01_L0")

    @property
    def L0_lbl(self) -> Path:
        return Path(self._l0, f"{self.id.string}_V01_L0").with_suffix(".lbl")

    # ==== L1 Properties ====
    @property
    def rdn(self) -> PDSImagePath:
        return PDSImagePath.from_base(self._l1 / f"{self.id.string}_V03_RDN")

    @property
    def loc(self) -> PDSImagePath:
        return PDSImagePath.from_base(self._l1 / f"{self.id.string}_V03_LOC")

    @property
    def obs(self) -> PDSImagePath:
        return PDSImagePath.from_base(self._l1 / f"{self.id.string}_V03_OBS")

    @property
    def L1_lbl(self) -> Path:
        return Path(self._l1, f"{self.id.string}_V03_L1B").with_suffix(".lbl")

    # ==== L2 Properties ====
    @property
    def rfl(self) -> PDSImagePath:
        return PDSImagePath.from_base(self._l2 / f"{self.id.string}_V01_RFL")

    @property
    def sup(self) -> PDSImagePath:
        return PDSImagePath.from_base(self._l2 / f"{self.id.string}_V01_SUP")

    @property
    def L2_lbl(self) -> Path:
        return Path(self._l2, f"{self.id.string}_V01_L2.LBL")

    # ==== Calibration Properties ====
    @property
    def solar_spectrum(self) -> PDSTabDataPath:
        return PDSTabDataPath(
            lbl=self._cal_dir / "RFL_SOLAR_SPEC.lbl",
            tab=self._cal_dir / "RFL_SOLAR_SPEC.tab",
        )

    @property
    def ground_truth(self) -> PDSTabDataPath:
        return PDSTabDataPath(
            lbl=self._cal_dir / "RFL_GRND_TRU.lbl",
            tab=self._cal_dir / "RFL_GRND_TRU.tab",
        )

    @property
    def statistical_polish(self) -> PDSTabDataPath:
        return PDSTabDataPath(
            lbl=self._cal_dir / "RFL_STAT_POL.lbl",
            tab=self._cal_dir / "RFL_STAT_POL.tab",
        )

    @property
    def photometry_correction(self) -> PDSTabDataPath:
        return PDSTabDataPath(
            lbl=self._cal_dir / "RFL_F_ALPHA.lbl",
            tab=self._cal_dir / "RFL_F_ALPHA.tab",
        )

    def check_status(self, verbose: bool = False) -> bool:
        dirs = (
            self._data_dir.exists()
            and self._cal_dir.exists()
            and self._l0.exists()
            and self._l1.exists()
            and self._l2.exists()
        )

        l0_files = self.L0_lbl.exists() and self.raw.exists()
        l1_files = (
            self.L1_lbl.exists()
            and self.rdn.exists()
            and self.obs.exists()
            and self.loc.exists()
        )
        l2_files = (
            self.L2_lbl.exists() and self.rfl.exists() and self.sup.exists()
        )
        cal_files = (
            self.solar_spectrum.exists()
            and self.ground_truth.exists()
            and self.statistical_polish.exists()
            and self.photometry_correction.exists()
        )

        if verbose:
            print(
                f"All Directories Exist: {dirs}\n"
                f"All L0 Files Exist: {l0_files}\n"
                f"All L1 Files Exist: {l1_files}\n"
                f"All L2 Files Exist: {l2_files}\n"
                f"All Calibration Files Exist: {cal_files}"
            )

        if (
            (dirs is False)
            or (l0_files is False)
            or (l1_files is False)
            or (l2_files is False)
            or (cal_files is False)
        ):
            warn("Not all files are present. Check status for details.")
            return False

        return True
