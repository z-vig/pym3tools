# Standard Librariess
from dataclasses import dataclass


@dataclass
class AcqTypeFormat():
    """
    Describes the format of a single M3 acquisition type.

    Attributes
    ----------
    nbands : int
        Number of spectral bands in the acquisition.
    ncols : int
        Number of columns (pixels per line) in the acquisition.
    dtype : str
        Data type of the stored values (e.g., '<h' for 16-bit int, '<f' for
        32-bit float).
    header_length : int
        Length of the file header in bytes.
    """
    nbands: int
    ncols: int
    dtype: str
    header_length: int


@dataclass
class M3DataFormat():
    """
    Stores data about how M3 image files are stored in the .IMG binary format.

    Attributes
    ----------
    global_fmt: AcqTypeFormat
        Format information for global acquisition
    targeted_fmt: AcqTypeFormat
        Format information for targeted acquisition

    Notes
    -----
    Attributes must be accessed using the aliases "global" and "targeted" i.e.,
        L0.global  # Returns `global_fmt` attribute
        L0.targeted  # Returns `targeted_fmt` attribute
    """
    global_fmt: AcqTypeFormat
    targeted_fmt: AcqTypeFormat

    def __getattr__(self, name):
        if name == "global":
            return self.global_fmt
        if name == "targeted":
            return self.targeted_fmt
        raise AttributeError(
            f"{type(self).__name__} object has no attribute {name}"
        )


L0 = M3DataFormat(
    AcqTypeFormat(86, 320, "<h", 1280),
    AcqTypeFormat(260, 640, "<h", 1280)
)

L1 = M3DataFormat(
    AcqTypeFormat(85, 304, "<f", 0),
    AcqTypeFormat(256, 608, "<f", 0)
)

L2 = M3DataFormat(
    AcqTypeFormat(85, 304, "<f", 0),
    AcqTypeFormat(256, 608, "<f", 0)
)

LOC = M3DataFormat(
    AcqTypeFormat(3, 304, "<d", 0),
    AcqTypeFormat(3, 608, "<d", 0)
)

OBS = M3DataFormat(
    AcqTypeFormat(10, 304, "<f", 0),
    AcqTypeFormat(10, 608, "<f", 0)
)

SUP = M3DataFormat(
    AcqTypeFormat(3, 304, "<f", 0),
    AcqTypeFormat(3, 608, "<f", 0)
)
