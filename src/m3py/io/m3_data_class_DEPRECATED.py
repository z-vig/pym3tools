# Standard Libraries
import os
from pathlib import Path

# Dependencies
import psutil
import numpy as np


class M3Data:
    """
    Parent class for reading and storing any M3Data.

    Parameters
    ----------
    img_path: str or path-like
        Path to *.img image file. *.HDR and *.LBL files must be in the same
        directory.
    data_format: dict[str, dict[str, str | int]]
        Data format dictionary. See "constants/m3_data_format.py".
    acq_type: str
        Acquisition type. Options are:

        - "global"
        - "targeted"

    memory_threshold: float, optional
        Maximum percentage of memory that can be used for image
        loading. Default is 0.75 (75%).

    Attributes
    ----------
    """

    def __init__(
        self,
        img_path: str | os.PathLike,
        data_format: dict[str, dict[str, str | int]],
        acq_type: str,
        memory_threshold: float = 0.75
    ):
        self._p = Path(img_path)

        self._nbands = data_format[acq_type]["nbands"]
        self._ncols = data_format[acq_type]["ncols"]
        self._dtype = data_format[acq_type]["dtype"]
        self._hdrlen = data_format[acq_type]["header_length"]
        if self._dtype == "<f":
            self._nbytes = 32 // 8
        elif self._dtype == "<h":
            self._nbytes = 16 // 8
        else:
            raise ValueError(f"{self._dtype} is an invalid dtype.")

        self._filesize = os.path.getsize(self._p)

        self._col_bytes = self._hdrlen + (self._ncols * self._nbands *
                                          self._nbytes)
        self._total_rows = self._filesize // self._col_bytes

        self._memchk = self._memory_check(memory_threshold)

        if self._memchk:
            self._chunk_rows = self._total_rows
        else:
            self._chunk_rows = (self._mem_size * memory_threshold) \
                // self._col_bytes

        self._byte_index = 0
        self._chunk_index = 0

        self._nchunks = (self._total_rows // self._chunk_rows) + 1

    def _memory_check(self, memory_threshold: float):
        """
        Returns False if the file size is too large to resonably fit into
        memory.

        Parameters
        ----------
        memory_threshold: float
            Maximum percentage of memory that can be used for image
            loading.
        """
        self._mem_size = psutil.virtual_memory().available
        print(f"Loaded File Size: {self._filesize*10**-9:.2f} GB "
              f"({self._total_rows} Rows, {self._filesize/self._mem_size:.1%}"
              " of RAM)\n"
              f"Available RAM: {self._mem_size*10**-9:.2f} GB")
        if self._filesize > memory_threshold * self._mem_size:
            print("Image larger than memory threshold detected.")
            return False
        else:
            return True

    def __iter__(self):
        return self

    def __next__(self):
        if self._byte_index < self._filesize:
            self._chunk_index += 1
            if self._chunk_index == self._nchunks:
                chunk_size = int(self._total_rows % self._chunk_rows)
            else:
                chunk_size = int(self._chunk_rows)

            chunk = np.empty([chunk_size, self._ncols, self._nbands])

            with open(self._p, "rb") as f:
                f.seek(self._byte_index)
                for i in range(0, chunk_size):
                    f.seek(self._hdrlen + self._byte_index)
                    for j in range(0, self._nbands):
                        bindat = f.read(self._ncols * self._nbytes)
                        row = np.frombuffer(bindat, dtype=self._dtype)
                        if row.shape[0] == 0:
                            print(self._byte_index, row.shape)
                        chunk[i, :, j] = row
                        self._byte_index = f.tell()

            print(f"CHUNK {self._chunk_index} complete.")
            return chunk
        else:
            raise StopIteration
