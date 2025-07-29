# Standard Libraries
from pathlib import Path
import os

# Dependencies
# import numpy as np
import matplotlib.pyplot as plt

# Relative Imports
from .read_m3 import read_m3, Window


class M3DataWindow():
    def __init__(
        self,
        img_path: str | os.PathLike,
        data_format: str,
        acq_type: dict[str | dict[int | str]],
        width: int,
        height: int,
    ):
        self.path = Path(img_path)
        self._fmt = data_format
        self._acq = acq_type

        self.row = 0
        self.col = 0

        self._window = Window(
            X=self.col,
            Y=self.row,
            H=height,
            W=width
        )

        self.max_x = data_format[acq_type]["ncols"]

        nbands = data_format[acq_type]["nbands"]
        dtype = data_format[acq_type]["dtype"]
        hdrlen = data_format[acq_type]["header_length"]

        if dtype == "<f":
            nbytes = 32 // 8
        elif dtype == "<h":
            nbytes = 16 // 8
        else:
            raise ValueError(f"{dtype} is an invalid dtype.")

        full_col_bytes = hdrlen + (self.max_x * nbands * nbytes)
        self.max_y = os.path.getsize(img_path) // full_col_bytes

    def read_tile(self):
        """Reads current tile window"""
        arr = read_m3(
            img_path=self.path,
            data_format=self._fmt,
            acq_type=self._acq,
            window=self._window
        )
        return arr

    def move(self, dx, dy):
        print(self.col, self.row, dx, dy)
        self.col = max(0, min(self.col + dx, self.max_x - self._window.W))
        self.row = max(0, min(self.row + dy, self.max_y - self._window.H))
        self._window.X = self.col
        self._window.Y = self.row
        return self.read_tile()


def show_lazy_image(
    path: str | os.PathLike,
    format: dict[str, dict[str | int]],
    acq_type: str,
    tile_h=512,
    tile_w=100,
    sensitivity: float = 0.5
):
    scroller = M3DataWindow(path, format, acq_type, tile_w, tile_h)
    tile = scroller.read_tile()

    fig, ax = plt.subplots()
    img = ax.imshow(tile[:, :, 20])
    ax.set_title("Use arrow keys to scroll")
    ax.axis("off")

    def on_key(event):
        key = event.key
        dx = dy = 0
        if key == "up":
            dy = int(tile_h * sensitivity)
        elif key == "down":
            dy = -int(tile_h * sensitivity)
        elif key == "left":
            dx = -int(tile_w * sensitivity)
        elif key == "right":
            dx = int(tile_w * sensitivity)
        else:
            return
        tile = scroller.move(dx, dy)
        img.set_data(tile[:, :, 20])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
