# Standard Libraries
import os
from pathlib import Path

# Dependencies
import numpy as np

type PathType = str | os.PathLike


class Step:
    def __init__(
        self,
        name: str,
        enabled: bool = True,
        save_output=False
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.save_output = save_output

    def run(self, data: np.ndarray):
        raise NotImplementedError("Subclasses must implement run()")

    def save(self, output: PathType):
        output_path = Path(output)
        print(f"Not saved (yet) to: {output_path}")

    def load(self):
        pass

    def execute(self, data: np.ndarray):
        if not self.enabled:
            print(f"Skipping {self.name}")
            return

        cached_step = self.load()
        if cached_step is not None:
            return cached_step

        output = self.run(data)

        if self.save_output:
            self.save(output)
