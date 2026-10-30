"""Create a small synthetic BLISS file: uv run examples/create_demo.py /tmp/sxdm-demo.h5."""

from pathlib import Path
import argparse

import h5py
import numpy as np


def create_demo(path):
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[:24, :32]
    with h5py.File(path, "x") as h5:
        for scan_no, offset in enumerate((0, 4, 8), 1):
            x, y = np.meshgrid(
                np.linspace(offset, offset + 5, 16), np.linspace(0, 4, 12)
            )
            # Interrupt the last scan to demonstrate partial acquisition.
            count = x.size if scan_no < 3 else 93
            x, y = x.ravel()[:count], y.ravel()[:count]
            scan = h5.create_group(f"{scan_no}.1")
            scan["title"] = f"mesh samx {offset} {offset + 5} 15 samy 0 4 11 0.01"
            scan["start_time"] = f"2026-10-30T10:0{scan_no}:00+00:00"
            for motor, values in (("samx", x), ("samy", y)):
                scan[f"measurement/{motor}"] = values
                scan[f"measurement/{motor}"].attrs["units"] = "um"
            intensity = 30 + 400 * np.exp(-((x - offset - 2.5) ** 2 + (y - 2) ** 2) / 2)
            scan["measurement/det_roi"] = intensity
            frames = rng.poisson(
                intensity[:, None, None]
                * np.exp(-((xx - 16) ** 2 + (yy - 12) ** 2) / 25)
                + 2
            ).astype("u2")
            detector = scan.create_group("instrument/det")
            detector.attrs["NX_class"] = "NXdetector"
            detector.create_dataset(
                "data", data=frames, chunks=(1, 24, 32), compression="gzip"
            )
            scan["measurement/det"] = detector["data"]
    return Path(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    print(create_demo(parser.parse_args().path))
