"""Regression tests for numerical analysis and masked detector data."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from sxdm.process import math as smath
from sxdm.utils import general


class SequentialPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def imap(self, fun, values):
        return map(fun, values)


class TestAnalysisRegressions(unittest.TestCase):
    def test_com_top_pixels_larger_than_array(self):
        arr = np.array([1.0, 3.0])
        coords = np.array([0.0, 2.0])
        np.testing.assert_allclose(
            smath.calc_com_3d(arr, coords, coords, coords, n_pix=10), [1.5] * 3
        )
        with self.assertRaises(ValueError):
            smath.calc_com_3d(arr, coords, coords, coords, n_pix=0)

    def test_angle_parallel_vectors_roundoff(self):
        vec = np.array(
            [[[0.2804087579860399, 0.48519097443163506, 0.9807371998012386]]]
        )
        angle = smath.ang_between(vec, vec)
        self.assertTrue(np.isfinite(angle).all())
        np.testing.assert_allclose(angle, 0, atol=1e-6)

    def test_qspace_cache_tracks_file_mask_and_coordinate_system(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"{i}.h5" for i in range(2)]
            for i, path in enumerate(paths):
                with h5py.File(path, "w") as f:
                    f["Data/qspace"] = np.ones((1, 2, 2, 2))
                    for name in ("qx", "qy", "qz"):
                        f[f"Data/{name}"] = np.array([0.0, 2.0]) + 10 * i
                    for name in ("pitch", "roll", "radial"):
                        f[f"Data/{name}"] = np.array([20.0, 22.0])
            mask = np.zeros((2, 2, 2), dtype=bool)
            with patch.object(smath, "_per_process_cache", None):
                np.testing.assert_allclose(
                    smath._calc_com_qspace3d(paths[0], mask, 0), [1] * 3
                )
                np.testing.assert_allclose(
                    smath._calc_com_qspace3d(paths[1], mask, 0), [11] * 3
                )
                np.testing.assert_allclose(
                    smath._calc_com_qspace3d(paths[0], mask, 0, spherical=True),
                    [21] * 3,
                )
                mask[1] = True
                np.testing.assert_allclose(
                    smath._calc_com_qspace3d(paths[0], mask, 0), [0, 1, 1]
                )

    def test_detector_com_excludes_pixels_inside_mask_bounding_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.h5"
            with h5py.File(path, "w") as f:
                f["1.1/instrument/det/data"] = np.array(
                    [[[1.0, 1000.0], [1000.0, 3.0]]]
                )
            mask = np.array([[False, True], [True, False]])
            coords = np.array([[0.0, 10.0], [10.0, 2.0]])
            with (
                patch.object(smath, "get_detector_aliases", return_value=["det"]),
                patch.object(smath, "_get_chunk_indexes", return_value=[(0, 1)]),
                patch.object(smath.mp, "Pool", SequentialPool),
            ):
                got = smath.calc_coms_qspace2d(
                    path,
                    "1.1",
                    coords,
                    coords,
                    coords,
                    mask_rec=mask,
                    detector="det",
                    pbar=False,
                )
            np.testing.assert_allclose(got, [[1.5] * 3])

    def test_log_maps_zero_and_negative_pixels_are_initialized(self):
        image = np.array([[0.0, -1.0], [1.0, np.e]])
        with (
            patch.object(general, "get_roidata", return_value=image),
            patch.object(general, "get_shift", return_value=np.array([[0, 0]])),
        ):
            _, raw, _ = general.get_shift_dset(
                "unused", "roi", ["1.1"], log=True, return_maps=True
            )
        np.testing.assert_allclose(raw[0], [[0, 0], [0, 1]])
