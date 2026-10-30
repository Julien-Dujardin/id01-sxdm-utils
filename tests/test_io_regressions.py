"""Synthetic detector data regressions without worker processes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from sxdm.io import bliss


class SequentialPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def imap(self, fun, values):
        return map(fun, values)


class TestDetectorSums(unittest.TestCase):
    def test_irregular_and_fully_masked_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.h5"
            with h5py.File(path, "w") as f:
                f["1.1/instrument/det/data"] = np.array(
                    [[[1, 100], [100, 3]], [[2, 100], [100, 4]]]
                )
            with (
                patch.object(bliss, "get_detector_aliases", return_value=["det"]),
                patch.object(
                    bliss, "_get_chunk_indexes", return_value=[(0, 1), (1, 2)]
                ),
                patch.object(bliss.mp, "Pool", SequentialPool),
            ):
                result = bliss.get_sxdm_pos_sum(
                    path,
                    "1.1",
                    mask_detector=np.array([[False, True], [True, False]]),
                    pbar=False,
                )
                np.testing.assert_array_equal(result, [4, 6])
                result = bliss.get_sxdm_pos_sum(
                    path, "1.1", mask_detector=np.ones((2, 2), dtype=bool), pbar=False
                )
                np.testing.assert_array_equal(result, [0, 0])

    def test_multi_scan_forwards_detector_and_unformatted_path(self):
        template = "/{scan_no}/custom/{detector}"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.h5"
            with h5py.File(path, "w") as f:
                f["1.1/custom/other"] = np.ones((2, 2, 2))
            with (
                patch.object(
                    bliss, "get_detector_aliases", return_value=["det", "other"]
                ),
                patch.object(
                    bliss,
                    "get_sxdm_frame_sum",
                    side_effect=[np.ones((2, 2)), np.full((2, 2), 2.0)],
                ) as calc,
            ):
                result = bliss.get_sxdm_frame_sum_multi(
                    path,
                    scan_nums=["1.1", "2.1"],
                    detector="other",
                    path_data_h5=template,
                )
            self.assertEqual(result.shape, (2, 2, 2))
            np.testing.assert_array_equal(result[:, 0, 0], [1, 2])
            self.assertEqual(
                calc.call_args_list[0].kwargs,
                dict(detector="other", path_data_h5=template, pbar=False),
            )
            self.assertEqual(calc.call_args_list[1].args[1], "2.1")
            self.assertEqual(calc.call_args_list[1].kwargs["path_data_h5"], template)
