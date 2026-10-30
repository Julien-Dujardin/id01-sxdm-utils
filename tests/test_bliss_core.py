"""Standalone BLISS readers and bounded reductions against NumPy reference data."""

import h5py
import hdf5plugin
import numpy as np
import pytest

from sxdm.io import bliss
from sxdm.io._bliss import get_command, get_counter, get_positioner, get_scan_shape
from sxdm.process.math import project


@pytest.fixture
def detector_file(tmp_path):
    path = tmp_path / "bliss.h5"
    data = np.arange(7 * 3 * 4, dtype=np.uint16).reshape(7, 3, 4)
    with h5py.File(path, "w") as h5:
        h5["1.1/title"] = "sxdm(x, 0, 3, 4, y, 0, 1, 2, 0.1)"
        h5["1.1/technique/dim0"] = 4
        h5["1.1/technique/dim1"] = 2
        h5["1.1/instrument/positioners/x"] = np.arange(7)
        h5["1.1/measurement/roi"] = np.arange(7)
        det = h5.create_group("1.1/instrument/custom_detector")
        det.attrs["NX_class"] = "NXdetector"
        det.create_dataset("data", data=data, **hdf5plugin.Bitshuffle())
    return path, data


def test_core_readers_accept_paths_and_caller_owned_handles(detector_file):
    path, _ = detector_file
    with h5py.File(path) as h5:
        assert get_scan_shape(h5, 1) == [2, 4]
        assert get_command(h5, 1).startswith("sxdm")
        np.testing.assert_array_equal(get_counter(h5, 1, "roi"), np.arange(7))
        np.testing.assert_array_equal(get_positioner(h5, 1, "x"), np.arange(7))
        assert h5.id.valid
    assert get_scan_shape(path, "1.1") == [2, 4]


def test_single_compressed_frame_with_cropping(detector_file):
    path, data = detector_file
    np.testing.assert_array_equal(bliss.get_detector_frame(path, 1, 3), data[3])
    np.testing.assert_array_equal(
        bliss.get_detector_frame(path, "1.1", 2, roi=(1, 3, 0, 2)), data[2, 1:3, :2]
    )
    with pytest.raises(IndexError):
        bliss.get_detector_frame(path, 1, 7)
    with pytest.raises(ValueError, match="ROI"):
        bliss.get_detector_frame(path, 1, roi=(0, 30, 0, 2))


def test_streamed_reductions_masks_roi_and_multiple_batches(detector_file, monkeypatch):
    path, data = detector_file
    monkeypatch.setattr(bliss, "_batch_size", lambda *args: 2)
    mask = np.array([False, True, False, False, True, False, False])
    got = bliss.get_sxdm_frame_sum(
        path, "1.1", mask_sample=mask, roi=(1, 3, 0, 3), pbar=False
    )
    np.testing.assert_array_equal(got, data[~mask, 1:3, :3].sum(0))
    det_mask = np.indices((3, 4)).sum(0) % 2 == 0
    got = bliss.get_sxdm_pos_sum(path, "1.1", mask_detector=det_mask, pbar=False)
    np.testing.assert_array_equal(got, np.where(det_mask, 0, data).sum((1, 2)))
    got = bliss.get_sxdm_frame_sum(
        path, "1.1", mask_sample=np.ones(7, dtype=bool), pbar=False
    )
    np.testing.assert_array_equal(got, np.zeros((3, 4)))


def test_integer_projections_support_float_background_subtraction():
    data = np.ones((2, 3, 4), dtype=np.uint16)
    x, y, z = project(data)
    assert x.dtype.kind == "f"
    x -= np.full(2, 0.5)
    np.testing.assert_allclose(x, [11.5, 11.5])
    np.testing.assert_allclose(y, [8, 8, 8])
    np.testing.assert_allclose(z, [6, 6, 6, 6])
