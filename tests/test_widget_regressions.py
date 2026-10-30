from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip("ipywidgets", reason="Legacy widget tests need the jupyter extra")
plt = pytest.importorskip("matplotlib.pyplot", reason="Legacy widgets need matplotlib")

from sxdm.widgets.Inspect4DArray import Inspect4DArray  # noqa: E402
from sxdm.widgets.bliss.GetShift import GetShift, GetShiftCustom  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_hdf5_inspector_projections_and_navigation(tmp_path):
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    with h5py.File(tmp_path / "data.h5", "w") as h5:
        dataset = h5.create_dataset("data", data=data)
        viewer = Inspect4DArray(dataset)
        np.testing.assert_allclose(viewer.dir_space_data, data.sum((2, 3)))
        np.testing.assert_allclose(viewer.rec_space_data, data.sum((0, 1)))
        viewer._onkey(SimpleNamespace(key="left"))
        assert viewer.col == 0
        for _ in range(5):
            viewer._onkey(SimpleNamespace(key="right"))
        assert viewer.col == 2
        np.testing.assert_array_equal(viewer.higher_img.get_array(), data[0, 2])


def test_array_inspector_navigation_before_click_and_at_boundaries():
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    viewer = Inspect4DArray(data)
    viewer._onkey(SimpleNamespace(key="left"))
    assert viewer.col == 0
    for _ in range(5):
        viewer._onkey(SimpleNamespace(key="up"))
    assert viewer.row == 1
    np.testing.assert_array_equal(viewer.higher_img.get_array(), data[1, 0])


@pytest.mark.parametrize("zero", [False, True])
def test_getshift_retains_supplied_shifts(monkeypatch, zero):
    # Isolate file-independent initialization while still exercising shift setup.
    import importlib

    module = importlib.import_module("sxdm.widgets.bliss.GetShift")
    monkeypatch.setattr(module, "get_detector_aliases", lambda *a: ["det"])
    monkeypatch.setattr(module, "list_available_counters", lambda *a: ["det_roi"])
    monkeypatch.setattr(
        module,
        "get_roidata",
        lambda *a, **kw: np.zeros((3, 4)) if zero else np.ones((3, 4)),
    )
    monkeypatch.setattr(module, "get_piezo_motor_names", lambda *a: ("x", "y"))
    shifts = np.array([[0.0, 0.0], [1.0, 2.0]])
    viewer = GetShift("unused.h5", ["1.1", "2.1"], shifts=shifts)
    assert viewer.selector.layout.align_items == "stretch"
    np.testing.assert_array_equal(viewer.shifts, shifts)
    viewer.shifts[1, 0] = 99
    assert shifts[1, 0] == 1


def test_custom_getshift_retains_initial_shifts():
    shifts = np.array([[0.0, 0.0], [1.0, 2.0]])
    viewer = GetShiftCustom([np.ones((3, 4)), np.ones((3, 4))], init_shifts=shifts)
    assert viewer.selector.layout.align_items == "stretch"
    np.testing.assert_array_equal(viewer.shifts, shifts)
    viewer._calc_shifts()
    assert viewer.selector.layout.align_items == "stretch"
    np.testing.assert_array_equal(viewer.shifts, shifts)


def test_custom_getshift_zero_image():
    viewer = GetShiftCustom([np.zeros((3, 4))])
    assert viewer.clims == [0, 1]
