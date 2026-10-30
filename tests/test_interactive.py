"""Browser figures preserve coordinates and hover data without widget runtimes."""

import json
import sys

import h5py
import numpy as np
import pytest

pytest.importorskip("plotly")

from sxdm.plot.interactive import get_counter_map, get_scan_map


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as f:
        f["1.1/title"] = "mesh piy 0 10 100 piz 0 20 100 0.1"
        f["1.1/start_time"] = "2026-10-30T10:00:00"
        f["1.1/measurement/piy"] = np.arange(12, dtype=float)
        f["1.1/measurement/piz"] = np.ones(12)
        f["1.1/measurement/intensity"] = np.arange(12) * 10.0
    return path


def test_trajectory_hover_and_export(path, tmp_path):
    fig = get_scan_map(path, max_points=4)
    trace = fig.data[0]
    assert trace.type == "scattergl"
    assert trace.mode == "lines+markers"
    np.testing.assert_array_equal(trace.x, [0, 3, 7, 11])
    assert trace.customdata[1] == [
        "1.1",
        "mesh piy 0 10 100 piz 0 20 100 0.1",
        "2026-10-30T10:00:00",
        3,
    ]
    assert "Scan 1.1" in trace.hovertemplate
    assert "mesh piy" in trace.hovertemplate
    assert fig.layout.xaxis.title.text == "piy"
    assert trace.meta["acquired_points"] == 12
    assert trace.meta["displayed_points"] == 4
    assert json.loads(fig.to_json())["data"][0]["name"] == "1.1"
    target = tmp_path / "scans.html"
    fig.write_html(target)
    assert "plotly.js" in target.read_text()


def test_sampling_keeps_nonfinite_gaps(path):
    with h5py.File(path, "a") as f:
        f["1.1/measurement/piy"][5] = np.nan
    fig = get_scan_map(path, max_points=2)
    assert np.isnan(fig.data[0].x).any()
    assert fig.data[0].connectgaps is False


def test_counter_map_keeps_acquired_values_and_handles_partial_scan(path):
    with h5py.File(path, "a") as f:
        del f["1.1/measurement/intensity"]
        f["1.1/measurement/intensity"] = [0, 10, 100]
    with pytest.warns(UserWarning, match="shared acquired prefix"):
        fig = get_counter_map(path, "1.1", "intensity", log=True)
    trace = fig.data[0]
    np.testing.assert_array_equal(trace.x, [1, 2])
    np.testing.assert_array_equal(trace.marker.color, [1, 2])
    np.testing.assert_array_equal(np.asarray(trace.customdata)[:, 1], [10, 100])
    assert trace.marker.colorbar.title.text == "log10(intensity)"


def test_counter_map_rejects_detector_images(path):
    with h5py.File(path, "a") as f:
        f["1.1/measurement/detector"] = np.zeros((12, 2, 2))
    with pytest.raises(ValueError, match="numeric measurement"):
        get_counter_map(path, "1.1", "detector")


def test_integer_scan_number_normalization(path):
    fig = get_counter_map(path, 1, "intensity")
    assert fig.data[0].name == "1.1"


def test_invalid_sample_count_and_empty_result(path, tmp_path):
    for count in (True, 0, 1, -1, 1.5):
        with pytest.raises(ValueError, match="max_points"):
            get_scan_map(path, max_points=count)
    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w"):
        pass
    fig = get_scan_map(empty)
    assert not fig.data
    assert "No scans" in fig.layout.annotations[0].text


def test_units_must_match_between_scans(path):
    with h5py.File(path, "a") as f:
        f.copy("1.1", "2.1")
        f["2.1/measurement/piy"].attrs["units"] = "mm"
    with pytest.raises(ValueError, match="units differ"):
        get_scan_map(path)


def test_actionable_optional_dependency_error(path, monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(ImportError, match="uv sync --extra marimo"):
        get_scan_map(path)
