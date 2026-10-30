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
    assert trace.customdata[1] == ["1.1", 3]
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
    assert len(fig.data[0].x) <= 2
    assert fig.data[0].mode == "markers"
    assert fig.data[0].connectgaps is False


def test_total_budget_preserves_all_scans_endpoints_and_bounds(path):
    with h5py.File(path, "a") as f:
        for number in range(2, 6):
            f.copy("1.1", f"{number}.1")
    fig = get_scan_map(path, max_points=8, max_total_points=17)
    assert len(fig.data) == 5
    assert sum(len(trace.x) for trace in fig.data) <= 17
    for number, trace in enumerate(fig.data, 1):
        assert len(trace.x) <= 8
        assert trace.name == f"{number}.1"
        assert [row[1] for row in trace.customdata][:: len(trace.x) - 1] == [0, 11]
    with pytest.raises(ValueError, match="at least 10"):
        get_scan_map(path, max_total_points=9)
    complete = get_scan_map(path, max_points=None, max_total_points=None)
    assert sum(len(trace.x) for trace in complete.data) == 60


def test_gap_separators_fit_both_budgets_without_false_connections(path):
    with h5py.File(path, "a") as f:
        for motor, data in (
            ("piy", np.arange(100, dtype=float)),
            ("piz", np.ones(100)),
        ):
            del f[f"1.1/measurement/{motor}"]
            f[f"1.1/measurement/{motor}"] = data
        f["1.1/measurement/piy"][1::2] = np.nan
    fig = get_scan_map(path, max_points=12, max_total_points=11)
    trace = fig.data[0]
    assert len(trace.x) <= 11
    indexes = np.asarray(trace.customdata)[:, 1].astype(int)
    finite = np.isfinite(trace.x) & np.isfinite(trace.y)
    if trace.mode == "lines+markers":
        for i in range(len(indexes) - 1):
            # Every finite-to-finite segment is a real uninterrupted acquisition
            # segment. Alternating missing samples must always break the line.
            assert not (finite[i] and finite[i + 1] and indexes[i + 1] > indexes[i] + 1)


def test_singleton_scan_keeps_its_own_minimum_budget(path):
    with h5py.File(path, "a") as f:
        f["2.1/title"] = "ct 0.1"
        f["2.1/instrument/positioners/piy"] = 2.0
        f["2.1/instrument/positioners/piz"] = 3.0
    fig = get_scan_map(path, max_total_points=3)
    assert [len(trace.x) for trace in fig.data] == [2, 1]
    assert fig.data[1].customdata[0] == ["2.1", 0]
    with pytest.raises(ValueError, match="at least 3"):
        get_scan_map(path, max_total_points=2)


def test_sampling_retains_finite_endpoints_and_spatial_extrema(path):
    with h5py.File(path, "a") as f:
        for motor, data in (
            ("piy", [np.nan, 0, 100, -100, 4, np.nan]),
            ("piz", [1, 1, 1, 1, 1, 1]),
        ):
            del f[f"1.1/measurement/{motor}"]
            f[f"1.1/measurement/{motor}"] = data
    trace = get_scan_map(path, max_total_points=4).data[0]
    np.testing.assert_array_equal(trace.x, [0, 100, -100, 4])
    assert [row[1] for row in trace.customdata] == [1, 2, 3, 4]


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
    for count in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="max_total_points"):
            get_scan_map("missing.h5", max_total_points=count)
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
