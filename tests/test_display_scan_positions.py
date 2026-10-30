"""Shared affine mapping preserves relative scan positions and native hover data."""

import subprocess
import sys

import h5py
import numpy as np
import pytest

pytest.importorskip("plotly")
from sxdm.plot import display_scan_positions


@pytest.fixture
def scans(tmp_path):
    path = tmp_path / "scans.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("metadata")
        for name, x, y in (("1.1", [0, 100, 10], [2, 4, 2]), ("2.1", [20, 30], [3, 4])):
            scan = h5.create_group(name)
            scan["title"] = "amesh samx 0 100 100 samy 2 4 100 0.1"
            scan["start_time"] = "2026-10-30T10:00:00"
            scan["measurement/samx"] = x
            scan["measurement/samy"] = y
            for motor in ("samx", "samy"):
                scan[f"measurement/{motor}"].attrs["units"] = "um"
    return path


def test_shared_bounds_include_extrema_removed_by_sampling(scans):
    fig = display_scan_positions(scans, max_points=2)
    np.testing.assert_allclose(fig.data[0].x, [0, 0.1])
    np.testing.assert_allclose(fig.data[1].x, [0.2, 0.3])
    np.testing.assert_allclose(fig.data[1].y, [0.5, 1])
    assert fig.layout.meta["motor_bounds"] == (0, 100, 2, 4)
    assert "Mapped" in fig.layout.xaxis.title.text
    assert "um" not in fig.layout.xaxis.title.text
    assert "samx (um)" in fig.data[0].hovertemplate
    assert fig.data[0].customdata[1][4:] == [10, 2]
    assert fig.layout.shapes[0].xref == "x"


def test_custom_rectangle_fixed_native_bounds_and_subset(scans):
    from sxdm.io.bliss import display_scan_positions as display

    fig = display(
        scans, rectangle=(-2, 2, 0, 3), motor_bounds=(0, 100, 2, 4), scan_nos=["2.1"]
    )
    np.testing.assert_allclose(fig.data[0].x, [-1.2, -0.8])
    np.testing.assert_allclose(fig.data[0].y, [1.5, 3])
    assert fig.layout.meta["rectangle"] == (-2, 2, 0, 3)
    # A narrower source window extrapolates rather than clipping to its edges.
    fig = display(scans, motor_bounds=(0, 10, 2, 4), scan_nos=["2.1"])
    np.testing.assert_allclose(fig.data[0].x, [2, 3])


def test_constant_axes_and_nonfinite_gaps_are_preserved(tmp_path):
    path = tmp_path / "line.h5"
    with h5py.File(path, "w") as h5:
        h5["1.1/title"] = "ascan samx 0 2 2 0.1"
        h5["1.1/measurement/samx"] = [0, np.nan, 2]
        h5["1.1/instrument/positioners/samy"] = 4.0
    fig = display_scan_positions(path, x_motor="samx", y_motor="samy")
    np.testing.assert_allclose(fig.data[0].x, [0, np.nan, 1], equal_nan=True)
    np.testing.assert_allclose(fig.data[0].y, [0.5, 0.5, 0.5])
    assert fig.data[0].connectgaps is False


@pytest.mark.parametrize(
    "bounds",
    [(0, 0, 0, 1), (0, 1, 2, 1), (0, np.inf, 0, 1), (0, 1, 2), (False, 1, 0, 1)],
)
@pytest.mark.parametrize("parameter", ["rectangle", "motor_bounds"])
def test_invalid_bounds_fail_before_file_read(bounds, parameter):
    with pytest.raises(ValueError, match=parameter):
        display_scan_positions("missing.h5", **{parameter: bounds})


def test_empty_dataset_still_displays_reference_rectangle(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass
    fig = display_scan_positions(path)
    assert not fig.data
    assert len(fig.layout.shapes) == 1
    assert fig.layout.meta["motor_bounds"] is None


def test_html_table_does_not_import_widgets_and_escapes_metadata(tmp_path):
    path = tmp_path / "table.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("metadata")
        h5["2.1/title"] = "ascan <motor> 0 1 2 0.1"
        # Missing or invalid date is displayed rather than aborting the table.
        h5["2.1/start_time"] = "unknown"
        h5["1.1/title"] = "ct 0.1"
    script = """
import sys
from sxdm.io.bliss import get_scan_table
html = get_scan_table(sys.argv[1], as_widget=False)
assert isinstance(html, str)
assert '&lt;motor&gt;' in html and '<motor>' not in html
assert html.index('1.1') < html.index('2.1')
assert 'unknown' in html
assert 'ipywidgets' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
