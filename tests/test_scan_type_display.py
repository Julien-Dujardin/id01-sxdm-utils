"""The selected family drives both maps and plain HTML scan tables."""

import h5py
import pytest

from sxdm.io.bliss import get_scan_table


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "types.h5"
    commands = [
        "asxdm",
        "amesh",
        "ascan",
        "ct",
        "sxdm",
        "dmesh",
        "dscan",
        "asxdm_extra",
    ]
    with h5py.File(path, "w") as h5:
        for i, command in enumerate(commands, 1):
            title = (
                f"{command} x 0 2 2 y 0 1 1 0.1"
                if command not in {"ascan", "dscan", "ct"}
                else f"{command} x 0 2 2 0.1"
            )
            h5[f"{i}.1/title"] = title
            h5[f"{i}.1/measurement/x"] = [0, 1, 2]
            h5[f"{i}.1/measurement/y"] = [i, i, i]
    return path


@pytest.mark.parametrize(
    "scan_type, expected",
    [
        (None, [1, 2, 3, 4, 5, 6, 7, 8]),
        ("sxdm", [1, 5]),
        ("mesh", [2, 6]),
        ("ascan", [3, 7]),
        (["sxdm", "mesh"], [1, 2, 5, 6]),
        ([], []),
        ("asxdm", [1]),
    ],
)
def test_table_family_filter(path, scan_type, expected):
    html = get_scan_table(path, as_widget=False, scan_type=scan_type)
    for i in range(1, 9):
        assert (f"<th>{i}.1</th>" in html) == (i in expected)


@pytest.mark.parametrize("function", ["get_scan_map", "display_scan_positions"])
@pytest.mark.parametrize(
    "scan_type, expected",
    [
        ("sxdm", ["1.1", "5.1"]),
        ("mesh", ["2.1", "6.1"]),
        ("ascan", ["3.1", "7.1"]),
        ([], []),
    ],
)
def test_map_family_filter_without_explicit_axes(path, function, scan_type, expected):
    pytest.importorskip("plotly")
    from sxdm.io import bliss

    fig = getattr(bliss, function)(path, scan_type=scan_type)
    assert [trace.name for trace in fig.data] == expected
    assert sum(len(t.x) for t in fig.data) <= 10000


def test_table_unknown_type_fails_before_file_read():
    with pytest.raises(ValueError, match="scan_type"):
        get_scan_table("missing.h5", as_widget=False, scan_type="typo")


def test_plot_zoom_revision_changes_with_motor_frame_and_mapping(path):
    pytest.importorskip("plotly")
    from sxdm.plot import get_scan_map, display_scan_positions

    original = get_scan_map(path)
    with h5py.File(path, "a") as h5:
        for scan in h5.values():
            scan["measurement/u"] = [0, 1, 2]
            scan["measurement/v"] = [3, 4, 5]
    changed = get_scan_map(path, x_motor="u", y_motor="v")
    assert original.layout.uirevision != changed.layout.uirevision
    original = display_scan_positions(path)
    changed = display_scan_positions(path, motor_bounds=(-1, 10, 0, 20))
    assert original.layout.uirevision != changed.layout.uirevision
    assert original.layout.uirevision != get_scan_map(path).layout.uirevision


def test_notebook_filters_preserve_reference_frame(path):
    pytest.importorskip("plotly")
    pytest.importorskip("marimo")
    import runpy
    from pathlib import Path
    from types import SimpleNamespace

    app = runpy.run_path(
        str(Path(__file__).parents[1] / "examples/display_scan_positions.py")
    )["app"]
    reference = None
    for family in ("all", "sxdm", "mesh", "ascan"):
        _, definitions = app.run(
            defs={
                "dataset": SimpleNamespace(value=str(path)),
                "scan_type_choice": SimpleNamespace(value=family),
            }
        )
        metadata = definitions["figure"].layout.meta
        actual = (metadata["x_motor"], metadata["y_motor"], metadata["motor_bounds"])
        if reference is None:
            reference = actual
        assert actual == reference


def test_notebook_rejects_mixed_reference_units_even_when_filtered(path):
    pytest.importorskip("plotly")
    pytest.importorskip("marimo")
    import runpy
    from pathlib import Path
    from types import SimpleNamespace

    with h5py.File(path, "a") as h5:
        for scan in h5.values():
            scan["measurement/x"].attrs["units"] = "um"
            scan["measurement/y"].attrs["units"] = "um"
        h5["2.1/measurement/x"].attrs["units"] = "mm"
    app = runpy.run_path(
        str(Path(__file__).parents[1] / "examples/display_scan_positions.py")
    )["app"]
    _, definitions = app.run(
        defs={
            "dataset": SimpleNamespace(value=str(path)),
            "scan_type_choice": SimpleNamespace(value="sxdm"),
        }
    )
    assert "figure" not in definitions
