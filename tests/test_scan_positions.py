"""Coordinates come from acquired motor channels, never requested bounds."""

import h5py
import numpy as np
import pytest

from sxdm.io.scan_positions import get_scan_positions


def add_scan(h5f, name, title, x=None, y=None, *, static_y=None):
    scan = h5f.create_group(name)
    scan["title"] = title
    scan["start_time"] = "2026-10-30T09:00:00+00:00"
    if x is not None:
        scan["measurement/piy"] = x
        scan["measurement/piy"].attrs["units"] = "um"
    if y is not None:
        scan["measurement/piz_position"] = y
        scan["measurement/piz_position"].attrs["units"] = "um"
    if static_y is not None:
        scan["instrument/positioners/piz"] = static_y
        scan["instrument/positioners/piz"].attrs["units"] = "um"
    return scan


@pytest.fixture
def scans(tmp_path):
    path = tmp_path / "scans.h5"
    with h5py.File(path, "w") as f:
        f.create_group("metadata")
        add_scan(f, "10.1", "ct 0.1", [7], [8])
        add_scan(f, "2.1", "ascan piy 0 10 100 0.1", [1, 2, 3], static_y=9)
        add_scan(
            f,
            "1.1",
            "sxdm(piy, 0, 10, 100, piz, 0, 10, 100, 0.1)",
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        )
    return path


def test_mesh_line_and_point_sorted_measured_only(scans):
    result = get_scan_positions(scans)
    assert [scan.scan_no for scan in result] == ["1.1", "2.1", "10.1"]
    assert (result[0].x_motor, result[0].y_motor) == ("piy", "piz")
    np.testing.assert_array_equal(result[0].x, [0, 1, 0, 1])
    np.testing.assert_array_equal(result[1].x, [1, 2, 3])
    np.testing.assert_array_equal(result[1].y, [9, 9, 9])
    np.testing.assert_array_equal(result[2].x, [7])
    assert result[0].x_unit == "um"
    assert result[0].title.startswith("sxdm(")


@pytest.mark.parametrize("command", ["amesh", "asxdm", "dsxdm"])
def test_mesh_aliases_infer_axes_without_reconstructing_requested_bounds(
    tmp_path, command
):
    path = tmp_path / "aliases.h5"
    with h5py.File(path, "w") as f:
        add_scan(
            f,
            "1.1",
            f"{command}(piy, 0, 100, 10, piz, 0, 200, 10, 0.1)",
            [1, 2, 3],
            [4, 4, 5],
        )
    scan = get_scan_positions(path)[0]
    assert (scan.x_motor, scan.y_motor) == ("piy", "piz")
    np.testing.assert_array_equal(scan.x, [1, 2, 3])
    np.testing.assert_array_equal(scan.y, [4, 4, 5])


@pytest.mark.parametrize("command", ["amesh", "asxdm", "dsxdm"])
def test_aliases_do_not_broadcast_a_moving_motor_snapshot(tmp_path, command):
    path = tmp_path / "alias-snapshots.h5"
    with h5py.File(path, "w") as f:
        add_scan(
            f, "1.1", f"{command} piy 0 100 10 piz 0 200 10 0.1", [1, 2, 3], static_y=4
        )
    with pytest.warns(UserWarning, match="static snapshot"):
        assert get_scan_positions(path) == []


def test_nonspatial_scan_is_a_point_in_selected_motor_axes(tmp_path):
    path = tmp_path / "point.h5"
    with h5py.File(path, "w") as f:
        scan = add_scan(f, "1.1", "ct 0.1", static_y=4)
        scan["instrument/positioners/piy"] = 3
    positions = get_scan_positions(path, x_motor="piy", y_motor="piz")
    assert len(positions) == 1
    np.testing.assert_array_equal(positions[0].x, [3])
    np.testing.assert_array_equal(positions[0].y, [4])


def test_explicit_axes_and_scan_selection(scans):
    result = get_scan_positions(scans, scan_nos="2.1", x_motor="piy", y_motor="piz")
    assert len(result) == 1
    assert result[0].scan_no == "2.1"
    with pytest.raises(ValueError, match="infer"):
        get_scan_positions(scans, scan_nos=["2.1"])
    with pytest.raises(ValueError, match="both"):
        get_scan_positions(scans, x_motor="piy")
    with pytest.raises(ValueError, match="distinct"):
        get_scan_positions(scans, x_motor="piy", y_motor="piy")
    with pytest.raises(KeyError, match="not found"):
        get_scan_positions(scans, scan_nos=["99.1"])


def test_interrupted_scan_keeps_shared_prefix_and_missing_gaps(tmp_path):
    path = tmp_path / "partial.h5"
    with h5py.File(path, "w") as f:
        add_scan(f, "1.1", "mesh piy 0 4 5 piz 0 4 5 0.1", [0, 1, np.nan, 3], [2, 2, 2])
    with pytest.warns(UserWarning, match="shared acquired prefix"):
        scan = get_scan_positions(path)[0]
    np.testing.assert_array_equal(scan.point_index, [0, 1, 2])
    np.testing.assert_array_equal(scan.x, [0, 1, np.nan])


def test_single_acquired_point_is_not_a_static_position(tmp_path):
    path = tmp_path / "one-point.h5"
    with h5py.File(path, "w") as f:
        add_scan(f, "1.1", "mesh piy 0 4 5 piz 0 4 5 0.1", [0], [1, 2, 3])
    with pytest.warns(UserWarning, match="shared acquired prefix"):
        scan = get_scan_positions(path)[0]
    np.testing.assert_array_equal(scan.x, [0])
    np.testing.assert_array_equal(scan.y, [1])


def test_acquired_positioner_array_takes_precedence_over_initial_measurement(tmp_path):
    path = tmp_path / "positioners.h5"
    with h5py.File(path, "w") as f:
        scan = add_scan(f, "1.1", "mesh piy 0 4 5 piz 0 4 5 0.1", [0], [0, 1, 2])
        scan["instrument/positioners/piy_position"] = [3, 4, 5]
    scan = get_scan_positions(path)[0]
    np.testing.assert_array_equal(scan.x, [3, 4, 5])


def test_compressed_motor_channels_register_filters(tmp_path):
    import hdf5plugin

    path = tmp_path / "compressed.h5"
    with h5py.File(path, "w") as f:
        scan = add_scan(f, "1.1", "mesh piy 0 4 5 piz 0 4 5 0.1", y=[0, 1, 2])
        scan.create_dataset(
            "measurement/piy", data=[3, 4, 5], **hdf5plugin.Bitshuffle()
        )
    np.testing.assert_array_equal(get_scan_positions(path)[0].x, [3, 4, 5])


def test_static_snapshots_of_moving_motors_are_not_trajectories(tmp_path):
    path = tmp_path / "snapshots.h5"
    with h5py.File(path, "w") as f:
        s = add_scan(f, "1.1", "mesh piy 0 4 5 piz 0 4 5 0.1", static_y=4)
        s["instrument/positioners/piy"] = 3
    with pytest.warns(UserWarning, match="static snapshot"):
        assert get_scan_positions(path) == []


def test_missing_non_numeric_or_image_motor_channels_skip_without_read(tmp_path):
    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as f:
        add_scan(f, "1.1", "mesh piy 0 4 5 piz 0 4 5 0.1", np.zeros((2, 2, 2)), [0, 1])
        add_scan(f, "2.1", "ct 1", [1], [np.nan])
        f["3.1"] = h5py.ExternalLink("does-not-exist.h5", "/entry")
    with pytest.warns(UserWarning) as recorded:
        assert get_scan_positions(path) == []
    messages = " ".join(str(w.message) for w in recorded)
    assert "inaccessible" in messages
    assert "missing coordinates" in messages
    assert "no finite" in messages


def test_empty_and_bytes(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass
    assert get_scan_positions(path) == []
    with h5py.File(path, "a") as f:
        add_scan(f, "1.1", np.bytes_("mesh piy 0 1 1 piz 0 1 1 0.1"), [0], [1])
    assert isinstance(get_scan_positions(path)[0].title, str)


def test_unicode_motor_names_and_scan_titles(tmp_path):
    path = tmp_path / "unicode.h5"
    with h5py.File(path, "w") as f:
        f["1.1/title"] = "mesh πy 0 1 2 πz 0 1 2 0.1 # échantillon"
        f["1.1/measurement/πy"] = [1, 2]
        f["1.1/measurement/πz"] = [3, 4]
        f["1.1/measurement/πy"].attrs["units"] = "µm"
    scan = get_scan_positions(path)[0]
    assert scan.x_motor == "πy"
    assert scan.x_unit == "µm"
    assert "échantillon" in scan.title


def test_missing_external_motor_channel_is_skipped(tmp_path):
    path = tmp_path / "broken-motor.h5"
    with h5py.File(path, "w") as f:
        scan = add_scan(f, "1.1", "mesh piy 0 4 5 piz 0 4 5 0.1", y=[0, 1])
        scan["measurement/piy"] = h5py.ExternalLink("missing.h5", "/positions")
    with pytest.warns(UserWarning, match="coordinates"):
        assert get_scan_positions(path) == []


@pytest.fixture
def mixed_scan_types(tmp_path):
    path = tmp_path / "mixed-types.h5"
    commands = [
        "sxdm",
        "asxdm",
        "dsxdm",
        "kmap",
        "mesh",
        "amesh",
        "dmesh",
        "ascan",
        "dscan",
        "ct",
        "a2scan",
        "fscan",
        "sxdm_extra",
    ]
    with h5py.File(path, "w") as f:
        for number, command in enumerate(commands, 1):
            if command in {"ascan", "dscan"}:
                title = f"{command} piy 0 1 2 0.1"
            elif command == "ct":
                title = "ct 0.1 # mesh sxdm ascan are comments, not commands"
            elif command == "a2scan":
                title = "a2scan piy 0 1 piz 0 1 2 0.1"
            else:
                title = f"{command}(piy, 0, 1, 2, piz, 0, 1, 2, 0.1)"
            add_scan(f, f"{number}.1", title, [0, 1], [2, 3])
    return path


@pytest.mark.parametrize(
    "scan_type, numbers",
    [
        ("sxdm", [1, 2, 3, 4]),
        ("mesh", [5, 6, 7]),
        ("ascan", [8, 9]),
        (" ASXDM ", [2]),
        (["mesh", "AScan", "asxdm"], [2, 5, 6, 7, 8, 9]),
        (["ct", "a2scan", "fscan"], [10, 11, 12]),
    ],
)
def test_scan_type_families_aliases_and_unions(mixed_scan_types, scan_type, numbers):
    scans = get_scan_positions(mixed_scan_types, scan_type=scan_type)
    assert [scan.scan_no for scan in scans] == [f"{number}.1" for number in numbers]
    assert all((scan.x_motor, scan.y_motor) == ("piy", "piz") for scan in scans)


def test_scan_type_intersects_scan_numbers_and_validates_missing_ids(mixed_scan_types):
    scans = get_scan_positions(
        mixed_scan_types, scan_type="mesh", scan_nos=[2, 5, 6, 8]
    )
    assert [scan.scan_no for scan in scans] == ["5.1", "6.1"]
    assert get_scan_positions(mixed_scan_types, scan_type="mesh", scan_nos=[2]) == []
    with pytest.raises(KeyError, match="not found"):
        get_scan_positions(mixed_scan_types, scan_type=[], scan_nos=[99])


def test_empty_type_selection_has_no_axis_inference_error(tmp_path):
    path = tmp_path / "no-matching-types.h5"
    with h5py.File(path, "w") as f:
        add_scan(f, "1.1", "ct 1", [0], [1])
    assert get_scan_positions(path, scan_type=[]) == []
    assert get_scan_positions(path, scan_type="mesh") == []


@pytest.mark.parametrize(
    "selection", ["mseh", "", 1, b"mesh", {"mesh": True}, ["mesh", 1]]
)
def test_invalid_scan_type_fails_before_file_read(selection):
    with pytest.raises(ValueError, match="scan_type"):
        get_scan_positions("missing.h5", scan_type=selection)


def test_selected_2d_types_infer_their_own_axes_first(tmp_path):
    path = tmp_path / "different-motors.h5"
    with h5py.File(path, "w") as f:
        f["1.1/title"] = "sxdm samx 0 1 2 samy 0 1 2 0.1"
        add_scan(f, "2.1", "mesh piy 0 1 2 piz 0 1 2 0.1", [0, 1], [2, 3])
    selected = get_scan_positions(path, scan_type="mesh")
    assert len(selected) == 1
    assert (selected[0].x_motor, selected[0].y_motor) == ("piy", "piz")


def test_scan_type_filter_runs_before_reading_motor_channels(
    mixed_scan_types, monkeypatch
):
    from sxdm.io import scan_positions

    original = scan_positions._read_motor
    visited = []

    def read_selected_only(scan, motor):
        visited.append(scan.name)
        assert scan.name in {"/8.1", "/9.1"}
        return original(scan, motor)

    monkeypatch.setattr(scan_positions, "_read_motor", read_selected_only)
    selected = get_scan_positions(mixed_scan_types, scan_type="ascan")
    assert len(selected) == 2
    assert visited == ["/8.1", "/8.1", "/9.1", "/9.1"]
