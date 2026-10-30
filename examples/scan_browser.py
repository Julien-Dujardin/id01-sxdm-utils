"""Browse BLISS scans with marimo; see README for launch and demo commands."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import h5py
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from sxdm.io.bliss import get_detector_aliases, get_detector_frame, get_sxdm_pos_sum
    from sxdm.io.scan_positions import get_scan_positions
    from sxdm.plot import get_scan_map, get_counter_map

    return (
        Path,
        get_counter_map,
        get_detector_aliases,
        get_detector_frame,
        get_scan_map,
        get_scan_positions,
        get_sxdm_pos_sum,
        go,
        h5py,
        mo,
        np,
    )


@app.cell
def _(mo):
    mo.md("""
    # SXDM scan browser
    Open a BLISS master HDF5 file. Hover over a point to see its scan and command;
    zoom, pan or toggle scans in the legend. Select a scan below to explore its
    counters and detector frames. Positions use the file's native motor units.
    """)
    return


@app.cell
def _(mo):
    dataset_input = mo.ui.text(
        value=str(mo.cli_args().get("dataset", "")), label="BLISS file", full_width=True
    )
    x_input = mo.ui.text(
        value=str(mo.cli_args().get("x-motor", "")), label="X motor (optional)"
    )
    y_input = mo.ui.text(
        value=str(mo.cli_args().get("y-motor", "")), label="Y motor (optional)"
    )
    refresh = mo.ui.run_button(label="Refresh file")
    mo.vstack([dataset_input, mo.hstack([x_input, y_input, refresh])])
    return dataset_input, refresh, x_input, y_input


@app.cell
def _(
    Path, dataset_input, get_scan_map, get_scan_positions, mo, refresh, x_input, y_input
):
    refresh.value
    mo.stop(
        not dataset_input.value,
        mo.md("Enter a dataset path, or launch with `-- --dataset /path/to/data.h5`."),
    )
    path = Path(dataset_input.value).expanduser()
    mo.stop(not path.is_file(), mo.md("The dataset path is not a file."))
    axes = {"x_motor": x_input.value or None, "y_motor": y_input.value or None}
    error = ""
    positions = []
    overview = None
    try:
        positions = get_scan_positions(path, **axes)
        overview = get_scan_map(path, **axes)
    except (OSError, ValueError, KeyError) as exc:
        error = str(exc)
    mo.stop(bool(error), mo.callout(mo.plain_text(error), kind="warn"))
    mo.stop(
        not positions,
        mo.md("No acquired scan positions found. Try explicit motor names."),
    )
    axes = {"x_motor": positions[0].x_motor, "y_motor": positions[0].y_motor}
    mo.ui.plotly(overview)
    return axes, path, positions


@app.cell
def _(mo, positions):
    scan_choice = mo.ui.dropdown(
        [p.scan_no for p in positions],
        value=positions[0].scan_no,
        label="Scan",
        searchable=True,
    )
    scan_choice
    return (scan_choice,)


@app.cell
def _(axes, get_counter_map, h5py, mo, np, path, refresh, scan_choice):
    refresh.value
    with h5py.File(path, "r") as _h5:
        _group = _h5.get(f"{scan_choice.value}/measurement")
        _counters = (
            []
            if _group is None
            else [
                _name
                for _name in _group
                if isinstance(_group.get(_name), h5py.Dataset)
                and _group[_name].ndim <= 2
                and _group[_name].dtype.kind in "iuf"
            ]
        )
    counter_choice = mo.ui.dropdown(
        _counters,
        value=_counters[0] if _counters else None,
        label="Counter",
        searchable=True,
    )
    log_counter = mo.ui.checkbox(label="Log counter colors")
    mo.hstack([counter_choice, log_counter])
    return counter_choice, log_counter


@app.cell
def _(
    axes, counter_choice, get_counter_map, log_counter, mo, path, refresh, scan_choice
):
    refresh.value
    mo.stop(
        counter_choice.value is None, mo.md("No scalar counter channels in this scan.")
    )
    try:
        _counter_figure = get_counter_map(
            path, scan_choice.value, counter_choice.value, log=log_counter.value, **axes
        )
        _counter_display = mo.ui.plotly(_counter_figure)
    except (ValueError, KeyError, OSError) as _exc:
        _counter_display = mo.callout(mo.plain_text(str(_exc)), kind="warn")
    _counter_display
    return


@app.cell
def _(get_detector_aliases, h5py, mo, path, refresh, scan_choice):
    refresh.value
    try:
        _detectors = get_detector_aliases(path, scan_choice.value)
        with h5py.File(path, "r") as _h5:
            _detectors = [
                _name
                for _name in _detectors
                if isinstance(
                    _h5.get(f"{scan_choice.value}/instrument/{_name}/data"),
                    h5py.Dataset,
                )
                and _h5[f"{scan_choice.value}/instrument/{_name}/data"].ndim == 3
            ]
    except (KeyError, ValueError):
        _detectors = []
    detector_choice = mo.ui.dropdown(
        _detectors, value=_detectors[0] if _detectors else None, label="Detector"
    )
    mo.md("## Detector inspection")
    detector_choice
    return (detector_choice,)


@app.cell
def _(detector_choice, h5py, mo, path, refresh, scan_choice):
    refresh.value
    mo.stop(detector_choice.value is None, mo.md("No detector stack available."))
    with h5py.File(path, "r") as _h5:
        _shape = _h5[
            f"{scan_choice.value}/instrument/{detector_choice.value}/data"
        ].shape
    mo.stop(len(_shape) != 3 or _shape[0] == 0, mo.md("No acquired detector frames."))
    frame_choice = mo.ui.slider(
        0,
        _shape[0] - 1,
        value=0,
        debounce=True,
        label="Acquisition point",
        show_value=True,
    )
    integrate = mo.ui.run_button(label="Integrate detector over all points")
    mo.hstack([frame_choice, integrate])
    return frame_choice, integrate


@app.cell
def _(
    detector_choice,
    frame_choice,
    get_detector_frame,
    go,
    mo,
    np,
    path,
    refresh,
    scan_choice,
):
    refresh.value
    _frame = get_detector_frame(
        path, scan_choice.value, frame_choice.value, detector=detector_choice.value
    )
    _fig = go.Figure(
        go.Heatmap(
            z=np.log1p(_frame),
            colorscale="Inferno",
            colorbar={"title": "log(1 + counts)"},
        )
    )
    _fig.update_layout(
        title=f"Scan {scan_choice.value} · point {frame_choice.value}",
        xaxis_title="Detector column",
        yaxis_title="Detector row",
    )
    _fig.update_yaxes(scaleanchor="x", autorange="reversed")
    mo.ui.plotly(_fig)
    return


@app.cell
def _(
    axes,
    detector_choice,
    get_scan_positions,
    get_sxdm_pos_sum,
    go,
    integrate,
    mo,
    path,
    refresh,
    scan_choice,
):
    refresh.value
    mo.stop(not integrate.value)
    _values = get_sxdm_pos_sum(
        path, scan_choice.value, detector=detector_choice.value, pbar=False
    )
    _scan = get_scan_positions(path, scan_nos=[scan_choice.value], **axes)[0]
    _count = min(_scan.x.size, _values.size)
    _fig = go.Figure(
        go.Scattergl(
            x=_scan.x[:_count],
            y=_scan.y[:_count],
            mode="markers",
            marker={
                "color": _values[:_count],
                "showscale": True,
                "colorbar": {"title": "Integrated counts"},
            },
        )
    )
    _fig.update_layout(
        title="Detector-integrated sample map",
        xaxis_title=_scan.x_motor,
        yaxis_title=_scan.y_motor,
    )
    mo.ui.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()
