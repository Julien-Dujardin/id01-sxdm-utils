"""Minimal marimo notebook: display BLISS scans on a normalized rectangle."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import sxdm

    return Path, mo, sxdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Display scan positions

    Display measured scan lines and points on a shared normalized rectangle.
    Hover to see the scan name, command, and original motor coordinates.

    Supported spatial commands include `ascan`, `amesh`, and `asxdm`.
    Positions are read from the file; scan endpoints are not used to invent
    missing measurements. For a dataset containing only line/point scans,
    supply the two motor names below.
    """)
    return


@app.cell
def _(mo):
    dataset = mo.ui.text(
        value=str(mo.cli_args().get("dataset", "")),
        placeholder="C:/Users/.../S1_AM16_0001.h5",
        label="BLISS dataset path",
        full_width=True,
    )
    dataset
    return (dataset,)


@app.cell
def _(Path, dataset, mo):
    mo.stop(not dataset.value, mo.md("Enter the path to your BLISS dataset above."))
    path_dset = Path(dataset.value).expanduser()
    mo.stop(not path_dset.is_file(), mo.md("The dataset path is not a file."))
    return (path_dset,)


@app.cell
def _(mo):
    scan_type_choice = mo.ui.dropdown(
        ["all", "sxdm", "mesh", "ascan"], value="all", label="Scan type"
    )
    scan_type_choice
    return (scan_type_choice,)


@app.cell
def _(mo):
    # Or replace None with the motor names, e.g. "samx" and "samy".
    x_motor = mo.cli_args().get("x-motor", None)
    y_motor = mo.cli_args().get("y-motor", None)
    rectangle = (0, 1, 0, 1)  # display xmin, xmax, ymin, ymax
    # Optional native bounds keep the same mapping across different datasets.
    motor_bounds = None
    return motor_bounds, rectangle, x_motor, y_motor


@app.cell
def _(mo, motor_bounds, path_dset, sxdm, x_motor, y_motor):
    import numpy as _np

    # Infer the frame ONCE from all scans so changing type only filters points.
    _positions = sxdm.io.get_scan_positions(path_dset, x_motor=x_motor, y_motor=y_motor)
    mo.stop(not _positions, mo.md("No usable scan positions in this coordinate frame."))
    _units = (_positions[0].x_unit, _positions[0].y_unit)
    mo.stop(
        any((_scan.x_unit, _scan.y_unit) != _units for _scan in _positions),
        mo.callout(
            "Motor units differ between scans. A shared reference rectangle requires "
            "matching units; convert the coordinates before comparing scan types.",
            kind="warn",
        ),
    )
    reference_axes = {
        "x_motor": _positions[0].x_motor,
        "y_motor": _positions[0].y_motor,
    }
    reference_bounds = motor_bounds
    if reference_bounds is None:
        _limits = []
        for _scan in _positions:
            _valid = _np.isfinite(_scan.x) & _np.isfinite(_scan.y)
            _limits.append(
                (
                    _scan.x[_valid].min(),
                    _scan.x[_valid].max(),
                    _scan.y[_valid].min(),
                    _scan.y[_valid].max(),
                )
            )
        _limits = _np.asarray(_limits)
        reference_bounds = (
            float(_limits[:, 0].min()),
            float(_limits[:, 1].max()),
            float(_limits[:, 2].min()),
            float(_limits[:, 3].max()),
        )
        # Explicit bounds require positive widths; center a stationary axis in
        # a small reference interval so point/line-only files still work.
        _bounds = list(reference_bounds)
        for _axis in (0, 2):
            if _bounds[_axis] == _bounds[_axis + 1]:
                _bounds[_axis] -= 0.5
                _bounds[_axis + 1] += 0.5
        reference_bounds = tuple(_bounds)
    return reference_axes, reference_bounds


@app.cell
def _(mo, path_dset, scan_type_choice, sxdm):
    # The table and map use the same selection, without ipywidgets.
    mo.Html(
        sxdm.io.bliss.get_scan_table(
            path_dset,
            as_widget=False,
            scan_type=None
            if scan_type_choice.value == "all"
            else scan_type_choice.value,
        )
    )
    return


@app.cell
def _(
    mo, path_dset, rectangle, reference_axes, reference_bounds, scan_type_choice, sxdm
):
    figure = sxdm.io.bliss.display_scan_positions(
        path_dset,
        **reference_axes,
        rectangle=rectangle,
        motor_bounds=reference_bounds,
        scan_type=None if scan_type_choice.value == "all" else scan_type_choice.value,
        max_total_points=10_000,  # shared display budget, not per scan
    )
    mo.ui.plotly(figure)
    return (figure,)


if __name__ == "__main__":
    app.run()
