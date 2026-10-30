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
    # Or replace None with the motor names, e.g. "samx" and "samy".
    x_motor = mo.cli_args().get("x-motor", None)
    y_motor = mo.cli_args().get("y-motor", None)
    rectangle = (0, 1, 0, 1)  # display xmin, xmax, ymin, ymax
    # Optional native bounds keep the same mapping across different datasets.
    motor_bounds = None
    return motor_bounds, rectangle, x_motor, y_motor


@app.cell
def _(mo, path_dset, sxdm):
    # A normal HTML table in marimo; no ipywidgets dependency.
    mo.Html(sxdm.io.bliss.get_scan_table(path_dset, as_widget=False))
    return


@app.cell
def _(mo, motor_bounds, path_dset, rectangle, sxdm, x_motor, y_motor):
    figure = sxdm.io.bliss.display_scan_positions(
        path_dset,
        x_motor=x_motor,
        y_motor=y_motor,
        rectangle=rectangle,
        motor_bounds=motor_bounds,
    )
    mo.ui.plotly(figure)
    return (figure,)


if __name__ == "__main__":
    app.run()
