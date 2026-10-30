"""Browser-native interactive plots for marimo, notebooks and HTML export.

The functions return ordinary Plotly figures, with no Jupyter widget backend.
In marimo, display a figure directly or wrap it in ``mo.ui.plotly(fig)`` to
receive point selections in Python. Hover, pan, zoom and legend toggles work
directly in the browser. ``fig.write_html('scans.html')`` exports a standalone
interactive page, including Plotly JavaScript by default.
"""

from __future__ import annotations

from html import escape
import warnings

import h5py
import numpy as np

from ..io.scan_positions import get_scan_positions

__all__ = ["get_scan_map", "get_counter_map", "display_scan_positions"]


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as err:
        raise ImportError(
            "Interactive plots require Plotly. Install with "
            "`uv pip install 'id01-sxdm[marimo]'` or `uv sync --extra marimo`."
        ) from err
    return go


def _sample_indexes(length, max_points):
    if max_points is not None and (
        isinstance(max_points, bool)
        or not isinstance(max_points, int)
        or max_points < 2
    ):
        raise ValueError("max_points must be an integer >= 2 or None.")
    if max_points is None or length <= max_points:
        return np.arange(length)
    return np.linspace(0, length - 1, max_points, dtype=int)


def _label(motor, unit):
    return f"{motor} ({unit})" if unit else motor


def _layout(fig, positions, title):
    x_label, y_label = "x", "y"
    if positions:
        first = positions[0]
        x_label = _label(first.x_motor, first.x_unit)
        y_label = _label(first.y_motor, first.y_unit)
        # Different native units must never share the same numerical axis.
        if any((p.x_unit, p.y_unit) != (first.x_unit, first.y_unit) for p in positions):
            raise ValueError(
                "Motor units differ between scans. Select scans with matching units "
                "or convert coordinates explicitly before plotting."
            )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="closest",
        legend_title_text="Scan",
        uirevision="sxdm-scan-map",
    )
    if positions and positions[0].x_unit == positions[0].y_unit:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def get_scan_map(
    path_dset,
    *,
    x_motor=None,
    y_motor=None,
    scan_nos=None,
    max_points=5000,
    title="Scan positions",
):
    """Plot scan trajectories as lines and acquired points, with names on hover.

    This is a spatial alternative to ``sxdm.io.bliss.get_scan_table``. Each scan
    is a separate WebGL trace; click its legend entry to toggle it. The hover
    label includes the scan number, full command, start time and acquisition
    point index. Positions use native motor units, not requested command bounds.
    See :func:`sxdm.io.scan_positions.get_scan_positions` for axis inference.

    ``max_points`` limits displayed positions per scan (including endpoints),
    with additional missing-position markers retained to break lines at gaps.
    Set it to ``None`` to retain all points. Sampling affects display only, and
    is recorded in each trace's metadata. Nonfinite gaps break lines.
    """
    _sample_indexes(0, max_points)  # Validate even for an empty file.
    positions = get_scan_positions(
        path_dset, x_motor=x_motor, y_motor=y_motor, scan_nos=scan_nos
    )
    return _scan_figure(positions, max_points=max_points, title=title)


def _scan_figure(positions, *, max_points, title):
    go = _plotly()
    fig = go.Figure()
    for scan in positions:
        indexes = _sample_indexes(scan.x.size, max_points)
        # Retain missing positions as gaps even when thinning a large trajectory.
        if indexes.size > 1:
            missing = ~(np.isfinite(scan.x) & np.isfinite(scan.y))
            starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
            indexes = np.union1d(indexes, starts)
        count = indexes.size
        scan_name = escape(scan.scan_no)
        customdata = [
            [scan.scan_no, scan.title, scan.start_time, int(i)]
            for i in scan.point_index[indexes]
        ]
        fig.add_trace(
            go.Scattergl(
                x=scan.x[indexes],
                y=scan.y[indexes],
                mode="lines+markers",
                name=scan.scan_no,
                customdata=customdata,
                marker={"size": 6},
                line={"width": 1},
                connectgaps=False,
                hovertemplate=(
                    f"<b>Scan {scan_name}</b><br>"
                    + escape(scan.title)
                    + "<br>"
                    + escape(scan.start_time)
                    + "<br>Point %{customdata[3]}"
                    + f"<br>{escape(scan.x_motor)}: %{{x}}"
                    + f"<br>{escape(scan.y_motor)}: %{{y}}<extra></extra>"
                ),
                meta={
                    "scan_no": scan.scan_no,
                    "acquired_points": scan.x.size,
                    "displayed_points": count,
                },
            )
        )
    if not positions:
        fig.add_annotation(text="No scans with usable positions", showarrow=False)
    return _layout(fig, positions, title)


def _validate_bounds(bounds, name):
    from numbers import Real

    try:
        values = tuple(bounds)
    except TypeError as exc:
        raise ValueError(f"{name} must be (xmin, xmax, ymin, ymax).") from exc
    if len(values) != 4 or any(
        isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v)
        for v in values
    ):
        raise ValueError(f"{name} must contain four finite numeric bounds.")
    xmin, xmax, ymin, ymax = map(float, values)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"{name} requires xmin < xmax and ymin < ymax.")
    return xmin, xmax, ymin, ymax


def display_scan_positions(
    path_dset,
    *,
    x_motor=None,
    y_motor=None,
    rectangle=(0, 1, 0, 1),
    motor_bounds=None,
    scan_nos=None,
    max_points=5000,
    title="Scan positions on a normalized rectangle",
):
    """Return a marimo-compatible figure mapping scans onto one shared rectangle.

    Use this directly as the final expression of a notebook cell, or wrap the
    result in ``mo.ui.plotly``. No marimo or ipywidgets imports are required here.

    ``rectangle=(xmin, xmax, ymin, ymax)`` defines display coordinates, defaulting
    to a unit square. ``motor_bounds`` uses the same order in native motor units.
    When omitted, native bounds are inferred from ALL finite acquired coordinate
    pairs of the selected scans before display sampling. All scans use the SAME
    affine transform; individual scans are never independently stretched to fit.
    An inferred constant axis is centered in the rectangle. Explicit bounds must
    have nonzero width and height; points outside them are shown without clipping.

    Hover labels include scan number, command, timestamp, acquisition index,
    original motor coordinates and mapped coordinates. To compare subsets with
    the same mapping, reuse explicit ``motor_bounds``. Normalization rescales
    axes independently and is not a physical coordinate-frame transformation.
    Only recorded positions are drawn; missing trajectories are not reconstructed
    from requested scan-command endpoints. See ``get_scan_positions`` for the
    supported motor-channel layout and axis inference.
    """
    rectangle = _validate_bounds(rectangle, "rectangle")
    if motor_bounds is not None:
        motor_bounds = _validate_bounds(motor_bounds, "motor_bounds")
    _sample_indexes(0, max_points)
    positions = get_scan_positions(
        path_dset, x_motor=x_motor, y_motor=y_motor, scan_nos=scan_nos
    )
    fig = _scan_figure(positions, max_points=max_points, title=title)
    if positions and motor_bounds is None:
        limits = []
        for scan in positions:
            valid = np.isfinite(scan.x) & np.isfinite(scan.y)
            limits.append(
                (
                    scan.x[valid].min(),
                    scan.x[valid].max(),
                    scan.y[valid].min(),
                    scan.y[valid].max(),
                )
            )
        limits = np.asarray(limits)
        motor_bounds = (
            float(limits[:, 0].min()),
            float(limits[:, 1].max()),
            float(limits[:, 2].min()),
            float(limits[:, 3].max()),
        )

    def mapped(values, axis):
        lo, hi = motor_bounds[axis : axis + 2]
        start, stop = rectangle[axis : axis + 2]
        values = np.asarray(values, dtype=float)
        if hi == lo:
            return np.where(np.isfinite(values), (start + stop) / 2, np.nan)
        return start + (values - lo) * ((stop - start) / (hi - lo))

    for trace, scan in zip(fig.data, positions):
        raw_x, raw_y = np.asarray(trace.x), np.asarray(trace.y)
        trace.customdata = [
            list(row) + [float(x), float(y)]
            for row, x, y in zip(trace.customdata, raw_x, raw_y)
        ]
        trace.x, trace.y = mapped(raw_x, 0), mapped(raw_y, 2)
        trace.hovertemplate = (
            f"<b>Scan {escape(scan.scan_no)}</b><br>"
            + escape(scan.title)
            + "<br>"
            + escape(scan.start_time)
            + "<br>Point %{customdata[3]}"
            + f"<br>{escape(_label(scan.x_motor, scan.x_unit))}: %{{customdata[4]}}"
            + f"<br>{escape(_label(scan.y_motor, scan.y_unit))}: %{{customdata[5]}}"
            + "<br>Mapped X: %{x}<br>Mapped Y: %{y}<extra></extra>"
        )
    xmin, xmax, ymin, ymax = rectangle
    fig.add_shape(
        type="rect",
        x0=xmin,
        x1=xmax,
        y0=ymin,
        y1=ymax,
        xref="x",
        yref="y",
        layer="below",
        line={"color": "#64748b", "width": 2},
        fillcolor="rgba(148, 163, 184, 0.08)",
    )
    x_name = positions[0].x_motor if positions else x_motor
    y_name = positions[0].y_motor if positions else y_motor
    fig.update_layout(
        xaxis_title=f"Mapped X ({x_name})" if x_name else "Mapped X",
        yaxis_title=f"Mapped Y ({y_name})" if y_name else "Mapped Y",
        meta={
            "coordinate_system": "normalized",
            "rectangle": rectangle,
            "motor_bounds": motor_bounds,
            "x_motor": x_name,
            "y_motor": y_name,
        },
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def get_counter_map(
    path_dset,
    scan_no,
    counter,
    *,
    x_motor=None,
    y_motor=None,
    max_points=50000,
    log=False,
    colorscale="Viridis",
):
    """Color acquired sample positions by one counter, without detector images.

    A scatter map preserves irregular, snake and interrupted scans without
    inventing a rectangular grid or filling missing samples with zeros. Hover
    shows the original counter value. ``log=True`` colors by log10(value), hiding
    nonpositive values; it does not alter the original values in hover data.
    The selected counter must be a scalar/1D/2D numeric measurement channel.
    """
    go = _plotly()
    _sample_indexes(0, max_points)
    scan_no = f"{scan_no}.1" if isinstance(scan_no, int) else str(scan_no)
    positions = get_scan_positions(
        path_dset, x_motor=x_motor, y_motor=y_motor, scan_nos=[scan_no]
    )
    if not positions:
        raise ValueError(f"Scan {scan_no} has no usable motor positions.")
    scan = positions[0]
    with h5py.File(path_dset, "r") as h5f:
        key = f"{scan_no}/measurement/{counter}"
        node = h5f[key]
        if (
            not isinstance(node, h5py.Dataset)
            or node.ndim > 2
            or node.dtype.kind not in "iuf"
        ):
            raise ValueError(
                "Counter must be a scalar/1D/2D numeric measurement channel."
            )
        values = np.asarray(node[()], dtype=float).reshape(-1)
    if values.size != scan.x.size:
        warnings.warn(
            f"Scan {scan_no}: counter and position lengths differ; using their shared acquired prefix.",
            UserWarning,
            stacklevel=2,
        )
    length = min(values.size, scan.x.size)
    values = values[:length]
    x, y = scan.x[:length], scan.y[:length]
    indexes = np.arange(length)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if log:
        valid &= values > 0
    x, y, values, indexes = x[valid], y[valid], values[valid], indexes[valid]
    if not values.size:
        raise ValueError(
            "No finite counter samples to plot (log colors require positive values)."
        )
    sampled = _sample_indexes(len(values), max_points)
    x, y, values, indexes = x[sampled], y[sampled], values[sampled], indexes[sampled]
    color = np.log10(values) if log else values
    fig = go.Figure(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            name=str(scan_no),
            customdata=np.column_stack([indexes, values]),
            marker={
                "size": 7,
                "color": color,
                "colorscale": colorscale,
                "showscale": True,
                "colorbar": {"title": f"log10({counter})" if log else str(counter)},
            },
            hovertemplate=(
                f"<b>Scan {escape(str(scan_no))}</b><br>"
                + escape(scan.title)
                + f"<br>{escape(str(counter))}: %{{customdata[1]}}"
                + "<br>Point %{customdata[0]}"
                + f"<br>{escape(scan.x_motor)}: %{{x}}"
                + f"<br>{escape(scan.y_motor)}: %{{y}}<extra></extra>"
            ),
            meta={
                "scan_no": str(scan_no),
                "acquired_points": length,
                "displayed_points": len(values),
            },
        )
    )
    return _layout(fig, positions, f"Scan {scan_no} · {counter}")
