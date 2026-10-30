"""Read scan trajectories without loading detector images or notebook backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import h5py
import hdf5plugin  # noqa: F401 - register BLISS compression filters before reads
import numpy as np

__all__ = ["ScanPositions", "get_scan_positions"]

_SCAN_TYPE_FAMILIES = {
    "sxdm": frozenset({"sxdm", "asxdm", "dsxdm", "kmap"}),
    "mesh": frozenset({"mesh", "amesh", "dmesh"}),
    "ascan": frozenset({"ascan", "dscan"}),
}
_EXACT_SCAN_COMMANDS = frozenset().union(
    *_SCAN_TYPE_FAMILIES.values(),
    {"ct", "fscan", "a2scan", "d2scan", "a3scan", "d3scan", "loopscan", "timescan"},
)


@dataclass(frozen=True)
class ScanPositions:
    """Acquired coordinates for one scan, in acquisition order.

    ``point_index`` is the zero-based acquisition index. Nonfinite coordinates
    remain in the arrays so plotting does not connect across missing positions.
    All arrays are detached from the file, which is closed before returning.
    """

    scan_no: str
    title: str
    start_time: str
    x_motor: str
    y_motor: str
    x: np.ndarray
    y: np.ndarray
    point_index: np.ndarray
    x_unit: str = ""
    y_unit: str = ""


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _scalar_text(group, name, default=""):
    node = group.get(name)
    if isinstance(node, h5py.Dataset) and node.shape == ():
        return _text(node[()])
    return default


def _scan_key(name):
    return tuple(int(part) if part.isdigit() else part for part in name.split("."))


def _scan_entries(h5f):
    # BLISS entries use scan.subscan names; ignore metadata groups and broken links.
    entries = []
    for name in h5f:
        if not re.fullmatch(r"\d+(?:\.\d+)?", name):
            continue
        try:
            if isinstance(h5f[name], h5py.Group):
                entries.append(name)
        except (KeyError, OSError):
            warnings.warn(
                f"Skipping inaccessible scan {name}.", UserWarning, stacklevel=3
            )
    return sorted(entries, key=_scan_key)


def _command_tokens(title):
    return re.sub(r"[(),]", " ", title).split()


def _scan_command(title):
    """Return the exact command token, independently of its scan parameters."""
    tokens = _command_tokens(title)
    return tokens[0].lower() if tokens else ""


def _scan_type_commands(scan_type):
    """Validate a scan-family/exact-command selection and expand its union."""
    if scan_type is None:
        return None
    if isinstance(scan_type, str):
        selected = [scan_type]
    elif isinstance(scan_type, Sequence) and not isinstance(
        scan_type, (bytes, bytearray)
    ):
        selected = scan_type
    else:
        raise ValueError(
            "scan_type must be a command/family string, a sequence of strings, or None."
        )
    commands = set()
    for value in selected:
        if not isinstance(value, str):
            raise ValueError("scan_type entries must be command/family strings.")
        value = value.strip().lower()
        if value in _SCAN_TYPE_FAMILIES:
            commands.update(_SCAN_TYPE_FAMILIES[value])
        elif value in _EXACT_SCAN_COMMANDS:
            commands.add(value)
        else:
            supported = ", ".join(sorted(_EXACT_SCAN_COMMANDS))
            raise ValueError(
                f"Unknown scan_type {value!r}. Use a family (sxdm, mesh, ascan) "
                f"or an exact supported command: {supported}."
            )
    return frozenset(commands)


def _command_motors(title):
    """Recognize motor names, never derive trajectories from requested bounds."""
    tokens = _command_tokens(title)
    if not tokens:
        return ()
    command = tokens[0].lower()
    if (
        command in {"sxdm", "asxdm", "dsxdm", "mesh", "amesh", "dmesh", "kmap"}
        and len(tokens) >= 9
    ):
        return tokens[1], tokens[5]
    if command in {"ascan", "dscan", "a2scan", "d2scan", "a3scan", "d3scan"}:
        count = int(command[1]) if command[1].isdigit() else 1
        if len(tokens) >= 1 + 3 * count:
            return tuple(tokens[1 + 3 * i] for i in range(count))
    return ()


def _read_motor(scan, motor):
    # Acquisition channels take precedence over the initial motor snapshot.
    candidates = [
        f"measurement/{motor}",
        f"measurement/{motor}_position",
        f"instrument/positioners/{motor}",
        f"instrument/positioners/{motor}_position",
    ]
    scalar_fallback = None
    for name in candidates:
        node = scan.get(name)
        if not isinstance(node, h5py.Dataset):
            continue
        if node.ndim > 2 or node.dtype.kind not in "iuf":
            continue
        values = np.asarray(node[()], dtype=float).reshape(-1)
        if not values.size:
            continue
        unit = _text(node.attrs.get("units", node.attrs.get("unit", "")))
        info = values, unit, name.startswith("measurement/"), node.shape == ()
        if values.size > 1:
            return info
        if scalar_fallback is None:
            scalar_fallback = info
    return scalar_fallback


def get_scan_positions(
    path_dset: str | Path,
    *,
    x_motor: str | None = None,
    y_motor: str | None = None,
    scan_nos=None,
    scan_type=None,
) -> list[ScanPositions]:
    """Read acquired 2D trajectories for scans in a BLISS master HDF5 file.

    Both axes must describe the same motors for every scan. If omitted, they are
    inferred from the first recognized 2D scan title (``mesh``, ``amesh``,
    ``dmesh``, ``sxdm``, ``asxdm``, ``dsxdm``, ``kmap``, or two-motor scans).
    For files containing only line or point scans, supply ``x_motor`` and
    ``y_motor`` explicitly. Motor channels are sought in ``measurement`` and
    ``instrument/positioners``.

    Scalar stationary motor snapshots are broadcast over acquired coordinates
    of the other motor. A moving motor with only an initial snapshot is skipped,
    rather than drawing its requested trajectory. Differing array lengths are
    truncated to their shared acquired prefix with a warning (interrupted scans).
    Scans without usable coordinates are skipped with a warning; no detector
    images are read. ``scan_nos`` optionally limits entries (e.g. ``["1.1"]``).

    ``scan_type`` optionally selects a family or sequence of families:
    ``"sxdm"`` includes sxdm/asxdm/dsxdm/kmap, ``"mesh"`` includes
    mesh/amesh/dmesh, and ``"ascan"`` includes ascan/dscan. Other supported
    command names (e.g. ``"asxdm"``, ``"ct"``, ``"a2scan"``) select exact
    commands. Matching is case-insensitive and uses the first command token.
    ``None`` selects all types; an empty sequence selects none. This selection
    intersects ``scan_nos`` and is applied before reading motor coordinates.
    Axes are inferred from selected scans first, falling back to other scans
    within ``scan_nos`` when selecting only line/point scans from a mixed file.

    Returns an empty list when the file has no scans. Raises ``ValueError`` if
    axes cannot be inferred or only one axis is supplied. Coordinate values keep
    the file's native units; the function performs no unit conversion.
    """
    accepted_commands = _scan_type_commands(scan_type)
    if (x_motor is None) != (y_motor is None):
        raise ValueError("Supply both x_motor and y_motor, or neither.")
    if x_motor is not None and (not x_motor or not y_motor or x_motor == y_motor):
        raise ValueError("x_motor and y_motor must be distinct, nonempty motor names.")
    if isinstance(scan_nos, (str, int)):
        scan_nos = [scan_nos]
    requested = (
        None
        if scan_nos is None
        else {f"{s}.1" if isinstance(s, int) else str(s) for s in scan_nos}
    )
    result = []
    with h5py.File(path_dset, "r") as h5f:
        entries = _scan_entries(h5f)
        if requested is not None:
            absent = requested.difference(entries)
            if absent:
                raise KeyError(f"Scans not found: {', '.join(sorted(absent))}")
            entries = [name for name in entries if name in requested]
        if not entries:
            return []
        titles = {name: _scalar_text(h5f[name], "title") for name in entries}
        candidates = entries
        if accepted_commands is not None:
            entries = [
                name
                for name in entries
                if _scan_command(titles[name]) in accepted_commands
            ]
        if not entries:
            return []
        if x_motor is None:
            # Prefer selected 2D scan axes. Other entries contribute metadata
            # only when a selected line/point needs a second spatial axis.
            inference_order = entries + [
                name for name in candidates if name not in entries
            ]
            for name in inference_order:
                motors = _command_motors(titles[name])
                if len(motors) == 2:
                    x_motor, y_motor = motors
                    break
            if x_motor is None:
                raise ValueError(
                    "Cannot infer two motor axes. Supply x_motor and y_motor "
                    "explicitly for line/point scans or unfamiliar scan commands."
                )
        for name in entries:
            scan = h5f[name]
            title = titles[name]
            motors = _command_motors(title)
            try:
                coords = [_read_motor(scan, motor) for motor in (x_motor, y_motor)]
            except (KeyError, OSError) as err:
                warnings.warn(
                    f"Skipping scan {name}: inaccessible motor coordinates ({err}).",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            if any(coord is None for coord in coords):
                warnings.warn(
                    f"Skipping scan {name}: missing coordinates for {x_motor}/{y_motor}.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            x_info, y_info = coords
            if any(
                motor in motors and len(info[0]) == 1 and not info[2]
                for motor, info in zip((x_motor, y_motor), coords)
            ):
                warnings.warn(
                    f"Skipping scan {name}: a scanned motor has only a static snapshot.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            x, y = x_info[0], y_info[0]
            # An acquired one-element array can be an interrupted motor channel;
            # it is not evidence that the motor stayed fixed for other samples.
            if x.size == 1 and x_motor not in motors and (not x_info[2] or x_info[3]):
                x = np.full(y.size, x.item())
            elif y.size == 1 and y_motor not in motors and (not y_info[2] or y_info[3]):
                y = np.full(x.size, y.item())
            if x.size != y.size:
                warnings.warn(
                    f"Scan {name}: coordinate lengths differ; using their shared acquired prefix.",
                    UserWarning,
                    stacklevel=2,
                )
                length = min(x.size, y.size)
                x, y = x[:length], y[:length]
            if not np.any(np.isfinite(x) & np.isfinite(y)):
                warnings.warn(
                    f"Skipping scan {name}: no finite coordinate pairs.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            result.append(
                ScanPositions(
                    name,
                    title,
                    _scalar_text(scan, "start_time"),
                    x_motor,
                    y_motor,
                    x,
                    y,
                    np.arange(x.size),
                    x_info[1],
                    y_info[1],
                )
            )
    return result
