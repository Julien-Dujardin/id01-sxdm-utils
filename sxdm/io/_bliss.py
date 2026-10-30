"""Small, self-contained readers for the BLISS HDF5 layout.

These helpers keep ordinary BLISS analysis independent of the optional ESRF
geometry and XSOCS packages. Paths and already-open HDF5 files are accepted.
"""

from functools import wraps
from numbers import Integral
import re

import h5py
import hdf5plugin  # Register the detector compression filters with h5py.
import numpy as np


def ioh5(func):
    """Open a path read-only; leave caller-owned HDF5 handles open."""

    @wraps(func)
    def wrapper(filename, *args, **kwargs):
        if isinstance(filename, (h5py.File, h5py.Group)):
            return func(filename, *args, **kwargs)
        with h5py.File(filename, "r") as h5f:
            return func(h5f, *args, **kwargs)

    return wrapper


def _scan_no(scan_no):
    return f"{scan_no}.1" if isinstance(scan_no, Integral) else str(scan_no)


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@ioh5
def get_command(h5f, scan_no):
    return _text(h5f[f"{_scan_no(scan_no)}/title"][()])


@ioh5
def get_counter(h5f, scan_no, counter_name):
    group = h5f[f"{_scan_no(scan_no)}/measurement"]
    try:
        return group[counter_name][()]
    except KeyError as exc:
        raise KeyError(
            f"Counter {counter_name!r} not found. Available counters: {list(group)}"
        ) from exc


@ioh5
def get_positioner(h5f, scan_no, motor_name):
    return h5f[f"{_scan_no(scan_no)}/instrument/positioners/{motor_name}"][()]


@ioh5
def get_scan_shape(h5f, scan_no):
    """Return (slow, fast) dimensions for SXDM/mesh, length for line scans."""
    scan_no = _scan_no(scan_no)
    group = h5f[scan_no]
    command = get_command(h5f, scan_no)
    tokens = re.split(r"[\s,()]+", command.strip())
    kind = tokens[0]
    if any(name in kind for name in ("sxdm", "kmap")):
        if "technique/dim0" in group and "technique/dim1" in group:
            return [int(group[f"technique/dim{i}"][()]) for i in (1, 0)]
        if len(tokens) > 8:
            return [int(tokens[i]) for i in (8, 4)]
    if "mesh" in kind and len(tokens) > 8:
        return [int(tokens[i]) + 1 for i in (8, 4)]
    if "scan" in kind:
        measurement = group["measurement"]
        for name in ("acq_time", tokens[1]):
            if name in measurement:
                return measurement[name].shape[0]
    raise ValueError(f"Scan shape cannot be inferred from command: {command}")


@ioh5
def get_detector_aliases(h5f, scan_no):
    """Find NXdetectors and the historical ID01 aliases without reading frames."""
    group = h5f[f"{_scan_no(scan_no)}/instrument"]
    known = ("mpx1x4", "mpxgaas", "eiger2M", "detector", "merlin", "andor_zyla")
    detectors = [name for name in known if name in group]
    for name in group:
        obj = group.get(name)
        if not isinstance(obj, h5py.Group) or name in detectors:
            continue
        data = obj.get("data")
        if isinstance(data, h5py.Dataset) and data.ndim == 3:
            detectors.append(name)
    if not detectors:
        raise ValueError(
            f"No detector group found in {h5f.filename}:/{_scan_no(scan_no)}/instrument"
        )
    return detectors


def _check_detector(path_dset, scan_no, detector):
    detectors = get_detector_aliases(path_dset, scan_no)
    if detector is None:
        return detectors[0]
    if detector not in detectors:
        raise ValueError(
            f"Detector {detector!r} not found. Available detectors: {detectors}"
        )
    return detector


@ioh5
def get_roi_names(h5f, scan_no, only_sum=False):
    aliases = get_detector_aliases(h5f, scan_no)
    names = [
        name
        for name in h5f[f"{_scan_no(scan_no)}/measurement"]
        if any(alias in name for alias in aliases) and name not in aliases
    ]
    if only_sum:
        names = [
            name
            for name in names
            if not any(part in name for part in ("avg", "max", "min", "std"))
        ]
    return names
