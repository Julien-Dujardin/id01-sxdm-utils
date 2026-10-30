from importlib import import_module

_MODULES = ["bliss", "spec", "xsocs", "scan_positions"]
_EXPORTS = {
    "_get_chunk_indexes": ("utils", "_get_chunk_indexes"),
    "_get_qspace_avg_chunk": ("utils", "_get_qspace_avg_chunk"),
    "get_scan_positions": ("scan_positions", "get_scan_positions"),
    "ScanPositions": ("scan_positions", "ScanPositions"),
}
__all__ = list(_MODULES) + list(_EXPORTS)


def __getattr__(name):
    if name in _MODULES:
        value = import_module(f"{__name__}.{name}")
    elif name in _EXPORTS:
        module, attr = _EXPORTS[name]
        value = getattr(import_module(f"{__name__}.{module}"), attr)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
