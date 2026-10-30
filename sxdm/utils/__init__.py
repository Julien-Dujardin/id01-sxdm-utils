from importlib import import_module

_MODULES = ["bliss", "nanomax", "general"]
_EXPORTS = {
    "get_detector_roilist": ("general", "get_detector_roilist"),
    "get_filelist": ("general", "get_filelist"),
    "get_pi_extents": ("general", "get_pi_extents"),
    "get_q_extents": ("general", "get_q_extents"),
    "get_qspace_coords": ("general", "get_qspace_coords"),
    "get_shift": ("general", "get_shift"),
    "calc_refl_id01": ("general", "calc_refl_id01"),
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
