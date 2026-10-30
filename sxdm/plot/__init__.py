from importlib import import_module

_MODULES = ["colors", "utils", "interactive"]
_EXPORTS = {
    "add_hsv_colorbar": ("utils", "add_hsv_colorbar"),
    "add_colorbar": ("utils", "add_colorbar"),
    "add_directions": ("utils", "add_directions"),
    "add_scalebar": ("utils", "add_scalebar"),
    "add_roi_box": ("utils", "add_roi_box"),
    "add_roilabel": ("utils", "add_roilabel"),
    "add_letter": ("utils", "add_letter"),
    "make_hsv": ("utils", "make_hsv"),
    "gif_sxdm_sums": ("utils", "gif_sxdm_sums"),
    "gif_sxdm": ("utils", "gif_sxdm"),
    "get_scan_map": ("interactive", "get_scan_map"),
    "get_counter_map": ("interactive", "get_counter_map"),
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
