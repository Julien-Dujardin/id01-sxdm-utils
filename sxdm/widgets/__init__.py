"""Legacy Jupyter interfaces; install the jupyter and analysis extras."""

from importlib import import_module

_MODULES = ["bliss", "spec", "xsocs"]
_EXPORTS = {"Inspect4DArray": ("Inspect4DArray", "Inspect4DArray")}
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
