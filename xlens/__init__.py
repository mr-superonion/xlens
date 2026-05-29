import importlib

from .__version__ import __version__  # noqa

_SUBMODULES = (
    "analysis",
    "catalog",
    "process_pipe",
    "processor",
    "simulator",
    "utils",
    "wcs",
)


def __getattr__(name):
    if name in _SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))


__all__ = list(_SUBMODULES)
