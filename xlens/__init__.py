from .__version__ import __version__  # noqa

try:
    from . import (
        catalog,
        analysis,
        process_pipe,
        processor,
        simulator,
        utils,
    )
    __all__ = [
        "catalog",
        "utils",
        "simulator",
        "processor",
        "analysis",
        "process_pipe",
    ]
except ModuleNotFoundError:
    from . import (
        catalog,
    )
    __all__ = ["catalog"]
