from .__version__ import __version__  # noqa

try:
    from . import (  # noqa
        analysis,
        catalog,
        process_pipe,
        processor,
        simulator,
        utils,
        wcs,
    )
    __all__ = [
        "catalog",
        "utils",
        "simulator",
        "processor",
        "analysis",
        "process_pipe",
        "wcs",
    ]
except ModuleNotFoundError:
    from . import (
        catalog,
    )
    __all__ = ["catalog"]
