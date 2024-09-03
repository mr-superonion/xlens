from typing import Any

from lsst.afw.image import ExposureF
from lsst.pipe.base import Task
from numpy.typing import NDArray

from .utils import LsstPsf


class MeasBaseTask(Task):
    _DefaultName = "MeasBaseTask"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        pass

    def run(
        self,
        *,
        pixel_scale: float,
        mag_zero: float,
        noise_std: float,
        gal_array: NDArray,
        psf_array: NDArray,
        mask_array: NDArray,
        noise_array: NDArray | None,
        detection: NDArray | None,
        psf_object: LsstPsf | None,
        **kwargs,
    ):
        raise NotImplementedError("'run' must be implemented by subclasses.")

    def prepare_data(
        self,
        *,
        exposure: ExposureF,
        seed: int,
        noise_corr: NDArray | None = None,
        detection: NDArray | None = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "'prepare_data' must be implemented by subclasses."
        )
