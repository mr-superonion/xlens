from typing import Any

from lsst.afw.image import ExposureF
from lsst.pipe.base import Task
from numpy.typing import NDArray


class SimBaseTask(Task):
    _DefaultName = "SimBaseTask"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        pass

    def get_perturbation_object(self, **kwargs: Any) -> object:
        raise NotImplementedError(
            "'get_perturbation_object' must be implemented by subclasses."
        )

    def get_dm_exposure(
        self, *, ifield: int, mode: int, rotId: int, band_list: list[str]
    ) -> ExposureF:
        raise NotImplementedError(
            "'get_dm_exposure' must be implemented by subclasses."
        )

    def get_noise_corr(self) -> NDArray | None:
        raise NotImplementedError(
            "'get_noise_corr' must be implemented by subclasses."
        )
