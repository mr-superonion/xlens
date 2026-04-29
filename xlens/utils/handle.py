"""Stand-in deferred butler handles for examples and tests.

LSST PipelineTask code paths call ``handle.get()`` for the exposure and
read ``handle.dataId`` for ID generation. ``ExposureHandle`` plus
``make_exposure_handles`` provide a minimal implementation of both, so
notebooks and tests can drive ``MeasureCoaddsPipe.run`` (and similar)
without a real butler.

``SimulatedExposureHandle`` is a richer variant whose ``.get()`` lazily
runs a ``MultibandSimTask`` to render the exposure on demand, used by
``MeasureCoaddsPipe`` when ``use_sim=True``.
"""

from typing import TYPE_CHECKING, Any

from lsst.afw.image import ExposureF, MaskX
from lsst.daf.butler import DataCoordinate, DimensionUniverse
from lsst.skymap import BaseSkyMap
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..simulator.sim import MultibandSimTask

DEFAULT_TRACT_MAX = 100_000
DEFAULT_PATCH_NX_MAX = 1000
DEFAULT_PATCH_NY_MAX = 1000


def _record_maxes_from_skymap(
    skyMap: BaseSkyMap,
) -> tuple[int, int, int]:
    """Read ``(tract_max, patch_nx_max, patch_ny_max)`` from a skymap.

    ``tract_max`` is one greater than the largest tract ID; the patch
    maxes come from the first tract's patch grid (assumed uniform).
    """
    tract_max = len(skyMap)
    nx, ny = skyMap[0].getNumPatches()
    return tract_max, int(nx), int(ny)


class ExposureHandle:
    """Minimal stand-in for a deferred butler dataset handle.

    Exposes ``.get()`` returning the wrapped exposure and ``.dataId``
    holding the associated ``DataCoordinate``.
    """

    def __init__(self, exposure: Any, data_id: DataCoordinate):
        self._exp = exposure
        self.dataId = data_id

    def get(self) -> Any:
        return self._exp


def make_data_id(
    *,
    skymap: str = "test",
    tract: int = 0,
    patch: int = 0,
    band: str = "i",
    instrument: str | None = None,
    universe: DimensionUniverse | None = None,
    skyMap: BaseSkyMap | None = None,
) -> DataCoordinate:
    """Build a fully-expanded ``DataCoordinate`` with stub skymap record.

    Suitable for the stand-in butler handles used in examples and tests.
    When ``skyMap`` is provided, ``tract_max`` / ``patch_nx_max`` /
    ``patch_ny_max`` on the stub record are derived from it; otherwise
    the default bounds are used.
    """
    if universe is None:
        universe = DimensionUniverse()
    if skyMap is not None:
        tract_max, patch_nx_max, patch_ny_max = _record_maxes_from_skymap(
            skyMap,
        )
    else:
        tract_max = DEFAULT_TRACT_MAX
        patch_nx_max = DEFAULT_PATCH_NX_MAX
        patch_ny_max = DEFAULT_PATCH_NY_MAX
    skymap_element = universe["skymap"]
    skymap_record = skymap_element.RecordClass(
        name=skymap, hash=b"0" * 32,
        tract_max=tract_max,
        patch_nx_max=patch_nx_max,
        patch_ny_max=patch_ny_max,
    )
    fields: dict[str, Any] = {
        "skymap": skymap, "tract": tract,
        "patch": patch, "band": band,
    }
    if instrument is not None:
        fields["instrument"] = instrument
    return DataCoordinate.standardize(
        fields, universe=universe,
    ).expanded({skymap_element: skymap_record})


def make_exposure_handles(
    exposures: Any,
    *,
    skymap: str = "test",
    tract: int = 0,
    patch: int = 0,
    band: str = "i",
    instrument: str | None = None,
    skyMap: BaseSkyMap | None = None,
) -> dict[str, ExposureHandle]:
    """Wrap one exposure or a ``{band: exposure}`` dict in handles.

    Returns a ``{band: ExposureHandle}`` dict suitable to pass as
    ``exposure_handles_dict`` to ``MeasureCoaddsPipe.run``. When a
    ``skyMap`` is provided it is forwarded to :func:`make_data_id` so
    the stub skymap-record bounds match the real skymap.
    """
    if not isinstance(exposures, dict):
        exposures = {band: exposures}
    universe = DimensionUniverse()
    out: dict[str, ExposureHandle] = {}
    for b, exp in exposures.items():
        data_id = make_data_id(
            skymap=skymap, tract=tract, patch=patch, band=b,
            instrument=instrument, universe=universe,
            skyMap=skyMap,
        )
        out[b] = ExposureHandle(exp, data_id)
    return out


class SimulatedExposureHandle:
    """Lazy stand-in for a butler ``DeferredDatasetHandle`` whose
    ``.get()`` produces a freshly simulated exposure for one band.

    Provides the same ``.get()`` / ``.dataId`` interface used by
    ``MeasureCoaddsPipe``, so the same downstream code path can
    measure both real coadds and simulated exposures without branching.
    """

    def __init__(
        self,
        *,
        simulator: "MultibandSimTask",
        tract_info,
        patch: int,
        band: str,
        seed: int,
        truthCatalog,
        psf_array: NDArray | None = None,
        corr_array: NDArray | None = None,
        mask: MaskX | None = None,
    ):
        self._simulator = simulator
        self._tract_info = tract_info
        self._patch = patch
        self._band = band
        self._seed = seed
        self._truthCatalog = truthCatalog
        self._psf_array = psf_array
        self._corr_array = corr_array
        self._mask = mask

    def get(self) -> ExposureF:
        kwargs: dict[str, Any] = {
            "tract_info": self._tract_info,
            "patch_id": self._patch,
            "band": self._band,
            "seed": self._seed,
            "truthCatalog": self._truthCatalog,
        }
        if self._psf_array is not None:
            kwargs["psfArray"] = self._psf_array
        if self._corr_array is not None:
            kwargs["noiseCorrArray"] = self._corr_array
        if self._mask is not None:
            kwargs["mask"] = self._mask
        return self._simulator.run(**kwargs).simExposure
