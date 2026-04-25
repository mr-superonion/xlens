"""Stand-in deferred butler handles for examples and tests.

LSST PipelineTask code paths call ``handle.get()`` for the exposure and
read ``handle.dataId`` for ID generation. ``ExposureHandle`` plus
``make_exposure_handles`` provide a minimal implementation of both, so
notebooks and tests can drive ``MeasureCoaddsPipe.run`` (and similar)
without a real butler.
"""

from typing import Any

from lsst.daf.butler import DataCoordinate, DimensionUniverse


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
) -> DataCoordinate:
    """Build a fully-expanded ``DataCoordinate`` with stub skymap record.

    Suitable for the stand-in butler handles used in examples and tests.
    """
    if universe is None:
        universe = DimensionUniverse()
    skymap_element = universe["skymap"]
    skymap_record = skymap_element.RecordClass(
        name=skymap, hash=b"0" * 32,
        tract_max=100, patch_nx_max=1000, patch_ny_max=1000,
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
) -> dict[str, ExposureHandle]:
    """Wrap one exposure or a ``{band: exposure}`` dict in handles.

    Returns a ``{band: ExposureHandle}`` dict suitable to pass as
    ``exposure_handles_dict`` to ``MeasureCoaddsPipe.run``.
    """
    if not isinstance(exposures, dict):
        exposures = {band: exposures}
    universe = DimensionUniverse()
    out: dict[str, ExposureHandle] = {}
    for b, exp in exposures.items():
        data_id = make_data_id(
            skymap=skymap, tract=tract, patch=patch, band=b,
            instrument=instrument, universe=universe,
        )
        out[b] = ExposureHandle(exp, data_id)
    return out
