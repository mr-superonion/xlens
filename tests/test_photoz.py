"""Smoke / regression tests for ``xlens.processor.photoz.photoZPipe``.

Drives the pipeline task on the bundled multi-band test catalog
(``tests/data/catalog.fits``) using the bundled FlexZBoost model
(``tests/data/model_inform_fzboost.pkl``), and asserts the per-distortion
point estimates match the reference values from
``tests/test_catalog.py::test_pz`` (the same catalog and model are used,
so the values must agree to single-precision tolerance).
"""

import os
from pathlib import Path

import fitsio
import numpy as np
import pytest
from numpy.lib import recfunctions as rfn

from xlens.processor.photoz import photoZPipe, photoZPipeConfig
from xlens.utils.constants import MAG_ZERO_AB

# The bundled fixture was measured at the legacy mag_zero=30. The pipeline now
# fixes the output zeropoint at MAG_ZERO_AB (31.4), so bring the fixture's flux
# family onto 31.4; the recovered AB magnitudes (and thus the z estimates) are
# then invariant vs. the reference values, which were computed at 30.
_FIXTURE_MAG_ZERO = 30.0

DATA_DIR = Path(__file__).resolve().parent / "data"
FZB_MODEL = DATA_DIR / "model_inform_fzboost.pkl"
PHOTOZ_CATALOG = DATA_DIR / "catalog.fits"

# Reference values come from tests/test_catalog.py::test_pz, which uses
# the same catalog and FlexZBoost model.
ZBEST_REF = np.array([4.28967696, 0.72257506, 1.52362052])
ZMODE_REF = np.array([0.34, 0.73, 1.53])


def _load_photoz_catalog() -> np.ndarray:
    catalog = fitsio.read(os.fspath(PHOTOZ_CATALOG))
    r = 10.0 ** ((MAG_ZERO_AB - _FIXTURE_MAG_ZERO) / 2.5)
    if r != 1.0:
        for name in catalog.dtype.names:
            if "flux_gauss" in name:  # flux / dflux / flux_err (not mag_*)
                catalog[name] = catalog[name] * r
    # The bundled fixture has single-letter per-band columns
    # (``u_flux_gauss2``);
    # the pipeline now uses survey-prefixed band names, so re-key the per-band
    # columns to ``lsst_<band>_...`` to match ``bands=["lsst_u", ...]``.
    band_rename = {
        n: f"lsst_{n}"
        for n in catalog.dtype.names
        if len(n) >= 2 and n[0] in "ugrizy" and n[1] == "_"
    }
    if band_rename:
        catalog = np.asarray(rfn.rename_fields(catalog, band_rename))
    if "object_id" not in catalog.dtype.names:
        catalog = rfn.append_fields(
            catalog, "object_id",
            np.arange(len(catalog), dtype=np.int64),
            usemask=False,
        )
    return catalog


def _make_config(
    *, output_pdfs: bool, output_distorted_pdfs: bool = False
) -> photoZPipeConfig:
    cfg = photoZPipeConfig()
    cfg.model_path = os.fspath(FZB_MODEL)
    cfg.flux_name = "gauss2"
    cfg.bands = [f"lsst_{b}" for b in "ugrizy"]
    cfg.ref_band = "lsst_i"
    cfg.do_distortions = False  # one undistorted call is enough
    cfg.output_pdfs = output_pdfs
    cfg.output_distorted_pdfs = output_distorted_pdfs
    return cfg


def test_photoz_point_estimates_match_reference():
    """photoZPipe.run on the bundled catalog must reproduce the
    reference zbest / zmode values from tests/test_catalog.py::test_pz.
    """
    if not FZB_MODEL.exists() or not PHOTOZ_CATALOG.exists():
        pytest.skip("photo-z fixtures not available")

    catalog = _load_photoz_catalog()
    task = photoZPipe(config=_make_config(output_pdfs=False))
    result = task.run(catalog=catalog)

    assert not hasattr(result, "redshiftPdfs")
    points = result.redshiftCatalog
    assert len(points) == len(catalog)
    assert "object_id" in points.colnames

    # photoZPipe emits float32 columns; relax to single-precision rtol.
    np.testing.assert_allclose(
        np.asarray(points["zbest_0"]), ZBEST_REF, rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(points["zmode_0"]), ZMODE_REF, rtol=1e-5,
    )


def test_photoz_returns_pdfs_when_requested():
    """``output_pdfs=True`` alone gives the UNDISTORTED grid, (N, nzbins).

    The distortion axis is opt-in (``output_distorted_pdfs``): carrying
    p(z) for every shear distortion multiplies the output size by the
    number of distortions, which is not what most callers want.
    """
    if not FZB_MODEL.exists() or not PHOTOZ_CATALOG.exists():
        pytest.skip("photo-z fixtures not available")

    catalog = _load_photoz_catalog()
    config = _make_config(output_pdfs=True)
    task = photoZPipe(config=config)
    result = task.run(catalog=catalog)
    pdfs = result.redshiftPdfs

    assert pdfs.shape == (len(catalog), config.nzbins)
    assert np.all(pdfs >= 0)
    assert np.all(np.isfinite(pdfs))


def test_photoz_distorted_pdfs_add_a_distortion_axis():
    """``output_distorted_pdfs=True`` gives (N, ndist, nzbins).

    ``do_distortions=False`` leaves a single ("0") distortion, so the
    extra axis has length one here -- but it is present, which is the
    difference from the default above.
    """
    if not FZB_MODEL.exists() or not PHOTOZ_CATALOG.exists():
        pytest.skip("photo-z fixtures not available")

    catalog = _load_photoz_catalog()
    config = _make_config(output_pdfs=True, output_distorted_pdfs=True)
    task = photoZPipe(config=config)
    result = task.run(catalog=catalog)
    pdfs = result.redshiftPdfs

    assert pdfs.shape == (len(catalog), 1, config.nzbins)
    assert np.all(pdfs >= 0)
    assert np.all(np.isfinite(pdfs))


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
