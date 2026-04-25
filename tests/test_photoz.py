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

DATA_DIR = Path(__file__).resolve().parent / "data"
FZB_MODEL = DATA_DIR / "model_inform_fzboost.pkl"
PHOTOZ_CATALOG = DATA_DIR / "catalog.fits"

# Reference values come from tests/test_catalog.py::test_pz, which uses
# the same catalog and FlexZBoost model.
ZBEST_REF = np.array([4.28967696, 0.72257506, 1.52362052])
ZMODE_REF = np.array([0.34, 0.73, 1.53])


def _load_photoz_catalog() -> np.ndarray:
    catalog = fitsio.read(os.fspath(PHOTOZ_CATALOG))
    if "object_id" not in catalog.dtype.names:
        catalog = rfn.append_fields(
            catalog, "object_id",
            np.arange(len(catalog), dtype=np.int64),
            usemask=False,
        )
    return catalog


def _make_config(*, output_pdfs: bool) -> photoZPipeConfig:
    cfg = photoZPipeConfig()
    cfg.model_path = os.fspath(FZB_MODEL)
    cfg.mag_zero = 30.0
    cfg.flux_name = "gauss2"
    cfg.bands = "ugrizy"
    cfg.ref_band = "i"
    cfg.do_distortions = False  # one undistorted call is enough
    cfg.output_pdfs = output_pdfs
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
    """With ``output_pdfs=True`` the result includes a
    (N, 1, NUM_Z_GRIDS) array of non-negative finite pdfs."""
    if not FZB_MODEL.exists() or not PHOTOZ_CATALOG.exists():
        pytest.skip("photo-z fixtures not available")

    from xlens.catalog.redshift import NUM_Z_GRIDS

    catalog = _load_photoz_catalog()
    task = photoZPipe(config=_make_config(output_pdfs=True))
    result = task.run(catalog=catalog)
    pdfs = result.redshiftPdfs

    assert pdfs.shape == (len(catalog), 1, NUM_Z_GRIDS)
    assert np.all(pdfs >= 0)
    assert np.all(np.isfinite(pdfs))


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
