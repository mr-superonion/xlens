"""End-to-end smoke test: simulate a tiny multiband patch, run
``MeasureCoaddsPipe`` to detect + force-measure FPFS fluxes, then
estimate photo-z with the bundled FlexZBoost model.

Bands match those in the bundled photo-z reference catalog
(``tests/data/catalog.fits``): u, g, r, i, z, y.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from lsst.skymap.discreteSkyMap import (
    DiscreteSkyMap,
    DiscreteSkyMapConfig,
)
from numpy.lib import recfunctions as rfn

from xlens.processor.measure_coadds import (
    MeasureCoaddsPipe,
    MeasureCoaddsPipeConfig,
)
from xlens.processor.photoz import photoZPipe, photoZPipeConfig
from xlens.simulator.catalog import (
    CatalogShearTask,
    CatalogShearTaskConfig,
)
from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask
from xlens.utils.handle import make_exposure_handles

DATA_DIR = Path(__file__).resolve().parent / "data"
FZB_MODEL = DATA_DIR / "model_inform_fzboost.pkl"
PHOTOZ_CATALOG = DATA_DIR / "catalog.fits"

PIXEL_SCALE = 0.2
PATCH_DIM = 200
MAG_ZERO = 30.0
BANDS = "ugrizy"
TRACT_ID = 0
PATCH_ID = 0


def _make_skymap() -> DiscreteSkyMap:
    cfg = DiscreteSkyMapConfig()
    cfg.projection = "TAN"
    cfg.raList = [0.0]
    cfg.decList = [0.0]
    cfg.radiusList = [0.05]
    cfg.rotation = 0.0
    cfg.patchInnerDimensions = [PATCH_DIM, PATCH_DIM]
    cfg.patchBorder = 50
    cfg.pixelScale = PIXEL_SCALE
    cfg.tractOverlap = 0.0
    return DiscreteSkyMap(cfg)


def _make_truth(skymap):
    cc = CatalogShearTaskConfig()
    cc.kappa_value = 0.0
    cc.layout = "random"
    cc.z_bounds = [-0.01, 1.0, 20.0]
    cc.mode = 0
    return CatalogShearTask(config=cc).run(
        tract_info=skymap[TRACT_ID], seed=0,
    ).truthCatalog


def _simulate_band(skymap, band: str, truth):
    sc = MultibandSimConfig()
    sc.survey_name = "lsst"
    sc.draw_image_noise = True
    return MultibandSimTask(config=sc).run(
        tract_info=skymap[TRACT_ID],
        patch_id=PATCH_ID,
        band=band,
        seed=0,
        truthCatalog=truth,
    ).simExposure


def _measure(skymap, exposures: dict):
    cfg = MeasureCoaddsPipeConfig()
    cfg.anacal.force_size = False
    cfg.anacal.sigma_arcsec = 0.38
    cfg.anacal.num_epochs = 0
    cfg.anacal.do_noise_bias_correction = True
    cfg.anacal.validate_psf = False
    cfg.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
    cfg.fpfs.do_noise_bias_correction = True
    cfg.fpfs.return_only_linear_modes = False
    pipe = MeasureCoaddsPipe(config=cfg)
    handles = make_exposure_handles(
        exposures, tract=TRACT_ID, patch=PATCH_ID,
    )
    return pipe.run(
        exposure_handles_dict=handles,
        corr_array=None,
        skyMap=skymap,
        tract=TRACT_ID,
        patch=PATCH_ID,
    ).anacalCatalog


def _ensure_object_id(catalog: np.ndarray) -> np.ndarray:
    if "object_id" in (catalog.dtype.names or ()):
        return catalog
    return np.asarray(rfn.append_fields(
        catalog, "object_id",
        np.arange(len(catalog), dtype=np.int64),
        usemask=False,
    ))


def test_sim_measure_photoz():
    """Tiny 200x200 multi-band sim → MeasureCoaddsPipe → photoZPipe."""
    if not FZB_MODEL.exists():
        pytest.skip(f"FZB model not available at {FZB_MODEL}")

    skymap = _make_skymap()
    truth = _make_truth(skymap)
    exposures = {b: _simulate_band(skymap, b, truth) for b in BANDS}

    catalog = _measure(skymap, exposures)
    assert len(catalog) > 0, "no objects detected in 200x200 patch"
    # MeasureCoaddsPipe now emits the photoZPipe-compatible schema
    # (``{b}_flux_fpfs1`` etc.) directly via rename_flux_to_photoz_format
    # in xlens.processor.measure_coadds._force.
    for b in BANDS:
        assert f"{b}_flux_fpfs1" in catalog.dtype.names, b
        assert f"{b}_flux_fpfs1_err" in catalog.dtype.names, b
        assert f"{b}_dflux_fpfs1_dg1" in catalog.dtype.names, b
        assert f"{b}_dflux_fpfs1_dg2" in catalog.dtype.names, b

    catalog = _ensure_object_id(catalog)

    pz_cfg = photoZPipeConfig()
    pz_cfg.model_path = os.fspath(FZB_MODEL)
    pz_cfg.mag_zero = MAG_ZERO
    pz_cfg.flux_name = "fpfs1"
    pz_cfg.bands = BANDS
    pz_cfg.ref_band = "i"
    pz_cfg.do_distortions = False
    pz_cfg.output_pdfs = False

    result = photoZPipe(config=pz_cfg).run(catalog=catalog)
    points = result.redshiftCatalog
    assert len(points) == len(catalog)
    assert "zbest_0" in points.colnames
    assert "zmode_0" in points.colnames

    zbest = np.asarray(points["zbest_0"])
    zmode = np.asarray(points["zmode_0"])
    assert np.all(np.isfinite(zbest))
    assert np.all(np.isfinite(zmode))
    assert np.all(zbest >= 0.0)
    assert np.all(zmode >= 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
