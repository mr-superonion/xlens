"""Smoke / shear-recovery test mirroring example4 (isotropic part).

Simulates a single noiseless patch with isotropic galaxies, runs
``MeasureCoaddsPipe.run()`` (detection + forced FPFS measurement in one
call), checks that the recovered shear matches the input, and runs
``matchPipe`` against the truth catalog. The patch size matches the
value used in
``examples/shear/example4_blended_sim_measure_match.ipynb``.
"""

import numpy as np
import pytest
from lsst.skymap.discreteSkyMap import (
    DiscreteSkyMap,
    DiscreteSkyMapConfig,
)

from xlens.processor.match import matchPipe, matchPipeConfig
from xlens.processor.measure_coadds import (
    MeasureCoaddsPipe,
    MeasureCoaddsPipeConfig,
)
from xlens.simulator.catalog import (
    CatalogShearTask,
    CatalogShearTaskConfig,
)
from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask
from xlens.utils.handle import make_exposure_handles

PIXEL_SCALE = 0.2
MAG_ZERO = 30.0
PATCH_DIM = 4000  # same value used in example4 notebook


def _make_skymap() -> DiscreteSkyMap:
    config = DiscreteSkyMapConfig()
    config.projection = "TAN"
    config.raList = [0.0]
    config.decList = [0.0]
    config.radiusList = [0.15]
    config.rotation = 0.0
    config.patchInnerDimensions = [PATCH_DIM, PATCH_DIM]
    config.patchBorder = 100
    config.pixelScale = PIXEL_SCALE
    config.tractOverlap = 0.0
    return DiscreteSkyMap(config)


def _make_truth(skymap, tract_id: int):
    cat_config = CatalogShearTaskConfig()
    cat_config.kappa_value = 0.0
    cat_config.layout = "random"
    cat_config.z_bounds = [-0.01, 1.0, 20.0]
    cat_config.mode = 0  # g1 = -0.02 in every redshift bin
    cat_task = CatalogShearTask(config=cat_config)
    return cat_task.run(
        tract_info=skymap[tract_id],
        seed=0,
    ).truthCatalog


def _simulate(skymap, tract_id: int, patch_id: int, band: str, truth):
    sim_config = MultibandSimConfig()
    sim_config.survey_name = "lsst"
    sim_config.draw_image_noise = False
    sim_config.force_isotropic = True
    sim_task = MultibandSimTask(config=sim_config)
    return sim_task.run(
        tract_info=skymap[tract_id],
        patch_id=patch_id,
        band=band,
        seed=0,
        truthCatalog=truth,
    )


def _measure_coadd(outcome, skymap, tract_id, patch_id, band):
    config = MeasureCoaddsPipeConfig()
    config.anacal.force_size = False
    config.anacal.sigma_arcsec = 0.38
    config.anacal.num_epochs = 0
    config.anacal.do_noise_bias_correction = False
    config.anacal.validate_psf = False
    config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
    config.fpfs.do_noise_bias_correction = False
    config.fpfs.return_only_linear_modes = False

    pipe = MeasureCoaddsPipe(config=config)
    handles = make_exposure_handles(
        outcome.simExposure,
        tract=tract_id, patch=patch_id, band=band,
    )
    return pipe.run(
        exposure_handles_dict=handles,
        corr_array=None,
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
    ).anacalCatalog


def _shear(catalog, e1_col: str, e2_col: str):
    de1_col = e1_col.replace("_e1", "_de1_dg1")
    de2_col = e2_col.replace("_e2", "_de2_dg2")
    e1 = catalog["wsel"] * catalog[e1_col]
    de1 = (
        catalog["dwsel_dg1"] * catalog[e1_col]
        + catalog["wsel"] * catalog[de1_col]
    )
    e2 = catalog["wsel"] * catalog[e2_col]
    de2 = (
        catalog["dwsel_dg2"] * catalog[e2_col]
        + catalog["wsel"] * catalog[de2_col]
    )
    return float(np.sum(e1) / np.sum(de1)), float(np.sum(e2) / np.sum(de2))


def test_sim_detect_isotropic_shear_recovery():
    """End-to-end: simulate noiseless isotropic patch, run
    MeasureCoaddsPipe.run() to detect and force-measure, check shear
    recovery is consistent with input, then run matchPipe."""
    tract_id = 0
    patch_id = 0
    band = "i"
    input_g1, input_g2 = -0.02, 0.0
    atol = 5e-3

    skymap = _make_skymap()
    truth = _make_truth(skymap, tract_id)
    outcome = _simulate(skymap, tract_id, patch_id, band, truth)

    catalog = _measure_coadd(outcome, skymap, tract_id, patch_id, band)
    assert len(catalog) > 0, "no objects detected"

    g1_det, g2_det = _shear(catalog, "fpfs_e1", "fpfs_e2")
    g1_k1, g2_k1 = _shear(catalog, "i_fpfs1_e1", "i_fpfs1_e2")

    np.testing.assert_allclose(
        g1_det, input_g1, atol=atol, err_msg="Det kernel g1"
    )
    np.testing.assert_allclose(
        g2_det, input_g2, atol=atol, err_msg="Det kernel g2"
    )
    np.testing.assert_allclose(
        g1_k1, input_g1, atol=atol, err_msg="Kernel1 g1"
    )
    np.testing.assert_allclose(
        g2_k1, input_g2, atol=atol, err_msg="Kernel1 g2"
    )

    match_config = matchPipeConfig()
    match_config.mag_zero = MAG_ZERO
    match_config.mag_max_truth = 28.0
    match_task = matchPipe(config=match_config)
    match = match_task.run(
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
        catalog=catalog,
        dm_catalog=None,
        truth_catalog=truth,
    ).catalog
    assert len(match) > 0, "matchPipe returned no rows"
    assert "truth_index" in match.dtype.names


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
