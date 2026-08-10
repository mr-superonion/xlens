"""Smoke / shear-recovery test mirroring example4 (isotropic part).

Simulates a single noiseless patch with isotropic galaxies under an
isotropic Moffat PSF (``psf_e1 = psf_e2 = 0``), runs
``MeasureCoaddsPipe.run()`` (detection + forced FPFS measurement +
HSM PSF moments in one call), and exercises the result with two
tests sharing the same simulation:

* ``test_sim_detect_isotropic_shear_recovery`` — recovered shear
  matches the input g1/g2 and ``matchPipe`` joins to the truth catalog.
* ``test_psf_hsm_moments_isotropic_zero_spin2`` — the per-source
  ``ext_shapeHSM_*`` PSF-moment columns have the expected isotropy
  symmetries (Ixy = 0, Ixx = Iyy, all odd-index higher-order moments
  vanish) and are broadcast identically to every detected source.

The patch size matches
``examples/shear/example4_blended_sim_measure_match.ipynb``.
"""

from types import SimpleNamespace

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
PATCH_DIM = 4000
TRACT_ID = 0
PATCH_ID = 0
BAND = "i"
INPUT_G1 = -0.02
INPUT_G2 = 0.0


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
    # Pin the PSF to a perfectly isotropic Moffat so the PSF HSM test
    # can assert that all spin-2 (and any odd-index higher-order)
    # moments vanish. These are the sim defaults; setting them
    # explicitly makes the assumption obvious to the next reader.
    sim_config.psf_e1 = 0.0
    sim_config.psf_e2 = 0.0
    sim_task = MultibandSimTask(config=sim_config)
    return sim_task.run(
        tract_info=skymap[tract_id],
        patch_id=patch_id,
        band=band,
        seed=0,
        truthCatalog=truth,
    )


def _measure_coadd(outcome, skymap, tract_id, patch_id, band):
    """Detect + force-measure on the simulated patch. Always enables
    ``doPsfHsmMoments`` so the HSM PSF-moment columns are populated;
    the shear-recovery test ignores those extra columns."""
    config = MeasureCoaddsPipeConfig()
    config.anacal.force_size = False
    config.anacal.sigma_arcsec = 0.38
    config.anacal.num_epochs = 0
    config.anacal.do_noise_bias_correction = False
    config.anacal.validate_psf = False
    config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
    config.fpfs.do_noise_bias_correction = False
    config.doPsfHsmMoments = True

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


@pytest.fixture(scope="module")
def measured():
    """Run the full sim + detect + measure pipeline ONCE for the
    module; both tests below consume the same catalog.

    Returns a namespace with ``skymap``, ``truth``, ``catalog`` plus
    the (tract, patch, band) ids."""
    skymap = _make_skymap()
    truth = _make_truth(skymap, TRACT_ID)
    outcome = _simulate(skymap, TRACT_ID, PATCH_ID, BAND, truth)
    catalog = _measure_coadd(outcome, skymap, TRACT_ID, PATCH_ID, BAND)
    assert len(catalog) > 0, "no objects detected"
    return SimpleNamespace(
        skymap=skymap, truth=truth, catalog=catalog,
        tract_id=TRACT_ID, patch_id=PATCH_ID, band=BAND,
    )


def test_sim_detect_isotropic_shear_recovery(measured):
    """End-to-end: simulate noiseless isotropic patch, run
    MeasureCoaddsPipe.run() to detect and force-measure, check shear
    recovery is consistent with input, then run matchPipe."""
    atol = 5e-3

    g1_det, g2_det = _shear(measured.catalog, "fpfs_e1", "fpfs_e2")
    g1_k1, g2_k1 = _shear(measured.catalog, "lsst_i_fpfs1_e1", "lsst_i_fpfs1_e2")

    np.testing.assert_allclose(
        g1_det, INPUT_G1, atol=atol, err_msg="Det kernel g1"
    )
    np.testing.assert_allclose(
        g2_det, INPUT_G2, atol=atol, err_msg="Det kernel g2"
    )
    np.testing.assert_allclose(
        g1_k1, INPUT_G1, atol=atol, err_msg="Kernel1 g1"
    )
    np.testing.assert_allclose(
        g2_k1, INPUT_G2, atol=atol, err_msg="Kernel1 g2"
    )

    match_config = matchPipeConfig()
    match_config.mag_zero = MAG_ZERO
    match_config.mag_max_truth = 28.0
    match_task = matchPipe(config=match_config)
    match = match_task.run(
        skyMap=measured.skymap,
        tract=measured.tract_id,
        patch=measured.patch_id,
        catalog=measured.catalog,
        dm_catalog=None,
        truth_catalog=measured.truth,
    ).catalog
    assert len(match) > 0, "matchPipe returned no rows"
    assert "truth_index" in match.dtype.names


def test_psf_hsm_moments_isotropic_zero_spin2(measured):
    """Isotropic Moffat PSF (``psf_e1 = psf_e2 = 0`` — see
    ``_simulate``) must yield zero spin-2 moments — i.e. ``Ixy = 0``
    and ``Ixx = Iyy`` for the 2nd-order adaptive moments, and every
    higher-order moment with any odd index also = 0 (per
    ``HigherOrderMomentsPSF`` symmetry rule).

    Also verifies the per-cell PSF moments are broadcast identically
    to every source in the per-source catalog (one HSM measurement
    per band shared across all rows)."""
    catalog = measured.catalog
    # measurement writes survey-prefixed columns (survey default "lsst")
    band = f"lsst_{measured.band}"

    ixx_col = f"{band}_ext_shapeHSM_HsmPsfMoments_xx"
    iyy_col = f"{band}_ext_shapeHSM_HsmPsfMoments_yy"
    ixy_col = f"{band}_ext_shapeHSM_HsmPsfMoments_xy"
    flag_col = f"{band}_ext_shapeHSM_HsmPsfMoments_flag"
    for col in (ixx_col, iyy_col, ixy_col, flag_col):
        assert col in catalog.dtype.names, f"missing column {col}"

    flag = catalog[flag_col]
    assert not flag.any(), (
        f"HSM PSF measurement flagged {int(flag.sum())} / {len(flag)} rows"
    )

    # Per-band broadcast invariant: every row carries the same
    # per-exposure PSF value (MeasureCoaddsPipe runs one HSM
    # measurement at the exposure bbox centre per band).
    ixx = catalog[ixx_col]
    iyy = catalog[iyy_col]
    ixy = catalog[ixy_col]
    np.testing.assert_array_equal(ixx, ixx[0])
    np.testing.assert_array_equal(iyy, iyy[0])
    np.testing.assert_array_equal(ixy, ixy[0])

    # Spin-2 of the 2nd-order adaptive moments vanishes for an
    # isotropic PSF. Tolerance is loose vs. Ixx ~ a few px² because
    # the Moffat is sampled on a square pixel grid.
    atol_2nd = 1e-3
    np.testing.assert_allclose(
        ixy[0], 0.0, atol=atol_2nd,
        err_msg=(f"Ixy = {ixy[0]:.6e} px^2; expected 0 for isotropic "
                 f"PSF (Ixx = {ixx[0]:.4f}, Iyy = {iyy[0]:.4f})"),
    )
    np.testing.assert_allclose(
        ixx[0], iyy[0], atol=atol_2nd,
        err_msg=(f"Ixx - Iyy = {ixx[0]-iyy[0]:+.6e} px^2; expected 0 "
                 "for isotropic PSF"),
    )

    # Higher-order moments with any odd index also vanish — direct
    # consequence of the plugin's "symmetric profile → odd-index
    # moments are zero" rule (see _hsm_higher_moments.py docstring).
    ho_atol = 1e-3
    for p, q in [(0, 3), (1, 2), (2, 1), (3, 0), (1, 3), (3, 1)]:
        col = f"{band}_ext_shapeHSM_HigherOrderMomentsPSF_{p}{q}"
        v = catalog[col][0]
        np.testing.assert_allclose(
            v, 0.0, atol=ho_atol,
            err_msg=(f"M_{p}{q} = {v:+.6e}; expected 0 for isotropic PSF"),
        )

    # Even-index higher-order moments: circular symmetry forces
    # M_40 == M_04, and M_22 should be positive.
    m40 = catalog[f"{band}_ext_shapeHSM_HigherOrderMomentsPSF_40"][0]
    m04 = catalog[f"{band}_ext_shapeHSM_HigherOrderMomentsPSF_04"][0]
    m22 = catalog[f"{band}_ext_shapeHSM_HigherOrderMomentsPSF_22"][0]
    np.testing.assert_allclose(
        m40, m04, atol=ho_atol,
        err_msg=f"M_40 ({m40:.4f}) and M_04 ({m04:.4f}) should match",
    )
    assert m22 > 0.0, f"M_22 should be positive for a non-degenerate profile (got {m22})"


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
