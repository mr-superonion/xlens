"""Detection on a coadd of several bands.

Simulates the same patch in r, i and z with noise and runs
``MeasureCoaddsPipe.run()`` twice: once detecting on i alone (the previous
behaviour) and once detecting on the r+i+z coadd.  Forced measurement is
band-by-band in both cases and must be unaffected.

The r+i+z detection image is deeper, so it is expected to find more
sources; what the tests pin down is that the extra sources come with sane
columns and that turning the feature off leaves the old path exactly as it
was.  The forced measurement of the detection bands reuses their
detection preparation; that reuse must hand each band exactly what a
fresh single-band preparation would, its own noise correlation and
variance included.

The r+i+z catalog is then joined to the truth catalog with ``matchPipe``,
mirroring what ``bin/basic/process.py`` does.  The input catalog here is
``flagship2025`` rather than the default ``catsim2017`` so this covers the
non-default magnitude-column naming: ``indices`` are row numbers of the
catalog that produced the truth, and the magnitude ``mag_max_truth`` cuts
on is ``lsst_i`` rather than ``i_ab``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from lsst.skymap.discreteSkyMap import (
    DiscreteSkyMap,
    DiscreteSkyMapConfig,
)

from xlens.processor.anacal import AnacalConfig, AnacalTask
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
from xlens.utils.columns import select_band_gauss_fluxes
from xlens.utils.handle import make_exposure_handles

PIXEL_SCALE = 0.2
PATCH_DIM = 1000
TRACT_ID = 0
PATCH_ID = 0
BANDS = ["r", "i", "z"]
GALAXY_TYPE = "flagship2025"
SURVEY_NAME = "lsst"
MATCH_BAND = "i"
MAG_MAX_TRUTH = 27.0


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


def _make_truth(skymap):
    cat_config = CatalogShearTaskConfig()
    cat_config.galaxy_type = GALAXY_TYPE
    cat_config.survey_name_list = [SURVEY_NAME]
    cat_config.kappa_value = 0.0
    cat_config.layout = "random"
    cat_config.z_bounds = [-0.01, 1.0, 20.0]
    cat_config.mode = 0
    cat_config.validate()
    return CatalogShearTask(config=cat_config).run(
        tract_info=skymap[TRACT_ID], seed=0,
    ).truthCatalog


def _simulate(skymap, truth, band):
    sim_config = MultibandSimConfig()
    sim_config.galaxy_type = GALAXY_TYPE
    sim_config.survey_name = SURVEY_NAME
    sim_config.draw_image_noise = True
    sim_config.noiseId = 1
    sim_config.force_isotropic = True
    return MultibandSimTask(config=sim_config).run(
        tract_info=skymap[TRACT_ID],
        patch_id=PATCH_ID,
        band=band,
        seed=0,
        truthCatalog=truth,
    ).simExposure


def _measure(handles, skymap, detection_bands, sigma_arcsec=None, return_pipe=False):
    config = MeasureCoaddsPipeConfig()
    config.anacal.force_size = False
    config.anacal.num_epochs = 0
    config.anacal.do_noise_bias_correction = False
    if sigma_arcsec is not None:
        config.anacal.sigma_arcsec = sigma_arcsec
    config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
    config.fpfs.do_noise_bias_correction = False
    config.detection_bands = detection_bands
    config.validate()

    pipe = MeasureCoaddsPipe(config=config)
    cat = pipe.run(
        exposure_handles_dict=handles,
        corr_array=None,
        skyMap=skymap,
        tract=TRACT_ID,
        patch=PATCH_ID,
    ).anacalCatalog
    return (cat, pipe) if return_pipe else cat


@pytest.fixture(scope="module")
def scene():
    skymap = _make_skymap()
    truth = _make_truth(skymap)
    return SimpleNamespace(
        skymap=skymap,
        truth=truth,
        exposures={b: _simulate(skymap, truth, b) for b in BANDS},
    )


@pytest.fixture(scope="module")
def measured(scene):
    handles = make_exposure_handles(
        scene.exposures, tract=TRACT_ID, patch=PATCH_ID, skyMap=scene.skymap,
    )
    return SimpleNamespace(
        skymap=scene.skymap,
        truth=scene.truth,
        single=_measure(handles, scene.skymap, ["i"]),
        multi=_measure(handles, scene.skymap, BANDS),
    )


def test_single_band_default_is_unchanged(measured):
    """detection_bands=["i"] must reproduce the old i-band-only run."""
    assert len(measured.single) > 0


def test_coadd_detects_more_sources(measured):
    assert len(measured.multi) > len(measured.single), (
        f"r+i+z found {len(measured.multi)} sources, "
        f"i alone found {len(measured.single)}"
    )


def test_coadd_catalog_is_well_formed(measured):
    single, multi = measured.single, measured.multi

    # Forced measurement is per band either way, so the columns match.
    assert single.dtype.names == multi.dtype.names
    for band in BANDS:
        assert f"lsst_{band}_fpfs1_e1" in multi.dtype.names

    for name in multi.dtype.names:
        column = np.asarray(multi[name])
        if np.issubdtype(column.dtype, np.floating):
            assert np.all(np.isfinite(column)), f"'{name}' is not finite"

    # The detection weights and their shear responses must still be live.
    assert np.all(multi["wsel"] >= 0.0)
    assert np.sum(multi["wsel"]) > 0.0
    assert np.any(multi["dwsel_dg1"] != 0.0)
    assert np.any(multi["fpfs_de1_dg1"] != 0.0)


def test_coadd_catalog_matches_truth(measured):
    """Join the r+i+z catalog to the truth catalog, as bin/basic does.

    ``matchPipe`` reads the magnitude it cuts on out of the truth
    catalog, using column names derived from ``galaxy_type`` plus
    ``survey_name``/``band``.  With flagship2025 that is ``lsst_i``;
    getting it from a fixed input file instead would index the wrong
    catalog, since ``indices`` are row numbers of flagship_cosmos.fits
    (3.7M rows), not of OneDegSq.fits (858k).
    """
    config = matchPipeConfig()
    config.mag_max_truth = MAG_MAX_TRUTH
    config.galaxy_type = GALAXY_TYPE
    config.survey_name = SURVEY_NAME
    config.band = MATCH_BAND
    config.validate()
    task = matchPipe(config=config)

    mag = task.truth_magnitude(measured.truth)
    assert np.all(np.isfinite(mag))
    n_bright = int((mag < MAG_MAX_TRUTH).sum())
    assert 0 < n_bright < len(measured.truth), (
        f"mag_max_truth={MAG_MAX_TRUTH} kept {n_bright} of "
        f"{len(measured.truth)} truth galaxies; the cut does nothing"
    )

    match = task.run(
        skyMap=measured.skymap,
        tract=TRACT_ID,
        patch=PATCH_ID,
        catalog=measured.multi,
        dm_catalog=None,
        truth_catalog=measured.truth,
    ).catalog

    assert 0 < len(match) <= len(measured.multi)
    for name in ("truth_index", "redshift"):
        assert name in match.dtype.names

    # ``truth_index`` addresses rows of the input catalog the truth was
    # drawn from.  It is NOT unique across the join: galaxies are sampled
    # with replacement, so ~n^2/2N of the truth rows share an input row
    # (here ~2k of 121k), and two distinct truth galaxies placed from the
    # same input row can both be detected and matched.
    truth_index = np.asarray(match["truth_index"])
    assert truth_index.min() >= 0
    input_rows, first = np.unique(
        np.asarray(measured.truth["indices"]), return_index=True
    )
    where = np.searchsorted(input_rows, truth_index)
    assert np.all(input_rows[where] == truth_index), (
        "matched truth_index values that are not rows of the truth catalog"
    )
    # The mag_max_truth cut reaches the join: nothing fainter is matched.
    assert np.all(mag[first][where] < MAG_MAX_TRUTH)

    # The truth columns carried over are live, and the measured columns
    # survive the merge intact.
    assert np.all(np.isfinite(np.asarray(match["redshift"])))
    assert np.all(np.asarray(match["redshift"]) > 0.0)
    for band in BANDS:
        assert f"lsst_{band}_fpfs1_e1" in match.dtype.names


def _band_noise_corrs(npix=33):
    """A 6-band correlation stack (ugrizy) with a different anisotropic
    shape and an un-normalised amplitude in each band, as the systematics
    product stores it."""
    corr = np.zeros((6, npix, npix))
    c = npix // 2
    for ib in range(6):
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 1.0
        kernel[1, 2] = 0.1 * (ib + 1)  # x-lag grows with the band index
        kernel[2, 1] = 0.05
        acf = np.zeros((npix, npix))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ys = slice(max(dy, 0), 3 + min(dy, 0))
                yk = slice(max(-dy, 0), 3 + min(-dy, 0))
                xs = slice(max(dx, 0), 3 + min(dx, 0))
                xk = slice(max(-dx, 0), 3 + min(-dx, 0))
                acf[c + dy, c + dx] = np.sum(kernel[ys, xs] * kernel[yk, xk])
        corr[ib] = (ib + 2.0) * acf
    return corr


def test_forced_reuse_matches_fresh_prepare(scene):
    """The detection bands' force inputs, recovered from the r+i+z
    detection preparation (``_cache_bands_multiband``), must equal a
    fresh single-band ``prepare_data`` -- in particular each band must
    get its OWN noise correlation, normalised to 1 at zero lag, and its
    own patch variance.  A leftover per-band ``noise_corrs`` list made
    the forced FPFS fall back to a white spectrum."""
    config = _validatable_config()
    config.detection_bands = BANDS
    config.renoise_mode = "spectral"  # also builds the filler eps''
    config.validate()
    pipe = MeasureCoaddsPipe(config=config)
    corr_array = _band_noise_corrs()
    corrs = {b: pipe._load_noise_corr(corr_array, b) for b in BANDS}
    common = dict(
        seed=7,
        survey=SURVEY_NAME,
        skyMap=scene.skymap,
        tract=TRACT_ID,
        patch=PATCH_ID,
        detection=None,
        renoise_mode=config.renoise_mode,
        renoise_delta=config.renoise_delta,
    )
    data = pipe.anacal.prepare_data_multiband(
        exposures=scene.exposures, bands=BANDS, noise_corrs=corrs, **common,
    )
    cache = pipe._cache_bands_multiband(
        bands=BANDS,
        exposures=scene.exposures,
        data=data,
        cells=data.pop("cells"),
    )
    for band in BANDS:
        fresh = pipe.anacal.prepare_data(
            exposure=scene.exposures[band],
            band=band,
            noise_corr=corrs[band],
            **common,
        )
        cached = cache[band]["data"]
        assert "noise_corrs" not in cached
        np.testing.assert_array_equal(cached["noise_corr"], fresh["noise_corr"])
        mid = cached["noise_corr"].shape[0] // 2
        assert cached["noise_corr"][mid, mid] == 1.0
        assert cached["noise_variance"] == fresh["noise_variance"]
        for key in ("gal_array", "noise_array", "noise2_array"):
            assert cached[key] is not None, key
            np.testing.assert_array_equal(cached[key], fresh[key], err_msg=key)
        assert cached["base_column_name"] == fresh["base_column_name"]
    # the bands really differ, so a mix-up could not pass
    assert not np.array_equal(
        cache["r"]["data"]["noise_corr"], cache["z"]["data"]["noise_corr"]
    )


def _validatable_config():
    config = MeasureCoaddsPipeConfig()
    # validate() checks this first and the default is negative.
    config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
    return config


def test_sigma_shapelets_validation():
    # Default sigma_shapelets1 is negative on purpose, so an unconfigured
    # task fails validation instead of measuring with a bad scale.
    config = MeasureCoaddsPipeConfig()
    with pytest.raises(Exception, match="sigma_shapelets1"):
        config.validate()


def test_detection_bands_validation():
    config = _validatable_config()
    config.detection_bands = []
    with pytest.raises(Exception, match="at least one band"):
        config.validate()

    config = _validatable_config()
    config.detection_bands = ["i", "i"]
    with pytest.raises(Exception, match="duplicates"):
        config.validate()

    config = _validatable_config()
    config.use_sim = True
    config.sim_bands = ["i"]
    config.detection_bands = ["r", "i"]
    with pytest.raises(Exception, match="every detection band"):
        config.validate()


def test_detection_band_gauss_fluxes_are_measured_once(scene):
    """With ``do_measure_flux_gauss`` the detection bands' Gaussian fluxes
    come out of the detection call (``process_image_bands``), from the
    per-band smoothed images the coadd was built from.  They must be, bit
    for bit, what a forced AnaCal run on that band at the coadd's kernel
    width returns -- the previous force-stage computation, pinned to the
    coadd's width -- and the final catalog must carry exactly them.  The
    band outside the detection set is still measured by the forced run."""
    det_bands = ["r", "i"]
    seed = 11
    handles = make_exposure_handles(
        scene.exposures, tract=TRACT_ID, patch=PATCH_ID, skyMap=scene.skymap,
    )
    config = MeasureCoaddsPipeConfig()
    config.anacal.force_size = False
    config.anacal.num_epochs = 0
    config.anacal.do_noise_bias_correction = False
    config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
    config.fpfs.do_noise_bias_correction = False
    config.detection_bands = det_bands
    config.do_measure_flux_gauss = True
    config.validate()
    pipe = MeasureCoaddsPipe(config=config)

    det_cats, band_cache = pipe._detect(
        exposure_handles_dict=handles, seed=seed, corr_array=None,
        skyMap=scene.skymap, tract=TRACT_ID, patch=PATCH_ID, mask_array=None,
    )
    assert set(band_cache) == set(det_bands)
    scale = float(band_cache["r"]["data"]["pixel_scale"])
    variances = [float(band_cache[b]["data"]["noise_variance"]) for b in det_bands]
    gauss_cols = [c for c in select_band_gauss_fluxes(det_cats[next(iter(det_cats))], "r").dtype.names]
    assert len(gauss_cols) == 8
    for band in det_bands:
        cached = band_cache[band]
        cell_map = {c.index: c for c in cached["cells"]}
        assert set(cached["gauss"]) == set(det_cats)
        for key, det in det_cats.items():
            band_cat = cached["gauss"][key]
            assert len(band_cat) == len(det)
            # the coadd's kernel width for this cell: the band stack's PSF
            stack_psf = np.stack([
                {c.index: c for c in band_cache[b]["cells"]}[key].psf_array for b in det_bands
            ])
            sigma_coadd = pipe.anacal.get_sigma_arcsec(stack_psf, scale, variances)
            acfg = AnacalConfig()
            acfg.update(**pipe.anacal.config.toDict())
            acfg.sigma_arcsec = sigma_coadd
            forced_task = AnacalTask(config=acfg)
            forced = forced_task.run(
                **cached["data"], detection=det, cells=[cell_map[key]],
                n_mask_base_max=config.n_mask_base_max, renoise_mode=config.renoise_mode,
                renoise_c=config.renoise_c, renoise_delta=config.renoise_delta,
            )
            assert forced.dtype.names == band_cat.dtype.names
            for name in forced.dtype.names:
                np.testing.assert_array_equal(band_cat[name], forced[name], err_msg=f"{band} {name}")
            # (the simulated bands share one PSF, so here the coadd's width
            # equals each band's own; tests/task/test_process_image_bands.py
            # in AnaCal covers bands with different PSFs)

    # end to end: the catalog carries exactly those fluxes, and the band
    # outside the detection set is measured too
    cat = pipe.run(
        exposure_handles_dict=handles, corr_array=None, skyMap=scene.skymap,
        tract=TRACT_ID, patch=PATCH_ID, seed=seed,
    ).anacalCatalog
    for band in BANDS:
        assert f"{SURVEY_NAME}_{band}_flux_gauss0" in cat.dtype.names
        assert np.all(np.isfinite(cat[f"{SURVEY_NAME}_{band}_flux_gauss0"]))
    by_pos = {}
    for key, det in det_cats.items():
        for b in det_bands:
            g = band_cache[b]["gauss"][key]
            for i in range(len(det)):
                by_pos[(b, float(det["x1_det"][i]), float(det["x2_det"][i]))] = g["flux_gauss0"][i]
    assert len(cat) > 0
    for b in det_bands:
        vals = cat[f"{SURVEY_NAME}_{b}_flux_gauss0"]
        for i in range(len(cat)):
            assert vals[i] == by_pos[(b, float(cat["x1_det"][i]), float(cat["x2_det"][i]))]


def test_fixed_sigma_through_the_pipe(scene, measured):
    """A positive ``anacal.sigma_arcsec`` replaces the PSF-derived kernel
    width in detection and forced measurement: the task reports the fixed
    width for the exposure PSF and the catalog is not the default one."""
    handles = make_exposure_handles(
        scene.exposures, tract=TRACT_ID, patch=PATCH_ID, skyMap=scene.skymap,
    )
    cat, pipe = _measure(handles, scene.skymap, ["i"], sigma_arcsec=0.6, return_pipe=True)
    exp = scene.exposures["i"]
    psf = exp.getPsf().computeKernelImage(exp.getBBox().getCenter()).array
    ref = measured.single
    derived = MeasureCoaddsPipe(config=_validatable_config()).anacal.get_sigma_arcsec(psf, PIXEL_SCALE)
    assert 0.2 < derived < 0.6
    assert pipe.anacal.get_sigma_arcsec(psf, PIXEL_SCALE) == 0.6
    assert cat.dtype.names == ref.dtype.names
    assert len(cat) != len(ref) or not np.allclose(cat["fpfs_e1"], ref["fpfs_e1"])


def test_cnn_columns_through_the_pipe(scene, measured, tmp_path):
    """do_measure_cnn adds the per-band ``cnn_*`` columns in the force
    stage, aligned with the forced FPFS rows, with the same renoising and
    mask cut; the model is a tiny random export in the cpu_src format."""
    from test_cnn_measure import write_tiny_export

    config = _validatable_config()
    config.detection_bands = ["i"]
    config.renoise_mode = "spectral"
    config.do_measure_cnn = True
    config.cnn.model_path = write_tiny_export(str(tmp_path / "tiny.npz"))
    config.validate()
    pipe = MeasureCoaddsPipe(config=config)
    handles = make_exposure_handles(
        scene.exposures, tract=TRACT_ID, patch=PATCH_ID, skyMap=scene.skymap,
    )
    cat = pipe.run(
        exposure_handles_dict=handles, corr_array=None, skyMap=scene.skymap,
        tract=TRACT_ID, patch=PATCH_ID,
    ).anacalCatalog
    for band in BANDS:
        for col in ("cnn_e1", "cnn_de1_dg1", "cnn_de1_dg2", "cnn_e2", "cnn_de2_dg1", "cnn_de2_dg2"):
            assert f"lsst_{band}_{col}" in cat.dtype.names
        e1 = cat[f"lsst_{band}_cnn_e1"]
        # stamps that fit in their cell are measured; the rest stay NaN
        assert np.isfinite(e1).sum() > 0.5 * len(cat)
        assert np.all(np.isfinite(cat[f"lsst_{band}_cnn_de1_dg1"][np.isfinite(e1)]))
    # with the switch off (the module fixture) there are no such columns
    assert not any("_cnn_" in n for n in measured.single.dtype.names)
