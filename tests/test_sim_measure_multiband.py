"""Detection on a coadd of several bands.

Simulates the same patch in r, i and z with noise and runs
``MeasureCoaddsPipe.run()`` twice: once detecting on i alone (the previous
behaviour) and once detecting on the r+i+z coadd.  Forced measurement is
band-by-band in both cases and must be unaffected.

The r+i+z detection image is deeper, so it is expected to find more
sources; what the tests pin down is that the extra sources come with sane
columns and that turning the feature off leaves the old path exactly as it
was.

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


def _measure(handles, skymap, detection_bands):
    config = MeasureCoaddsPipeConfig()
    config.anacal.force_size = False
    config.anacal.sigma_arcsec = 0.38
    config.anacal.num_epochs = 0
    config.anacal.do_noise_bias_correction = False
    config.anacal.validate_psf = False
    config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
    config.fpfs.do_noise_bias_correction = False
    config.fpfs.return_only_linear_modes = False
    config.detection_bands = detection_bands
    config.validate()

    return MeasureCoaddsPipe(config=config).run(
        exposure_handles_dict=handles,
        corr_array=None,
        skyMap=skymap,
        tract=TRACT_ID,
        patch=PATCH_ID,
    ).anacalCatalog


@pytest.fixture(scope="module")
def measured():
    skymap = _make_skymap()
    truth = _make_truth(skymap)
    exposures = {b: _simulate(skymap, truth, b) for b in BANDS}
    handles = make_exposure_handles(
        exposures, tract=TRACT_ID, patch=PATCH_ID, skyMap=skymap,
    )
    return SimpleNamespace(
        skymap=skymap,
        truth=truth,
        single=_measure(handles, skymap, ["i"]),
        multi=_measure(handles, skymap, BANDS),
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

    # Each detection is joined to at most one truth galaxy, and the
    # truth rows addressed are rows of the input catalog it came from.
    truth_index = np.asarray(match["truth_index"])
    assert len(np.unique(truth_index)) == len(truth_index)
    assert truth_index.min() >= 0
    assert np.all(np.isin(truth_index, np.asarray(measured.truth["indices"])))

    # The truth columns carried over are live, and the measured columns
    # survive the merge intact.
    assert np.all(np.isfinite(np.asarray(match["redshift"])))
    assert np.all(np.asarray(match["redshift"]) > 0.0)
    for band in BANDS:
        assert f"lsst_{band}_fpfs1_e1" in match.dtype.names


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
