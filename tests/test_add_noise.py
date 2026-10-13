"""AddNoisePipe reproduces the simulator's own noise.

A noiseless simulation (``draw_image_noise=False``) still carries the survey
noise variance in its variance plane; adding the noise afterwards with
``AddNoisePipe`` for the same seed must give, bit for bit, the exposure that
``draw_image_noise=True`` produces.  That is what lets the images be stored
noiseless and re-noised (also with other ``noiseId``) without re-rendering.
"""
import numpy as np
from lsst.skymap.discreteSkyMap import DiscreteSkyMap, DiscreteSkyMapConfig

from xlens.simulator.catalog import CatalogShearTask, CatalogShearTaskConfig
from xlens.simulator.noise import AddNoisePipe, AddNoisePipeConfig
from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask

SEED, BAND = 7, "i"


def _skymap():
    config = DiscreteSkyMapConfig()
    config.raList = [0.0]
    config.decList = [0.0]
    config.radiusList = [0.1]
    config.rotation = 0.0
    config.projection = "TAN"
    config.patchInnerDimensions = [300, 300]
    config.patchBorder = 0
    config.pixelScale = 0.2
    config.tractOverlap = 0.0
    return DiscreteSkyMap(config)


def _simulate(tract_info, truth, draw_noise):
    cfg = MultibandSimConfig()
    cfg.survey_name = "lsst"
    cfg.draw_image_noise = draw_noise
    cfg.truncate_stamp_size = 65
    cfg.include_point_source = False
    task = MultibandSimTask(config=cfg)
    return task.run(tract_info=tract_info, patch_id=0, band=BAND, seed=SEED,
                    truthCatalog=truth).simExposure


def _truth(tract_info):
    cfg = CatalogShearTaskConfig()
    cfg.z_bounds = [0.0, 0.63, 0.98, 1.48, 10.0]
    cfg.mode = 0
    cfg.rotId = 0
    cfg.kappa_value = 0.0
    cfg.test_value = 0.02
    cfg.test_target = "g1"
    cfg.layout = "random"
    cfg.extend_ratio = 1.0
    cfg.sep_arcsec = 14
    return CatalogShearTask(config=cfg).run(tract_info=tract_info, seed=SEED).truthCatalog


def _add_noise(exposure, **overrides):
    cfg = AddNoisePipeConfig()
    cfg.survey_name = "lsst"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.connections.noiseId = cfg.noiseId      # the output name carries the realisation
    cfg.validate()
    return AddNoisePipe(config=cfg).run(exposure=exposure, seed=SEED, band=BAND).simExposure


def test_noiseless_plus_add_noise_equals_noisy_simulation():
    tract_info = _skymap()[0]
    truth = _truth(tract_info)
    noisy = _simulate(tract_info, truth, draw_noise=True)
    quiet = _simulate(tract_info, truth, draw_noise=False)

    # the noiseless image carries the full noise variance in its plane
    var = quiet.getMaskedImage().variance.array
    assert np.all(var > 0.0)
    np.testing.assert_array_equal(var, noisy.getMaskedImage().variance.array)
    # and really is noiseless: its pixels are the galaxies alone
    assert quiet.image.array.std() < noisy.image.array.std()

    renoised = _add_noise(quiet)
    np.testing.assert_array_equal(renoised.image.array, noisy.image.array)
    np.testing.assert_array_equal(renoised.getMaskedImage().variance.array,
                                  noisy.getMaskedImage().variance.array)


def test_add_noise_sets_variance_plane():
    """An input without a variance plane gets the noise variance written in
    (and the same realisation, since the variance comes from the config,
    not from the plane); a wrong plane is refused."""
    tract_info = _skymap()[0]
    truth = _truth(tract_info)
    quiet = _simulate(tract_info, truth, draw_noise=False)
    ref = _add_noise(quiet.clone())
    blank = quiet.clone()
    blank.getMaskedImage().variance.array[:, :] = 0.0
    out = _add_noise(blank)
    np.testing.assert_array_equal(out.getMaskedImage().variance.array,
                                  ref.getMaskedImage().variance.array)
    np.testing.assert_array_equal(out.image.array, ref.image.array)
    wrong = quiet.clone()
    wrong.getMaskedImage().variance.array[:, :] *= 4.0
    try:
        _add_noise(wrong)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a variance plane that disagrees with the noise variance must be refused")


def test_add_noise_other_realisations_differ():
    tract_info = _skymap()[0]
    truth = _truth(tract_info)
    quiet = _simulate(tract_info, truth, draw_noise=False)
    base = _add_noise(quiet.clone()).image.array
    other_id = _add_noise(quiet.clone(), noiseId=1).image.array
    with_survey = _add_noise(quiet.clone(), survey_in_seed=True).image.array
    assert not np.array_equal(base, other_id)
    assert not np.array_equal(base, with_survey)
    # same noise level either way
    assert abs((other_id - quiet.image.array).std() / (base - quiet.image.array).std() - 1) < 0.05
