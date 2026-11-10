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


def test_total_variance_array_agrees_with_defaults_at_the_tabulated_zp():
    """``get_total_variance_array`` is ``defaults.get_variance_array``
    generalised off the tabulated zero point.  At that zero point the two
    must agree exactly, which keeps the defaults module the single
    definition of the noise model."""
    from xlens.simulator.defaults import (
        get_variance_array,
        noise_variance_defaults,
    )
    from xlens.simulator.noise import get_total_variance_array

    sky = noise_variance_defaults[BAND]["lsst"]
    flux = np.array([[0.0, 1.0], [10.0, 1000.0]])
    got = get_total_variance_array(image_array=flux, band=BAND,
                                   survey_name="lsst", sky_variance=sky,
                                   mag_zero=30.0)
    np.testing.assert_allclose(got, get_variance_array(flux, BAND, "lsst"),
                               rtol=1e-12)
    # a blank pixel carries the sky and nothing else
    np.testing.assert_allclose(got[0, 0], sky, rtol=1e-12)
    # negative pixels cannot drive the variance below the sky
    assert np.all(
        get_total_variance_array(image_array=np.array([[-5.0]]), band=BAND,
                                 survey_name="lsst", sky_variance=sky,
                                 mag_zero=30.0) == sky
    )


def test_source_noise_rescales_with_the_zero_point():
    """``c`` is calibrated at ZP=30, so a different mag_zero must rescale it.

    The sharp statement: with the same seeds, changing the zero point is a
    pure rescaling of the whole exposure -- galaxies, sky noise and photon
    noise alike -- by 10**(0.4 dz) in the image and 10**(0.8 dz) in the
    variance.  Were the photon term left as the naive ``c * f_Z``, the image
    would scale by LESS than 10**(0.4 dz) exactly in the bright pixels, and
    the ratio's spread would give it away.
    """
    from xlens.simulator.defaults import (
        noise_variance_defaults,
        source_noise_coeff_defaults,
    )
    from xlens.simulator.noise import get_total_variance_array

    dz = 2.0
    c = source_noise_coeff_defaults[BAND]["lsst"]
    sky = noise_variance_defaults[BAND]["lsst"]
    f0 = np.array([[0.0, 1.0, 50.0, 5000.0]])

    v0 = get_total_variance_array(image_array=f0, band=BAND,
                                  survey_name="lsst", sky_variance=sky,
                                  mag_zero=30.0)
    # the SAME physical scene at another zero point: its flux is up by
    # 10**(0.4 dz) and its sky variance by 10**(0.8 dz)
    vz = get_total_variance_array(
        image_array=f0 * 10.0 ** (0.4 * dz), band=BAND, survey_name="lsst",
        sky_variance=sky * 10.0 ** (0.8 * dz), mag_zero=30.0 + dz,
    )
    np.testing.assert_allclose(vz, v0 * 10.0 ** (0.8 * dz), rtol=1e-12)
    # it is the COEFFICIENT that is rescaled, by 10**(0.4 dz) because c is a
    # gain; leaving c alone would be short by exactly that factor
    naive = sky * 10.0 ** (0.8 * dz) + c * f0 * 10.0 ** (0.4 * dz)
    assert not np.allclose(vz[0, 1:], naive[0, 1:], rtol=1e-3)

    tract_info = _skymap()[0]
    truth = _truth(tract_info)

    def _sim(mag_zero):
        cfg = MultibandSimConfig()
        cfg.survey_name = "lsst"
        cfg.draw_image_noise = True
        cfg.draw_source_noise = True
        cfg.mag_zero = mag_zero
        cfg.truncate_stamp_size = 65
        cfg.include_point_source = False
        return MultibandSimTask(config=cfg).run(
            tract_info=tract_info, patch_id=0, band=BAND, seed=SEED,
            truthCatalog=truth).simExposure

    a, b = _sim(-1.0), _sim(30.0 + dz)          # -1 means the default, 30
    ia = a.image.array.astype(np.float64)
    ib = b.image.array.astype(np.float64)
    big = np.abs(ia) > 1e-3 * np.abs(ia).max()
    assert big.sum() > 1000
    # 2e-5 is the float32 storage floor, not physics
    np.testing.assert_allclose(ib[big] / ia[big], 10.0 ** (0.4 * dz),
                               rtol=2e-5)

    va = a.getMaskedImage().variance.array
    vb = b.getMaskedImage().variance.array
    sky = noise_variance_defaults[BAND]["lsst"]
    assert np.unique(vb).size == 1
    np.testing.assert_allclose(vb.flat[0],
                               sky * 10.0 ** (0.8 * dz), rtol=1e-6)
    np.testing.assert_allclose(vb.flat[0] / va.flat[0],
                               10.0 ** (0.8 * dz), rtol=1e-6)


def test_source_noise_is_one_field_scaled_by_sky_plus_photon():
    """One white field scaled by sqrt(sky + c * flux), not two fields.

    Consequences checked here: the noise variance is sky + c * flux per
    pixel; blank sky comes out bit-identical to a sky-only run for the same
    seed (numpy's normal(scale=s) is s * gauss off the same stream, so a
    pixel whose variance is unchanged draws the same number); and the
    variance plane is the constant sky value either way.
    """
    from xlens.simulator.defaults import (
        noise_variance_defaults,
        source_noise_coeff_defaults,
    )

    tract_info = _skymap()[0]
    truth = _truth(tract_info)
    quiet = _simulate(tract_info, truth, draw_noise=False)
    flux = quiet.image.array.copy()

    sky_only = _add_noise(quiet.clone())
    with_src = _add_noise(quiet.clone(), draw_source_noise=True)

    extra = with_src.image.array - sky_only.image.array
    blank = flux <= 0.0
    assert blank.sum() > 0
    np.testing.assert_array_equal(extra[blank], 0.0)
    assert np.any(extra[~blank] != 0.0)

    # the plane is the constant sky variance in both cases -- the estimator
    # must never be handed the photon term
    sky = noise_variance_defaults[BAND]["lsst"]
    for exp in (sky_only, with_src):
        var = exp.getMaskedImage().variance.array
        assert np.all(var == var.flat[0])
        np.testing.assert_allclose(var.flat[0], sky, rtol=1e-6)

    # the TOTAL noise has variance sky + c * flux (one field, so this is the
    # statement -- the difference between the two runs is NOT sqrt(c * flux))
    c = source_noise_coeff_defaults[BAND]["lsst"]
    noise = (with_src.image.array - flux).astype(np.float64)
    f64 = flux.astype(np.float64)
    bright = f64 > 10.0 * sky
    assert bright.sum() > 50
    ratio = (noise[bright] ** 2 / (sky + c * f64[bright])).mean()
    assert abs(ratio - 1.0) < 0.3, ratio
    # and over blank sky it is the sky variance alone
    ratio_blank = (noise[blank] ** 2 / sky).mean()
    assert abs(ratio_blank - 1.0) < 0.05, ratio_blank


def test_source_noise_refuses_correlated_noise():
    """The two are not combinable: a stationary correlated field cannot be
    rescaled pixel by pixel without destroying its correlation."""
    from lsst.afw.image import ImageF

    from xlens.simulator.defaults import noise_variance_defaults

    tract_info = _skymap()[0]
    truth = _truth(tract_info)
    quiet = _simulate(tract_info, truth, draw_noise=False)

    corr = ImageF(5, 5)
    corr.getArray()[2, 2] = noise_variance_defaults[BAND]["lsst"]

    cfg = AddNoisePipeConfig()
    cfg.survey_name = "lsst"
    cfg.draw_source_noise = True
    cfg.connections.noiseId = cfg.noiseId
    cfg.validate()
    try:
        AddNoisePipe(config=cfg).run(exposure=quiet, seed=SEED, band=BAND,
                                     noiseCorrImage=corr)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "source photon noise with a correlation kernel must be refused")


def test_sim_source_noise_matches_add_noise_source_noise():
    """draw_source_noise in the simulator and in AddNoisePipe are the same
    realisation, as draw_image_noise and AddNoisePipe already are."""
    tract_info = _skymap()[0]
    truth = _truth(tract_info)

    cfg = MultibandSimConfig()
    cfg.survey_name = "lsst"
    cfg.draw_image_noise = True
    cfg.draw_source_noise = True
    cfg.truncate_stamp_size = 65
    cfg.include_point_source = False
    noisy = MultibandSimTask(config=cfg).run(
        tract_info=tract_info, patch_id=0, band=BAND, seed=SEED,
        truthCatalog=truth).simExposure

    quiet = _simulate(tract_info, truth, draw_noise=False)
    renoised = _add_noise(quiet, draw_source_noise=True)
    np.testing.assert_array_equal(renoised.image.array, noisy.image.array)


def test_sim_variance_plane_is_constant_sky_only_with_source_noise():
    """What the processor reads must know nothing about the photon term.

    ``estimate_noise_variance`` collapses this plane to a single float that
    feeds both anacal's noise-bias correction and the renoising spectrum, and
    those model stationary SKY noise.  So the plane the simulator writes has
    to stay exactly constant at the sky variance even though the noise in the
    image is now non-stationary.
    """
    from xlens.simulator.defaults import (
        noise_variance_defaults,
        source_noise_coeff_defaults,
    )

    tract_info = _skymap()[0]
    truth = _truth(tract_info)

    def _sim(source):
        cfg = MultibandSimConfig()
        cfg.survey_name = "lsst"
        cfg.draw_image_noise = True
        cfg.draw_source_noise = source
        cfg.truncate_stamp_size = 65
        cfg.include_point_source = False
        return MultibandSimTask(config=cfg).run(
            tract_info=tract_info, patch_id=0, band=BAND, seed=SEED,
            truthCatalog=truth).simExposure

    off, on = _sim(False), _sim(True)
    v_off = off.getMaskedImage().variance.array
    v_on = on.getMaskedImage().variance.array
    sky = noise_variance_defaults[BAND]["lsst"]

    assert np.unique(v_on).size == 1, "the plane must be exactly constant"
    np.testing.assert_allclose(v_on.flat[0], sky, rtol=1e-6)
    np.testing.assert_array_equal(v_on, v_off)
    # the photon term is not hiding in the plane: were it there, the peak
    # pixel would have lifted the maximum by c * peak_flux
    c = source_noise_coeff_defaults[BAND]["lsst"]
    peak = float(off.image.array.max())
    assert c * peak > 1e-3 * sky, "pick a brighter scene; this proves nothing"
    assert v_on.max() <= sky * (1.0 + 1e-6)
    # control: the flag did change the image, so the above is not vacuous
    assert not np.array_equal(on.image.array, off.image.array)


def test_source_noise_requires_image_noise_and_lsst():
    from lsst.pex.config import FieldValidationError

    cfg = MultibandSimConfig()
    cfg.survey_name = "lsst"
    cfg.draw_image_noise = False
    cfg.draw_source_noise = True
    try:
        cfg.validate()
    except FieldValidationError:
        pass
    else:
        raise AssertionError("draw_source_noise without draw_image_noise must be refused")

    cfg = AddNoisePipeConfig()
    cfg.survey_name = "hsc"
    cfg.draw_source_noise = True
    cfg.connections.noiseId = cfg.noiseId
    try:
        cfg.validate()
    except FieldValidationError:
        pass
    else:
        raise AssertionError("source photon noise is lsst-only and must be refused elsewhere")
