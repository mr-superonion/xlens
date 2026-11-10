import anacal
import numpy as np
import pytest
from lsst.pex.config import FieldValidationError
from xlens.processor.anacal import AnacalConfig, AnacalTask

SCALE = 0.2


def _psf(cxx, cyy, cxy, n=64):
    y, x = np.mgrid[0:n, 0:n] * SCALE
    dx, dy = x - (n // 2) * SCALE, y - (n // 2) * SCALE
    det = cxx * cyy - cxy**2
    im = np.exp(
        -0.5 * (cyy * dx * dx - 2 * cxy * dx * dy + cxx * dy * dy) / det
    )
    return im / im.sum()


def test_sigma_is_metadetect_fitgauss():
    task = AnacalTask(config=AnacalConfig())
    # round PSF: d = 1, sigma is the PSF's own Gaussian width
    np.testing.assert_allclose(
        task.get_sigma_arcsec(_psf(0.18, 0.18, 0.0), SCALE),
        np.sqrt(0.18), rtol=1e-6,
    )
    # elliptical: sqrt(T d / 2), d from the PSF ellipticity
    cxx, cyy, cxy = 0.21, 0.15, 0.03
    T = cxx + cyy
    d = anacal.ngmix.ellip_dilation((cxx - cyy) / T, 2 * cxy / T, 1.1)
    np.testing.assert_allclose(
        task.get_sigma_arcsec(_psf(cxx, cyy, cxy), SCALE),
        np.sqrt(T * d / 2), rtol=1e-6,
    )


def test_sigma_band_stack_is_weighted_average():
    task = AnacalTask(config=AnacalConfig())
    stack = np.stack([_psf(0.16, 0.16, 0.0), _psf(0.24, 0.24, 0.0)])
    # inverse-variance weights 3:1 on T
    s = task.get_sigma_arcsec(stack, SCALE, noise_variance=[1.0, 3.0])
    np.testing.assert_allclose(
        s, np.sqrt(0.75 * 0.16 + 0.25 * 0.24), rtol=1e-6
    )


def test_dilation_cap_config():
    cfg = AnacalConfig()
    cfg.psf_dilation_max = 1.0
    task = AnacalTask(config=cfg)
    np.testing.assert_allclose(
        task.get_sigma_arcsec(_psf(0.21, 0.15, 0.03), SCALE),
        np.sqrt(0.36 / 2), rtol=1e-6,
    )
    cfg.psf_dilation_max = 0.9
    with pytest.raises(FieldValidationError):
        cfg.validate()


def test_fixed_sigma_override(caplog):
    cfg = AnacalConfig()
    assert cfg.sigma_arcsec < 0  # PSF-derived by default
    cfg.sigma_arcsec = 0.55
    task = AnacalTask(config=cfg)
    # wider than the PSF-derived width (sqrt(0.18) = 0.424): used as is,
    # for a single PSF and for a band stack alike
    assert task.get_sigma_arcsec(_psf(0.18, 0.18, 0.0), SCALE) == 0.55
    stack = np.stack([_psf(0.16, 0.16, 0.0), _psf(0.24, 0.24, 0.0)])
    assert task.get_sigma_arcsec(stack, SCALE, noise_variance=[1.0, 3.0]) == 0.55
    # narrower than the PSF-derived width: floored at it, warned once
    cfg = AnacalConfig()
    cfg.sigma_arcsec = 0.3
    task = AnacalTask(config=cfg)
    with caplog.at_level("WARNING"):
        s = task.get_sigma_arcsec(_psf(0.18, 0.18, 0.0), SCALE)
        np.testing.assert_allclose(s, np.sqrt(0.18), rtol=1e-6)
        s = task.get_sigma_arcsec(_psf(0.24, 0.24, 0.0), SCALE)
        np.testing.assert_allclose(s, np.sqrt(0.24), rtol=1e-6)
        assert task.get_sigma_arcsec(_psf(0.04, 0.04, 0.0), SCALE) == 0.3
    assert sum("narrower than the PSF-derived" in r.message for r in caplog.records) == 1
    # validation: 0 < sigma < 0.05 arcsec is a mistake
    cfg.sigma_arcsec = 0.01
    with pytest.raises(FieldValidationError):
        cfg.validate()
    cfg.sigma_arcsec = -1.0
    cfg.validate()


def test_psf_moments_are_memoised(monkeypatch):
    task = AnacalTask(config=AnacalConfig())
    calls = []
    real = anacal.ngmix.fit_psf_gauss

    def counting(psf, scale, *a, **k):
        calls.append(1)
        return real(psf, scale, *a, **k)

    monkeypatch.setattr(anacal.ngmix, "fit_psf_gauss", counting)
    stack = np.stack([_psf(0.16, 0.16, 0.0), _psf(0.24, 0.24, 0.0)])
    s1 = task.get_sigma_arcsec(stack, SCALE, noise_variance=[1.0, 3.0])
    assert len(calls) == 2
    # a byte-identical copy of one band (what the forced stage hands
    # over) is a cache hit; a different stamp is not
    s2 = task.get_sigma_arcsec(stack[1].copy(), SCALE)
    assert len(calls) == 2
    np.testing.assert_allclose(s2, np.sqrt(0.24), rtol=1e-6)
    task.get_sigma_arcsec(_psf(0.2, 0.2, 0.0), SCALE)
    assert len(calls) == 3
    np.testing.assert_allclose(s1, task.get_sigma_arcsec(stack, SCALE, noise_variance=[1.0, 3.0]))
    assert len(calls) == 3
