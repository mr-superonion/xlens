"""The renoise_mode switch of AnacalTask: 'rotate' (AnaCal's rotated-noise
correction) and 'spectral' (minimal-variance renoising from the patch noise
correlation function) run on the same scene through the same entry point.
"""

import anacal
import galsim
import numpy as np
import pytest
from lsst.pex.config import FieldValidationError

from xlens.processor.anacal import AnacalConfig, AnacalTask
from xlens.processor.measure_coadds import MeasureCoaddsPipeConfig
from xlens.utils.image.noise import generate_filler_array

SCALE = 0.2
MAG_ZERO = 30.0


def _psf(e1=0.1, e2=0.03, n=64):
    """Moffat PSF (beta 3.5, FWHM 0.6") on pixel (n // 2, n // 2).  A
    Moffat, not a Gaussian: with the metadetect-fitgauss re-smoothing
    scale (PSF size times at most 1.1) a Gaussian PSF's major axis is
    wider than the smoothing kernel, so h / p grows with k and both
    renoising options become very noisy."""
    obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=2.4).shear(e1=e1, e2=e2)
    im = obj.shift(0.5 * SCALE, 0.5 * SCALE).drawImage(nx=n, ny=n, scale=SCALE)
    return np.asarray(im.array, dtype=np.float64)


def _scene(psf, ngrid=64, nstamp=8, mag=22.5, seed=3):
    """nstamp x nstamp Gaussian galaxies (observed size 0.45") on a grid,
    convolved with the PSF via FFT, plus white noise."""
    n = ngrid * nstamp
    y, x = (np.mgrid[0:ngrid, 0:ngrid] - ngrid // 2) * SCALE
    gal = np.exp(-0.5 * (x * x + y * y) / 0.45**2)
    gal *= 10 ** ((MAG_ZERO - mag) / 2.5) / gal.sum()
    img = np.tile(gal, (nstamp, nstamp))
    kpsf = np.fft.rfft2(np.fft.ifftshift(psf), s=img.shape)
    img = np.fft.irfft2(np.fft.rfft2(img) * kpsf, s=img.shape)
    rng = np.random.default_rng(seed)
    noise_std = 0.5
    img = (img + rng.normal(0.0, noise_std, img.shape)).astype(np.float32)
    noise = rng.normal(0.0, noise_std, img.shape).astype(np.float32)
    return img, noise, noise_std**2, n


def _run(task, img, psf, noise, var, cells, **extra):
    return task.run(
        pixel_scale=SCALE,
        mag_zero=MAG_ZERO,
        noise_variance=var,
        gal_array=img,
        psf_array=psf,
        mask_array=np.zeros(img.shape, dtype=np.uint8),
        noise_array=noise,
        detection=None,
        cells=cells,
        **extra,
    )


def test_config_validation():
    cfg = MeasureCoaddsPipeConfig()
    cfg.fpfs.sigma_shapelets1 = 0.52  # the pipe's own validation needs it
    assert cfg.renoise_mode == "rotate"
    cfg.renoise_mode = "spectral"
    cfg.validate()
    cfg.renoise_mode = "other"
    with pytest.raises(FieldValidationError):
        cfg.validate()
    cfg.renoise_mode = "spectral"
    cfg.renoise_c = 0.0
    with pytest.raises(FieldValidationError):
        cfg.validate()
    # the subtasks take the mode as a run() argument, not a config field
    assert not hasattr(AnacalConfig(), "renoise_mode")


def test_spectral_switch_runs_and_matches_rotate():
    psf = _psf()
    img, noise, var, n = _scene(psf)
    # the spectral mode measures the image WITH the spin-2 filler of the
    # patch PSF (prepare_data adds it; here by hand)
    filler = generate_filler_array(
        ny=n,
        nx=n,
        pixel_scale=SCALE,
        seed=7,
        band="i",
        noise_variance=var,
        psf_array=psf,
        noise_corr=None,
    )
    assert filler.shape == img.shape and filler.dtype == np.float32
    # the filler carries the high-k power the smoothing kernel removes
    assert 0.03 < np.std(filler) / np.sqrt(var) < 1.0
    img_s = img + filler
    cells = anacal.geometry.get_cell_list(
        img_nx=n,
        img_ny=n,
        cell_nx=256,
        cell_ny=256,
        cell_overlap=64,
        scale=SCALE,
    )
    task = AnacalTask(config=AnacalConfig())
    cat_rot = _run(task, img, psf, noise, var, cells, renoise_mode="rotate")
    sp = {"renoise_mode": "spectral", "filler_added": True}
    cat_spec = _run(task, img_s, psf, noise, var, cells, **sp)
    # a (normalised) correlation function of white noise gives the same
    # result as none, through the per-band list and the single entry
    corr = np.zeros((9, 9))
    corr[4, 4] = 1.0
    cat_corr = _run(task, img_s, psf, noise, var, cells, noise_corr=corr, **sp)
    cat_corrs = _run(task, img_s, psf, noise, var, cells, noise_corrs=[corr], **sp)
    # Option 1 through the patch spectrum: the white correlation function
    # reproduces the plain white run, a correlated one changes the flux
    # errors (summed over S^d + S^d(R k)) but not the measurement
    rot = {"renoise_mode": "rotate"}
    cat_rot_w = _run(task, img, psf, noise, var, cells, noise_corr=corr, **rot)
    assert np.array_equal(cat_rot, cat_rot_w)
    corr2 = corr.copy()
    corr2[4, 5] = corr2[4, 3] = 0.3
    corr2[5, 4] = corr2[3, 4] = 0.1
    cat_rot_c = _run(task, img, psf, noise, var, cells, noise_corr=corr2, **rot)
    assert np.array_equal(cat_rot["e1"], cat_rot_c["e1"])
    assert not np.allclose(cat_rot["flux_gauss0_err"], cat_rot_c["flux_gauss0_err"])
    for cat in (cat_rot, cat_spec, cat_corr, cat_corrs):
        assert len(cat) > 20
        assert np.all(np.isfinite(cat["e1"]))
        assert np.all(np.isfinite(cat["de1_dg1"]))
    assert np.array_equal(cat_spec["e1"], cat_corr["e1"])
    assert np.array_equal(cat_spec["e1"], cat_corrs["e1"])
    # same sources (same image noise), different added noise
    assert abs(len(cat_spec) - len(cat_rot)) < 0.2 * len(cat_rot)
    assert not np.array_equal(cat_spec["e1"][:10], cat_rot["e1"][:10])
    # the spectral mode adds much less noise than the rotated copy
    r = np.median(cat_spec["flux_gauss0_err"]) / np.median(cat_rot["flux_gauss0_err"])
    assert 0.7 < r < 0.85, r
    # without the pure-noise image both modes are the plain measurement
    cat0 = _run(task, img, psf, None, var, cells, renoise_mode="spectral")
    cat1 = _run(task, img, psf, None, var, cells)
    assert np.array_equal(cat0, cat1)
    with pytest.raises(ValueError):
        _run(task, img, psf, noise, var, cells, renoise_mode="other")
    # the spectral mode refuses an image without the filler
    with pytest.raises(ValueError):
        _run(task, img, psf, noise, var, cells, renoise_mode="spectral")
