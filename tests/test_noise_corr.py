"""Noise-correlation estimate and its normalization: the large-lag
pedestal left by background structure beyond the stamp is removed, so
the correlation vanishes at the stamp edge and has a positive spectrum
(the spectral renoising divides by it)."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from xlens.processor.measure_coadds import MeasureCoaddsPipe
from xlens.processor.systematics_base import BuildSystematicsTaskBase
from xlens.utils.image import remove_noise_corr_pedestal

NPIX = 49


def _spectrum(corr, n=256):
    """Spectrum of the stamp zero-padded on an n x n grid."""
    half = corr.shape[0] // 2
    big = np.zeros((n, n))
    big[n // 2 - half : n // 2 + half + 1, n // 2 - half : n // 2 + half + 1] = corr
    return np.fft.fft2(np.fft.ifftshift(big)).real


def _gaussian_corr(npix=NPIX, sigma=1.2):
    yy, xx = np.indices((npix, npix)) - npix // 2
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


def _correlated_noise(rng, shape, kernel_sigma=1.2):
    """Stationary noise with a compact Gaussian correlation."""
    white = rng.standard_normal(shape)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    kern = np.exp(-2 * np.pi**2 * kernel_sigma**2 * (kx**2 + ky**2))
    return np.fft.irfft2(np.fft.rfft2(white) * kern, s=shape)


def test_pedestal_removal_recovers_compact_correlation():
    corr = _gaussian_corr()
    out, pedestal = remove_noise_corr_pedestal(corr + 0.02)
    assert pedestal == pytest.approx(0.02, abs=1e-6)
    np.testing.assert_allclose(out, corr, atol=1e-6)
    # a compact correlation is left alone
    out, pedestal = remove_noise_corr_pedestal(corr)
    assert abs(pedestal) < 1e-6
    assert out.dtype == corr.dtype


def test_pedestal_turns_spectrum_negative_and_removal_restores_it():
    """A pedestal that has not decayed at the stamp edge is cut by the box,
    whose spectrum has negative side lobes."""
    corr = _gaussian_corr()
    spec = _spectrum(corr + 0.02)
    assert spec.min() < -1e-2 * spec.max()
    fixed, _ = remove_noise_corr_pedestal(corr + 0.02)
    spec = _spectrum(fixed)
    assert spec.min() > 0


def test_correlate_removes_background_offset():
    """Estimator on noise plus a DC offset and a large-scale gradient:
    the stamp edge is zero and the spectrum positive."""
    rng = np.random.default_rng(3)
    shape = (600, 600)
    noise = _correlated_noise(rng, shape)
    yy, xx = np.indices(shape)
    # DC offset (a flat mu^2 pedestal) plus a mild gradient, whose
    # autocorrelation is a shallow quadratic in the lag
    noise += 0.3 + 0.1 * (xx - shape[1] / 2) / shape[1]
    window = np.ones(shape, dtype=np.float32)
    noise[~window.astype(bool)] = 0.0
    corr = BuildSystematicsTaskBase._correlate(noise.astype(np.float32), window, NPIX)
    corr = corr / corr[NPIX // 2, NPIX // 2]
    radius = np.hypot(*(np.indices(corr.shape) - NPIX // 2))
    assert abs(corr[radius >= 0.8 * (NPIX // 2)].mean()) < 1e-6
    # the edge row scatters at the sample-noise level of the estimate,
    # which also leaves the near-zero high-k spectrum at the +-1% level
    # (the pedestal alone would give -20% side lobes)
    assert np.abs(corr[0]).max() < 2e-2
    spec = _spectrum(corr)
    assert spec.min() > -1e-2 * spec.max()
    # the shape of the correlation itself is intact
    assert corr[NPIX // 2, NPIX // 2 + 1] == pytest.approx(np.exp(-1 / (4 * 1.2**2)), abs=0.05)


def test_normalize_noise_corr_removes_pedestal():
    """A stored (pre-fix) correlation with a pedestal is cleaned on load."""
    # the method only needs a logger: no pipe (and no validated config)
    task = SimpleNamespace(log=logging.getLogger("test_noise_corr"))
    corr = (3.0 * (_gaussian_corr() + 0.02)).astype(np.float32)
    out = MeasureCoaddsPipe._normalize_noise_corr(task, corr)
    assert out[NPIX // 2, NPIX // 2] == 1.0
    np.testing.assert_allclose(out, _gaussian_corr(), atol=1e-5)
    assert _spectrum(out).min() > 0
