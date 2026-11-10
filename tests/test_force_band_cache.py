"""The forced measurement reuses the multiband detection preparation per band
(``MeasureCoaddsPipe._cache_bands_multiband``).  Each band's slice must carry
that band's OWN noise correlation and the detection stage's noise realisation
of that band -- not the stack's list of correlations, which made the anacal
renoising raise ("3 noise correlation function(s) for 1 band(s)") and the
forced FPFS measurement fall back to white noise.
"""

from types import SimpleNamespace

import anacal
import numpy as np
import pytest

from xlens.processor.measure_coadds import MeasureCoaddsPipe, MeasureCoaddsPipeConfig

BANDS = ["r", "i", "z"]
NPIX = 64
SIZE = 500
SCALE = 0.168


def _corr(k):
    """A centrally peaked correlation, different for every band."""
    c = np.zeros((49, 49))
    c[24, 24] = 1.0
    c[24, 23] = c[24, 25] = 0.05 + 0.01 * k
    c[23, 24] = c[25, 24] = 0.06 + 0.01 * k
    return c


def _multiband_data(rng):
    nb = len(BANDS)
    psf = np.zeros((nb, NPIX, NPIX))
    psf[:, NPIX // 2, NPIX // 2] = 1.0
    geo = anacal.geometry.get_cell_list(
        img_ny=SIZE, img_nx=SIZE, cell_nx=250, cell_ny=250, cell_overlap=80, scale=SCALE,
    )
    for gg in geo:
        gg.psf_array = psf.copy()
    data = {
        "gal_array": rng.normal(size=(nb, SIZE, SIZE)).astype(np.float32),
        "noise_array": rng.normal(size=(nb, SIZE, SIZE)).astype(np.float32),
        "psf_array": psf,
        "noise_variance": [1.0 + ib for ib in range(nb)],
        "noise_corrs": [_corr(ib) for ib in range(nb)],
        "pixel_scale": SCALE,
        "mag_zero": 31.4,
        "begin_x": 0,
        "begin_y": 0,
        "base_column_name": None,
        "filler_added": False,
    }
    return data, geo


def test_band_slices_carry_their_own_noise():
    config = MeasureCoaddsPipeConfig()
    config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
    config.detection_bands = BANDS
    config.validate()
    task = MeasureCoaddsPipe(config=config)

    data, cells = _multiband_data(np.random.default_rng(0))
    exposures = {b: SimpleNamespace() for b in BANDS}   # unused for psf_model_type="patch"
    cache = task._cache_bands_multiband(bands=BANDS, exposures=exposures, data=data, cells=cells)

    for ib, band in enumerate(BANDS):
        bdata = cache[band]["data"]
        assert "noise_corrs" not in bdata
        assert bdata["noise_corr"] is data["noise_corrs"][ib]
        assert np.ndim(bdata["noise_variance"]) == 0
        assert bdata["noise_variance"] == pytest.approx(data["noise_variance"][ib])
        # the detection stage's noise realisation of THIS band, as a view
        assert np.shares_memory(bdata["noise_array"], data["noise_array"])
        np.testing.assert_array_equal(bdata["noise_array"], data["noise_array"][ib])
        np.testing.assert_array_equal(bdata["gal_array"], data["gal_array"][ib])
        assert len(cache[band]["cells"]) == len(cells)


def test_band_slices_without_correlation():
    config = MeasureCoaddsPipeConfig()
    config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
    config.detection_bands = BANDS
    config.validate()
    task = MeasureCoaddsPipe(config=config)

    data, cells = _multiband_data(np.random.default_rng(1))
    data.pop("noise_corrs")
    data["noise_array"] = None
    exposures = {b: SimpleNamespace() for b in BANDS}
    cache = task._cache_bands_multiband(bands=BANDS, exposures=exposures, data=data, cells=cells)
    for band in BANDS:
        bdata = cache[band]["data"]
        assert bdata["noise_corr"] is None
        assert bdata["noise_array"] is None
        assert "noise_corrs" not in bdata


class _RecordingCnn:
    """Stands in for CnnMeasurementTask: records what the force stage hands it."""

    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        import inspect

        from xlens.processor.cnn import CnnMeasurementTask

        # the real task must accept exactly this call
        inspect.signature(CnnMeasurementTask.run).bind(None, **kwargs)
        self.calls.append(kwargs)
        return np.zeros(len(kwargs["detection"]), dtype=[("cnn_dummy", "f8")])


def test_cnn_gets_each_band_own_noise_correlation():
    """The forced CNN reads the same cached band slice as the forced FPFS:
    one band's correlation (no stack list), that band's two noise
    realisations, variance and PSF -- what its spectral renoising needs."""
    config = MeasureCoaddsPipeConfig()
    config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
    config.detection_bands = BANDS
    config.validate()
    task = MeasureCoaddsPipe(config=config)

    rng = np.random.default_rng(2)
    data, cells = _multiband_data(rng)
    data["noise2_array"] = rng.normal(size=data["noise_array"].shape).astype(np.float32)
    for k, gg in enumerate(cells):
        gg.psf_array = gg.psf_array * (1.0 + 0.1 * np.arange(len(BANDS)))[:, None, None]
    exposures = {b: SimpleNamespace() for b in BANDS}
    cache = task._cache_bands_multiband(bands=BANDS, exposures=exposures, data=data, cells=cells)

    cnn = _RecordingCnn()
    host = SimpleNamespace(
        config=SimpleNamespace(
            do_measure_cnn=True, n_mask_base_max=0.04,
            renoise_mode="spectral", renoise_c=0.1, renoise_delta=0.01,
        ),
        cnn=cnn,
    )
    det = np.zeros(3, dtype=[("x1_det", "f8"), ("x2_det", "f8")])
    cat = np.zeros(3, dtype=[("fpfs_dummy", "f8")])
    for ib, band in enumerate(BANDS):
        bdata = cache[band]["data"]
        cell = cache[band]["cells"][0]
        MeasureCoaddsPipe._append_cnn_shapes(
            host, cat, data={**bdata, "detection": det, "cells": [cell]}, band=band
        )
        kw = cnn.calls[-1]
        assert kw["renoise_mode"] == "spectral"
        assert "noise_corrs" not in kw
        assert kw["noise_corr"] is data["noise_corrs"][ib]
        assert kw["noise_variance"] == pytest.approx(data["noise_variance"][ib])
        np.testing.assert_array_equal(kw["noise_array"], data["noise_array"][ib])
        np.testing.assert_array_equal(kw["noise2_array"], data["noise2_array"][ib])
        assert np.ndim(kw["cells"][0].psf_array) == 2
        np.testing.assert_array_equal(kw["cells"][0].psf_array, cells[0].psf_array[ib])
