import numpy as np
import pytest
from lsst.afw.image import ExposureF
from lsst.geom import Box2I, Extent2I, Point2I
from lsst.meas.base import SkyMapIdGeneratorConfig

from xlens.utils.handle import make_data_id
from xlens.utils.image import (
    _stack_bands,
    combine_sim_exposures,
    prepare_mask,
)


def _make_exposure(value: float, variance: float) -> ExposureF:
    bbox = Box2I(Point2I(0, 0), Extent2I(3, 2))  # shape (2, 3)
    exposure = ExposureF(bbox)
    exposure.getMaskedImage().image.array[:, :] = value
    exposure.getMaskedImage().variance.array[:, :] = variance
    return exposure


def _make_noise(value: float, shape=(2, 3), dtype=np.float32):
    return np.full(shape, value, dtype=dtype)


def test_combine_sim_exposures_inverse_variance_with_noise():
    # Two exposures: values 10 (var=1) and 4 (var=4)
    e_lo = _make_exposure(10.0, 1.0)
    e_hi = _make_exposure(4.0, 4.0)

    # Noise realizations to combine with the same weights
    n_lo = _make_noise(1.0)  # will get weight 1/1
    n_hi = _make_noise(2.0)  # will get weight 1/4

    combined, combined_noise = combine_sim_exposures([e_lo, e_hi], [n_lo, n_hi])

    # Expected inverse-variance weighted values
    w1 = 1.0 / 1.0
    w2 = 1.0 / 4.0
    wsum = w1 + w2

    expected_img = (10.0 * w1 + 4.0 * w2) / wsum
    expected_var = 1.0 / wsum
    expected_noise = (1.0 * w1 + 2.0 * w2) / wsum  # => 1.2

    np.testing.assert_allclose(
        combined.getMaskedImage().image.array,
        expected_img,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        combined.getMaskedImage().variance.array,
        expected_var,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        combined_noise,
        expected_noise,
        rtol=0,
        atol=1e-6,
    )


def test_combine_sim_exposures_raises_on_empty_inputs():
    with pytest.raises(ValueError):
        combine_sim_exposures([], [])
    # Also mismatched lengths should raise
    with pytest.raises(ValueError):
        combine_sim_exposures([_make_exposure(1.0, 1.0)], [])


def test_make_data_id_catalog_id_sweep_patch():
    """For tract=0, patch in range(0, 1000) the SkyMapDimensionPacker
    formula reduces to ``catalog_id = patch + 1_000_000 * tract`` with
    the stand-in skymap record from ``xlens.utils.handle`` (tract_max=
    100, patch_nx_max=patch_ny_max=1000), so the catalog_id sequence
    must equal np.arange(0, 1000).
    """
    cfg = SkyMapIdGeneratorConfig()
    catalog_ids = np.array(
        [
            cfg.apply(make_data_id(tract=0, patch=p, band="i")).catalog_id
            for p in range(0, 1000)
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(catalog_ids, np.arange(0, 1000))


# ---------------------------------------------------------------------------
# Multi-band detection input
#
# Both the patch path (prepare_data_multiband) and the cell path
# (prepare_data_one_cell_multiband) funnel through these two helpers, so
# testing them here covers the stacking logic for both.
# ---------------------------------------------------------------------------


def _band_dict(gal, noise, psf, variance, **overrides):
    out = {
        "gal_array": gal,
        "noise_array": noise,
        "psf_array": psf,
        "noise_variance": variance,
        "pixel_scale": 0.2,
        "begin_x": 0,
        "begin_y": 0,
        "mag_zero": 31.4,
        "base_column_name": "lsst_i_",
    }
    out.update(overrides)
    return out


def test_stack_bands_builds_one_stack_per_plane():
    bands = ["r", "i", "z"]
    gals = [np.full((2, 3), float(i), dtype=np.float32) for i in range(3)]
    noises = [np.full((2, 3), -float(i), dtype=np.float32) for i in range(3)]
    psfs = [np.full((4, 4), 0.1 * i, dtype=np.float64) for i in range(3)]

    out = _stack_bands(
        (
            _band_dict(g, n, p, v)
            for g, n, p, v in zip(gals, noises, psfs, [1.0, 2.0, 3.0])
        ),
        bands,
    )

    assert out["gal_array"].shape == (3, 2, 3)
    assert out["gal_array"].dtype == np.float32
    assert out["noise_array"].shape == (3, 2, 3)
    assert out["psf_array"].shape == (3, 4, 4)
    assert out["noise_variance"] == [1.0, 2.0, 3.0]
    # The coadd belongs to no single band.
    assert out["base_column_name"] is None
    for i in range(3):
        np.testing.assert_array_equal(out["gal_array"][i], gals[i])
        np.testing.assert_array_equal(out["noise_array"][i], noises[i])
        np.testing.assert_array_equal(out["psf_array"][i], psfs[i])


def test_stack_bands_rejects_mismatched_bands():
    gal = np.zeros((2, 3), dtype=np.float32)
    psf = np.zeros((4, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="image shape"):
        _stack_bands(
            iter([
                _band_dict(gal, gal.copy(), psf, 1.0),
                _band_dict(
                    np.zeros((3, 3), dtype=np.float32), gal.copy(), psf, 1.0,
                ),
            ]),
            ["r", "i"],
        )

    with pytest.raises(ValueError, match="pixel_scale"):
        _stack_bands(
            iter([
                _band_dict(gal, gal.copy(), psf, 1.0),
                _band_dict(gal, gal.copy(), psf, 1.0, pixel_scale=0.17),
            ]),
            ["r", "i"],
        )

    with pytest.raises(ValueError, match="noise_array"):
        _stack_bands(
            iter([
                _band_dict(gal, gal.copy(), psf, 1.0),
                _band_dict(gal, None, psf, 1.0),
            ]),
            ["r", "i"],
        )


def test_prepare_mask_planes_negative_pixels_and_original():
    # prepare_mask is the single mask-building primitive used by the
    # systematics tasks: configured planes, the -6 sigma negative-pixel
    # guard, and OR with a pre-existing mask.
    exposure = _make_exposure(1.0, 1.0)
    exposure.mask.array[0, 0] = exposure.mask.getPlaneBitMask("BAD")
    exposure.image.array[1, 1] = -7.0  # < -6 * sqrt(variance=1)
    original = np.zeros((2, 3), dtype=np.int16)
    original[1, 2] = 1

    mask = prepare_mask(
        exposure.image.array,
        exposure.mask,
        exposure.variance.array,
        ["BAD", "NOT_A_PLANE"],  # unknown planes are silently skipped
        original_mask_array=original,
    )
    expected = np.zeros((2, 3), dtype=np.int16)
    expected[0, 0] = 1  # BAD plane
    expected[1, 1] = 1  # negative-pixel guard
    expected[1, 2] = 1  # carried over from original
    np.testing.assert_array_equal(mask, expected)
