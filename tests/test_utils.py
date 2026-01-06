import numpy as np
import pytest
from lsst.afw.image import ExposureF
from lsst.geom import Box2I, Extent2I, Point2I

from xlens.utils.image import combine_sim_exposures


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
