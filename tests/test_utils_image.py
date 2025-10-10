import pytest

np = pytest.importorskip("numpy")

from lsst.afw.image import ExposureF
from lsst.geom import Box2I, Extent2I, Point2I

from xlens.utils.image import combine_sim_exposures


def _make_exposure(value: float, variance: float) -> ExposureF:
    bbox = Box2I(Point2I(0, 0), Extent2I(3, 2))
    exposure = ExposureF(bbox)
    exposure.getMaskedImage().image.array[:, :] = value
    exposure.getMaskedImage().variance.array[:, :] = variance
    return exposure


def test_combine_sim_exposures_inverse_variance():
    exposure_low_noise = _make_exposure(10.0, 1.0)
    exposure_high_noise = _make_exposure(4.0, 4.0)

    combined = combine_sim_exposures([exposure_low_noise, exposure_high_noise])

    expected_weighted_value = (10.0 * 1.0 + 4.0 * 0.25) / (1.0 + 0.25)
    np.testing.assert_allclose(
        combined.getMaskedImage().image.array,
        expected_weighted_value,
        rtol=0,
        atol=1e-6,
    )

    expected_variance = 1.0 / (1.0 + 0.25)
    np.testing.assert_allclose(
        combined.getMaskedImage().variance.array,
        expected_variance,
        rtol=0,
        atol=1e-6,
    )


def test_combine_sim_exposures_raises_on_empty_list():
    with pytest.raises(ValueError):
        combine_sim_exposures([])


def test_combine_sim_exposures_allows_nan_variance():
    exposure = _make_exposure(6.0, 2.0)
    exposure_with_nan = _make_exposure(2.0, 8.0)

    variance_array = exposure_with_nan.getMaskedImage().variance.array
    variance_array[0, 0] = np.nan

    combined = combine_sim_exposures([exposure, exposure_with_nan])

    weight_one = 1.0 / 2.0
    weight_two = 1.0 / 8.0
    expected_value = (6.0 * weight_one + 2.0 * weight_two) / (weight_one + weight_two)
    expected_variance = 1.0 / (weight_one + weight_two)

    np.testing.assert_allclose(
        combined.getMaskedImage().image.array,
        expected_value,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        combined.getMaskedImage().variance.array,
        expected_variance,
        rtol=0,
        atol=1e-6,
    )
