"""The external detection catalog contract: sky positions + selection.

An external catalog identifies its sources by ``ra``/``dec`` and states
its own selection with ``wsel``/``dwsel_dg1``/``dwsel_dg2``.  Pixel
positions are DERIVED from the sky coordinates against the exposure
being measured, never read from the input -- x1/x2 belong to one
exposure's grid, so a catalog forced onto several bands, patches or
surveys cannot carry meaningful ones.
"""

import logging
from types import SimpleNamespace

import lsst.geom as geom
import numpy as np
import pytest
from lsst.afw.geom import makeCdMatrix, makeSkyWcs
from numpy.lib import recfunctions as rfn

from xlens.processor.measure_base import AnacalMeasureTaskBase

PIXEL_SCALE = 0.2


def _wcs():
    return makeSkyWcs(
        crpix=geom.Point2D(250.0, 250.0),
        crval=geom.SpherePoint(30.0 * geom.degrees, -10.0 * geom.degrees),
        cdMatrix=makeCdMatrix(scale=PIXEL_SCALE * geom.arcseconds),
    )


def _ingest(catalog, wcs=None):
    stub = SimpleNamespace(
        log=logging.getLogger("test_external_detection_contract")
    )
    return AnacalMeasureTaskBase._ingest_external_detection(
        stub, catalog, wcs or _wcs(), PIXEL_SCALE
    )


def _euclid_like(n=6, with_pixels=False):
    """A survey catalog: sky positions and a selection, no pixels."""
    fields = [
        ("ra", "f8"), ("dec", "f8"),
        ("wsel", "f8"), ("dwsel_dg1", "f8"), ("dwsel_dg2", "f8"),
    ]
    if with_pixels:
        fields += [
            ("x1", "f8"), ("x2", "f8"),
            ("x1_det", "f8"), ("x2_det", "f8"),
        ]
    cat = np.zeros(n, dtype=fields)
    wcs = _wcs()
    xs = np.linspace(50.0, 450.0, n)
    ys = np.linspace(60.0, 440.0, n)
    ra, dec = wcs.pixelToSkyArray(xs, ys, degrees=True)
    cat["ra"], cat["dec"] = ra, dec
    cat["wsel"] = 0.8
    cat["dwsel_dg1"] = 0.1
    cat["dwsel_dg2"] = -0.2
    return cat, xs, ys


def test_pixels_created_for_a_catalog_that_has_none():
    """The Euclid shape: ra/dec/wsel only, no pixel columns at all."""
    cat, xs, ys = _euclid_like()
    assert "x1" not in cat.dtype.names

    out = _ingest(cat)

    for name in ("x1", "x2", "x1_det", "x2_det"):
        assert name in out.dtype.names
    np.testing.assert_allclose(out["x1"] / PIXEL_SCALE, xs, atol=1e-9)
    np.testing.assert_allclose(out["x2"] / PIXEL_SCALE, ys, atol=1e-9)
    # detection positions start equal to the measurement positions;
    # forced measurement is what may move x1/x2 off them.
    np.testing.assert_array_equal(out["x1_det"], out["x1"])
    np.testing.assert_array_equal(out["x2_det"], out["x2"])


def test_supplied_pixels_are_ignored_not_trusted():
    """Garbage pixels alongside good sky positions must be overwritten."""
    cat, xs, ys = _euclid_like(with_pixels=True)
    cat["x1"] = 1.0e6
    cat["x2"] = -7.0
    cat["x1_det"] = np.nan
    cat["x2_det"] = np.inf

    out = _ingest(cat)

    np.testing.assert_allclose(out["x1"] / PIXEL_SCALE, xs, atol=1e-9)
    np.testing.assert_allclose(out["x2"] / PIXEL_SCALE, ys, atol=1e-9)
    assert np.all(np.isfinite(out["x1_det"]))
    assert np.all(np.isfinite(out["x2_det"]))


def test_selection_columns_are_carried_through_untouched():
    cat, _, _ = _euclid_like()
    out = _ingest(cat)
    for name in ("wsel", "dwsel_dg1", "dwsel_dg2", "ra", "dec"):
        np.testing.assert_array_equal(out[name], cat[name])


def test_input_is_not_modified_in_place():
    cat, _, _ = _euclid_like(with_pixels=True)
    cat["x1"] = 123.0
    before = cat.copy()
    _ingest(cat)
    np.testing.assert_array_equal(cat["x1"], before["x1"])


@pytest.mark.parametrize(
    "dropped",
    ["ra", "dec", "wsel", "dwsel_dg1", "dwsel_dg2"],
)
def test_required_column_missing_is_rejected(dropped):
    cat, _, _ = _euclid_like()
    cat = rfn.drop_fields(cat, dropped, usemask=False)
    with pytest.raises(ValueError, match=dropped):
        _ingest(cat)


def test_no_default_selection_weight():
    """wsel must be stated, never invented.

    Zero would silently empty the finalized catalog and 1.0 would
    assert a shear-independent selection -- only true when the
    selection is made on pre-lensed (truth) properties.  Neither is
    safe to assume, so the column is required.
    """
    cat, _, _ = _euclid_like()
    cat = rfn.drop_fields(cat, ["wsel", "dwsel_dg1", "dwsel_dg2"],
                          usemask=False)
    with pytest.raises(ValueError, match="selection weight"):
        _ingest(cat)


def test_all_zero_sky_positions_are_rejected():
    """The failure this contract exists to prevent.

    A catalog built with pixel positions only leaves ra = dec = 0, which
    used to sail through ingest and then lose every row downstream:
    is_primary is decided by which tract contains the sky position, and
    tract 0 does not contain (0, 0).  The result was an empty catalog
    and a nan shear, with nothing logged.
    """
    cat, _, _ = _euclid_like()
    cat["ra"] = 0.0
    cat["dec"] = 0.0
    with pytest.raises(ValueError, match="ra = dec = 0"):
        _ingest(cat)


def test_a_single_source_at_the_origin_is_allowed():
    """Only an ALL-zero catalog is a contract violation.

    (0, 0) is a real sky position; one row sitting there next to valid
    ones must not trip the guard.
    """
    cat, _, _ = _euclid_like()
    cat["ra"][0] = 0.0
    cat["dec"][0] = 0.0
    out = _ingest(cat)
    assert len(out) == len(cat)
