"""Unit tests for ``xlens.utils.mask`` — GAIA bright-star masking.

Covers the two shared helpers used by ``BuildSystematicsTask`` and
``BuildCellSystematicsTask``:

* :func:`get_gaia_table` — raw GAIA refCat → wide structured array.
* :func:`build_gaia_xyr` — wide table + bbox + ``star_mask_type`` →
  ``(x, y, r)`` structured array consumed by
  ``anacal.mask.add_bright_star_mask``.

The refCat is replaced by a tiny stand-in object that mimics the only
method these helpers touch (``.asAstropy()``) so the test runs without
loading a real ``ReferenceObjectLoader``.
"""
from __future__ import annotations

import lsst.geom as lsst_geom
import numpy as np
import pytest
from astropy.table import Table
from lsst.afw.geom import makeSkyWcs

from xlens.utils.mask import (
    GAIA_TABLE_DTYPE,
    build_gaia_xyr,
    get_gaia_table,
)

PIXEL_SCALE_ARCSEC = 0.2  # ComCam DP1 cell-coadd pixel scale
CRPIX = lsst_geom.Point2D(2000.0, 3000.0)  # reference pixel (arbitrary)
CRVAL = lsst_geom.SpherePoint(150.0, 2.0, lsst_geom.degrees)  # ref sky position


def _make_wcs():
    """Plain TAN WCS centred at (RA, Dec) = (150°, 2°) with 0.2"/px."""
    scale = PIXEL_SCALE_ARCSEC * lsst_geom.arcseconds
    return makeSkyWcs(
        crpix=CRPIX, crval=CRVAL,
        cdMatrix=lsst_geom.LinearTransform.makeScaling(
            scale.asRadians(),
        ).getMatrix(),
    )


class _FakeRefCat:
    """Minimal stand-in for an LSST ``SimpleCatalog`` GAIA refCat.

    The only thing ``get_gaia_table`` calls on the refCat is
    ``.asAstropy()``, so we return the astropy Table directly with the
    columns the helper reads: ``coord_ra`` / ``coord_dec`` (radians),
    ``phot_g_mean_flux`` (nJy), ``id``.
    """

    def __init__(self, ra_deg, dec_deg, mag_ab, ids):
        # GAIA refCat uses radians for sky coords (afw.geom Angle convention).
        coord_ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
        coord_dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
        # mag_AB = -2.5 log10(F_nJy) + 31.4  ->  F_nJy = 10**((31.4 - m)/2.5)
        flux_nJy = 10.0 ** ((31.4 - np.asarray(mag_ab, dtype=np.float64)) / 2.5)
        self._tab = Table({
            "coord_ra": coord_ra,
            "coord_dec": coord_dec,
            "phot_g_mean_flux": flux_nJy,
            "id": np.asarray(ids, dtype=np.int64),
        })

    def asAstropy(self):
        return self._tab


def _make_test_inputs():
    """Five GAIA stars at known (ra, dec, mag) on the TAN WCS above."""
    ra_deg = np.array([150.0, 150.001, 149.999, 150.002, 149.998])
    dec_deg = np.array([2.0, 2.001, 2.0, 1.999, 2.002])
    mag = np.array([8.0, 12.0, 16.0, 19.0, 23.0])  # one in each radius bin + one past mag 20
    ids = np.array([101, 102, 103, 104, 105], dtype=np.int64)
    return _FakeRefCat(ra_deg, dec_deg, mag, ids), ra_deg, dec_deg, mag, ids


def test_get_gaia_table_dtype_and_columns():
    """Output is exactly :data:`GAIA_TABLE_DTYPE`, with the input ra/dec
    in degrees, ids preserved, and pixel coords matching what
    ``wcs.skyToPixelArray`` would produce."""
    refcat, ra_deg, dec_deg, mag, ids = _make_test_inputs()
    wcs = _make_wcs()

    table = get_gaia_table(refcat, wcs)

    assert table.dtype == GAIA_TABLE_DTYPE
    assert len(table) == 5

    # ra/dec round-trip from radians -> degrees.
    np.testing.assert_allclose(table["ra"], ra_deg, atol=1e-10)
    np.testing.assert_allclose(table["dec"], dec_deg, atol=1e-10)

    # ids untouched.
    np.testing.assert_array_equal(table["gaia_source_id"], ids)

    # AB magnitudes come back to within 1 mmag of our injected values.
    np.testing.assert_allclose(table["gaia_g_mag"], mag, atol=1e-3)

    # x_in_tract / y_in_tract should match an independent call to
    # wcs.skyToPixelArray.
    x_ref, y_ref = wcs.skyToPixelArray(ra=ra_deg, dec=dec_deg, degrees=True)
    np.testing.assert_allclose(table["x_in_tract"], x_ref, atol=1e-10)
    np.testing.assert_allclose(table["y_in_tract"], y_ref, atol=1e-10)


def test_build_gaia_xyr_default_radius_per_bin():
    """``starMaskType='default'`` reproduces the step function
    450/200/100 px for mag ≤ 11/14/20 and drops mag > 20."""
    refcat, _, _, mag, ids = _make_test_inputs()
    wcs = _make_wcs()
    bbox = lsst_geom.Box2I(lsst_geom.Point2I(1000, 2000),
                           lsst_geom.Extent2I(3000, 3000))

    table = get_gaia_table(refcat, wcs)
    xyr = build_gaia_xyr(table, bbox, star_mask_type="default")

    assert xyr is not None
    assert xyr.dtype.names == ("x", "y", "r")
    # mag=23 should be dropped; 4 rows expected.
    assert len(xyr) == 4

    # Pixel coords are bbox-local: x = x_in_tract - bbox.minX.
    expected_x = table["x_in_tract"][:4] - bbox.getBeginX()
    expected_y = table["y_in_tract"][:4] - bbox.getBeginY()
    np.testing.assert_allclose(xyr["x"], expected_x, atol=1e-10)
    np.testing.assert_allclose(xyr["y"], expected_y, atol=1e-10)

    # Each surviving star sits in its expected mag bin.
    # mags [8, 12, 16, 19]  ->  r [450, 200, 100, 100]
    np.testing.assert_array_equal(xyr["r"], [450.0, 200.0, 100.0, 100.0])


def test_build_gaia_xyr_no_mask_flat_radius():
    """``starMaskType='no_mask'`` puts a flat r=10 px halo on every
    GAIA star with mag ≤ 20 and drops anything fainter."""
    refcat, _, _, mag, ids = _make_test_inputs()
    wcs = _make_wcs()
    bbox = lsst_geom.Box2I(lsst_geom.Point2I(1000, 2000),
                           lsst_geom.Extent2I(3000, 3000))

    table = get_gaia_table(refcat, wcs)
    xyr = build_gaia_xyr(table, bbox, star_mask_type="no_mask")

    assert xyr is not None
    assert len(xyr) == 4  # mag=23 dropped
    np.testing.assert_array_equal(xyr["r"], [10.0, 10.0, 10.0, 10.0])


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
