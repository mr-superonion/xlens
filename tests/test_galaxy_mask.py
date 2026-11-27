"""Tests for the HyperLEDA large-galaxy ellipse mask."""

import lsst.afw.geom as afwGeom
import lsst.geom as geom
import numpy as np

from xlens.utils.mask import add_galaxy_ellipse_mask

SCALE = 0.2  # arcsec / px


def _wcs(rotation_deg=0.0):
    """TAN WCS at (150, 2) deg, north up / east left rotated by rotation_deg."""
    cd = afwGeom.makeCdMatrix(
        scale=SCALE * geom.arcseconds,
        orientation=rotation_deg * geom.degrees,
        flipX=True,
    )
    return afwGeom.makeSkyWcs(
        crpix=geom.Point2D(500.0, 500.0),
        crval=geom.SpherePoint(150.0, 2.0, geom.degrees),
        cdMatrix=cd,
    )


def _mask(wcs, pa, logr25=np.log10(4.0), d25=0.5, scale=1.0):
    bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(1001, 1001))
    m = np.zeros((1001, 1001), dtype=np.uint8)
    n = add_galaxy_ellipse_mask(
        m, ra=[150.0], dec=[2.0], d25_arcmin=[d25], logr25=[logr25],
        pa=[pa], wcs=wcs, bbox=bbox, scale=scale,
    )
    return m, n


def _direction(m):
    """Pixel-frame angle of the masked region's major axis (deg from +x)."""
    y, x = np.nonzero(m)
    x = x - x.mean()
    y = y - y.mean()
    cov = np.cov(np.vstack([x, y]))
    w, v = np.linalg.eigh(cov)
    major = v[:, np.argmax(w)]
    return np.rad2deg(np.arctan2(major[1], major[0])) % 180.0


def test_size_and_axis_ratio():
    m, n = _mask(_wcs(), pa=0.0, logr25=np.log10(4.0), d25=0.5, scale=1.0)
    assert n == 1
    a_pix = 0.5 * 60 / 2 / SCALE          # 75 px
    expected = np.pi * a_pix * (a_pix / 4.0)
    assert abs(m.sum() / expected - 1) < 0.02


def test_pa_follows_north_through_the_wcs():
    # north along +y: pa = 0 -> major axis along y; pa = 90 (east) -> along x
    m0, _ = _mask(_wcs(0.0), pa=0.0)
    m90, _ = _mask(_wcs(0.0), pa=90.0)
    assert abs(_direction(m0) - 90.0) < 1.0
    assert min(_direction(m90), 180 - _direction(m90)) < 1.0
    # rotating the image rotates the mask with it
    m_rot, _ = _mask(_wcs(30.0), pa=0.0)
    m_ref, _ = _mask(_wcs(0.0), pa=0.0)
    assert abs(m_rot.sum() / m_ref.sum() - 1) < 0.02
    assert abs(((_direction(m_rot) - _direction(m_ref)) % 180.0) - 30.0) < 1.5 or \
        abs(((_direction(m_ref) - _direction(m_rot)) % 180.0) - 30.0) < 1.5


def test_cuts_and_missing_values():
    wcs = _wcs()
    m, n = _mask(wcs, pa=0.0, d25=0.2)       # below d25_min = 0.3
    assert n == 0 and m.sum() == 0
    m, n = _mask(wcs, pa=np.nan, logr25=np.nan, d25=0.5)   # round
    assert n == 1
    assert abs(_direction(m) % 180) >= 0  # defined
    y, x = np.nonzero(m)
    assert abs((x.max() - x.min()) / (y.max() - y.min()) - 1) < 0.03


def test_bits_are_ored():
    wcs = _wcs()
    bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(1001, 1001))
    m = np.full((1001, 1001), 2, dtype=np.uint8)
    add_galaxy_ellipse_mask(m, ra=[150.0], dec=[2.0], d25_arcmin=[0.5],
                            logr25=[0.0], pa=[0.0], wcs=wcs, bbox=bbox)
    assert set(np.unique(m)) == {2, 3}
