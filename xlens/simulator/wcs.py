"""WCS conversion utilities between LSST and GalSim coordinate systems.

Coordinate conventions
----------------------
Both LSST and GalSim use tangent-plane (intermediate world) coordinates,
Near the tangent point at (RA_0, Dec_0):

    u' ≈ (RA - RA_0) * cos(Dec_0)     (East,  LSST/FITS)
    v' ≈  Dec - Dec_0                 (North, LSST/FITS)

    u  ≈ -(RA - RA_0) * cos(Dec_0)   (West,  GalSim)
    v  ≈   Dec - Dec_0               (North, GalSim)

So u = -u' and v = v'. The cos(Dec_0) factor is absorbed into the
projection; the CD matrix entries do NOT scale with 1/cos(Dec).

LSST/FITS CD matrix (getCdMatrix, makeSkyWcs cdMatrix, base_LocalWcs_CDMatrix):
    [[du'/dx, du'/dy], [dv'/dx, dv'/dy]]
    Units: degrees/pixel (getCdMatrix) or radians/pixel
    (linearizePixelToSky with radians).

GalSim Jacobian (dudx, dudy, dvdx, dvdy in AffineTransform / JacobianWCS):
    [[du/dx, du/dy], [dv/dx, dv/dy]]
    Units: arcsec/pixel (when units=galsim.arcsec).

GalSim FITS CD matrix (TanWCS.cd property):
    Same as LSST/FITS convention in degrees/pixel.
    GalSim converts internally between this and its (u, v) Jacobian.

Conversion rule
---------------
To go from LSST/FITS CD matrix to GalSim Jacobian:
    Negate the first row (u' East -> u West) and convert units.
    du/dx = -du'/dx,  du/dy = -du'/dy
    dv/dx =  dv'/dx,  dv/dy =  dv'/dy

To go from GalSim to LSST:
    Use TanWCS.cd (already in FITS convention), pass directly to makeSkyWcs.
"""

import galsim
import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs

RAD2ASEC = 206264.80624709636


def tanwcs_dm2galsim(wcs):
    """Build a GalSim TanWCS from an LSST SkyWcs.

    Converts the LSST CD matrix (u'=East, degrees/pixel) to GalSim's
    Jacobian (u=West, arcsec/pixel) by negating the first row and
    scaling by 3600. See module docstring for coordinate conventions.

    Parameters
    ----------
    wcs : lsst.afw.geom.SkyWcs
        LSST sky WCS object.

    Returns
    -------
    galsim.TanWCS
    """
    sky_center = wcs.getSkyOrigin()
    pix_center = wcs.skyToPixel(sky_center)
    x0 = pix_center.getX()
    y0 = pix_center.getY()
    J_arcsec = wcs.getCdMatrix() * 3600
    aff = galsim.AffineTransform(
        dudx=-J_arcsec[0, 0], dudy=-J_arcsec[0, 1],
        dvdx=J_arcsec[1, 0], dvdy=J_arcsec[1, 1],
        origin=galsim.PositionD(x0, y0),
    )
    world_origin = galsim.CelestialCoord(
        sky_center.getRa().asRadians() * galsim.radians,
        sky_center.getDec().asRadians() * galsim.radians,
    )

    wcs_galsim = galsim.TanWCS(
        affine=aff, world_origin=world_origin, units=galsim.arcsec
    )
    return wcs_galsim


def tanwcs_galsim2dm(wcs_gs):
    """Convert a GalSim TanWCS to an LSST SkyWcs.

    Uses GalSim's .cd property, which returns the FITS CD matrix
    (u'=East, degrees/pixel) — the same convention as LSST's makeSkyWcs
    cdMatrix parameter. No sign flip is needed.
    See module docstring for coordinate conventions.

    Parameters
    ----------
    wcs_gs : galsim.TanWCS
        GalSim TAN WCS object.

    Returns
    -------
    lsst.afw.geom.SkyWcs
    """

    if wcs_gs.wcs_type == 'TAN':
        crpix = wcs_gs.crpix
        stack_crpix = geom.Point2D(crpix[0], crpix[1])
        cd_matrix = wcs_gs.cd

        crval = geom.SpherePoint(
            wcs_gs.center.ra.rad,
            wcs_gs.center.dec.rad,
            geom.radians,
        )
        wcs_dm = makeSkyWcs(
            crpix=stack_crpix,
            crval=crval,
            cdMatrix=cd_matrix,
        )
    else:
        raise RuntimeError(
            "Does not support wcs_gs type: %s" % wcs_gs.wcs_type
        )

    return wcs_dm
