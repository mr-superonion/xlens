import galsim
import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs

RAD2ASEC = 206264.80624709636


def make_galsim_tanwcs(wcs):
    """
    Build a GalSim TanWCS consistent with an LSST SkyWcs.

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


def make_dm_wcs(wcs_gs):
    """
    convert galsim wcs to stack wcs

    Parameters
    ----------
    wcs_gs: galsim WCS
        Should be TAN or TAN-SIP

    Returns
    -------
    DM Stack sky wcs
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
