import galsim
import lsst.geom as geom
import numpy as np
from lsst.afw.geom import makeSkyWcs

RAD2ASEC = 206264.80624709636


def make_galsim_tanwcs(tract_info):
    """
    Build a GalSim TanWCS consistent with an LSST SkyMap tract WCS.

    Parameters
    ----------

    Returns
    -------
    galsim.TanWCS
    """
    # Pixel location of the tangent point

    skyWcs = tract_info.getWcs()
    sky_center = tract_info.getCtrCoord()
    pix_center = skyWcs.skyToPixel(sky_center)
    x0 = pix_center.getX()
    y0 = pix_center.getY()
    lin = skyWcs.linearizePixelToSky(sky_center, geom.radians)
    J = np.array(lin.getLinear().getMatrix(), dtype=np.float64)
    # Convert to arcsec/pixel for GalSim
    J_arcsec = J * RAD2ASEC
    aff = galsim.AffineTransform(
        dudx=J_arcsec[0, 0], dudy=J_arcsec[0, 1],
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
        cd_matrix = wcs_gs.jacobian(
            galsim.PositionD(crpix[0], crpix[1])
        ).getMatrix() / 3600.0

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
