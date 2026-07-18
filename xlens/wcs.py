# This file is part of xlens.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

import astropy.wcs
import galsim
import lsst.geom as geom
import numpy as np
from lsst.afw.geom import SkyWcs, makeSkyWcs

RAD2ASEC = 206264.80624709636


def pixel_to_sky(wcs, x, y):
    """Convert global pixel coordinates to sky ``(ra, dec)`` in degrees.

    The inputs ``x, y`` are **global** pixel coordinates in the WCS's own
    frame -- i.e. DM *parent* pixels for a ``SkyWcs`` (which already include
    the exposure ``XY0``). Any local-cutout -> global lift is the caller's
    responsibility; this function applies no offset.

    A FITS-header WCS (e.g. a Euclid mosaic) must first be converted to a
    ``SkyWcs`` with :func:`header_to_dm_wcs`; astropy WCS is no longer
    accepted here.

    Parameters
    ----------
    wcs : lsst.afw.geom.SkyWcs
        The world coordinate system.
    x, y : array_like
        Global pixel coordinates, 0-based, with ``x`` = column, ``y`` = row.

    Returns
    -------
    ra, dec : np.ndarray
        Sky coordinates in degrees.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if isinstance(wcs, SkyWcs):
        return wcs.pixelToSkyArray(x, y, degrees=True)
    raise TypeError(f"unsupported WCS type: {type(wcs)}")


def sky_to_pixel(wcs, ra, dec):
    """Convert sky ``(ra, dec)`` in degrees to global pixel coordinates.

    Inverse of :func:`pixel_to_sky`. Returns **global** pixel coordinates in
    the WCS's own frame (DM parent pixels); no offset is applied. A FITS-header
    WCS (e.g. a Euclid mosaic) must first be converted with
    :func:`header_to_dm_wcs`; astropy WCS is no longer accepted here.

    Parameters
    ----------
    wcs : lsst.afw.geom.SkyWcs
        The world coordinate system.
    ra, dec : array_like
        Sky coordinates in degrees.

    Returns
    -------
    x, y : np.ndarray
        Global pixel coordinates, 0-based, with ``x`` = column, ``y`` = row.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if isinstance(wcs, SkyWcs):
        return wcs.skyToPixelArray(ra, dec, degrees=True)
    raise TypeError(f"unsupported WCS type: {type(wcs)}")


def pixel_scale_arcsec(wcs):
    """Return the WCS pixel scale in arcsec/pixel (geometric mean).

    Works for an LSST ``SkyWcs`` (``getPixelScale``) or an ``astropy.wcs.WCS``
    (``sqrt|det(CD)|``), matching the scalar ``pixel_scale`` AnaCal uses.

    Parameters
    ----------
    wcs : lsst.afw.geom.SkyWcs or astropy.wcs.WCS

    Returns
    -------
    float
        Pixel scale in arcsec/pixel.
    """
    if isinstance(wcs, SkyWcs):
        return float(wcs.getPixelScale().asArcseconds())
    if isinstance(wcs, astropy.wcs.WCS):
        cd = wcs.pixel_scale_matrix  # degrees/pixel
        return float(np.sqrt(np.abs(np.linalg.det(cd))) * 3600.0)
    raise TypeError(f"unsupported WCS type: {type(wcs)}")


def update_x1x2_from_sky(
    catalog,
    wcs,
    x0=0.0,
    y0=0.0,
    ra_col="ra",
    dec_col="dec",
    x1_col="x1",
    x2_col="x2",
):
    """Set the ``(x1, x2)`` position columns from the ``(ra, dec)`` columns.

    Inverse of the pixel->sky step in ``AnacalTask``: maps each object's
    ``(ra, dec)`` to global pixels through ``wcs`` (:func:`sky_to_pixel`),
    subtracts the exposure origin ``(x0, y0)``, and scales to arcsec with the
    WCS pixel scale. ``x1_col`` / ``x2_col`` are overwritten in place.

    ``x1`` / ``x2`` follow AnaCal's convention ``arcsec = pixel * scale``. With
    the default ``x0 = y0 = 0`` they are in the WCS's global (parent) frame --
    matching the final AnaCal catalog. Pass ``(x0, y0) = exposure.getXY0()``
    to express them in a sub-image-local frame.

    Parameters
    ----------
    catalog : structured np.ndarray or astropy.table.Table
        Catalog holding ``ra_col`` / ``dec_col``; ``x1_col`` / ``x2_col`` are
        written in place.
    wcs : lsst.afw.geom.SkyWcs or astropy.wcs.WCS
        World coordinate system.
    x0, y0 : float, optional
        Pixel origin subtracted from the global pixel position (default 0, 0).
    ra_col, dec_col, x1_col, x2_col : str, optional
        Column names. ``ra_col`` / ``dec_col`` are in degrees.

    Returns
    -------
    catalog
        The same object, with ``x1_col`` / ``x2_col`` updated (for chaining).
    """
    pixel_scale = pixel_scale_arcsec(wcs)
    x, y = sky_to_pixel(
        wcs,
        np.asarray(catalog[ra_col]),
        np.asarray(catalog[dec_col]),
    )
    catalog[x1_col] = (x - x0) * pixel_scale
    catalog[x2_col] = (y - y0) * pixel_scale
    return catalog


def jacobian_reconstruction(pixel_scale, g1, g2, rho, kappa=0.0):
    """Reconstruct a 2x2 Jacobian matrix from shear, rotation, and convergence.

    Inverse of jacobian_decomposition. The Jacobian is in the GalSim
    convention: [[du/dx, du/dy], [dv/dx, dv/dy]] where (u, v) are
    tangent-plane coordinates with u=West and v=North, in arcsec/pixel.
    Note: u ≈ -(RA - RA_0) * cos(Dec_0), NOT raw RA.

    The Jacobian is factored as:

        J = pixel_scale * (1-kappa) * R(rho) @ S(g1, g2)

    where R is a rotation matrix and S is the lensing shear matrix:

        R(rho) = [[cos(rho), sin(rho)], [-sin(rho), cos(rho)]]
        S(g1, g2) = [[1-g1, -g2], [-g2, 1+g1]]

    Positive rho corresponds to a clockwise rotation in the (u, v)
    tangent-plane (u=West, v=North).  At small rho this reduces to:
        J ≈ pixel_scale * (1-kappa) * [[1-g1, -g2+rho], [-g2-rho, 1+g1]]

    Parameters
    ----------
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    g1, g2 : float
        Reduced shear components in lensing convention (dimensionless).
    rho : float
        Rotation angle in radians.
    kappa : float, optional
        Convergence (dimensionless). Default is 0.

    Returns
    -------
    jac : 2x2 np.ndarray
        Jacobian matrix [[du/dx, du/dy], [dv/dx, dv/dy]] in arcsec/pixel,
        using GalSim tangent-plane convention (u=West, v=North).
    """
    cos_rho = np.cos(rho)
    sin_rho = np.sin(rho)
    R = np.array([[cos_rho, sin_rho], [-sin_rho, cos_rho]])
    S = np.array([[1.0 - g1, -g2], [-g2, 1.0 + g1]])
    return pixel_scale * (1.0 - kappa) * (R @ S)


def jacobian_decomposition(jac, pixel_scale):
    """Decompose a 2x2 Jacobian matrix into shear, rotation, and convergence.

    Inverse of jacobian_reconstruction. The input Jacobian should be in
    GalSim convention: [[du/dx, du/dy], [dv/dx, dv/dy]] where (u, v) are
    tangent-plane coordinates with u=West and v=North, in arcsec/pixel.
    Note: u ≈ -(RA - RA_0) * cos(Dec_0), NOT raw RA.

    The Jacobian is decomposed as:

        J = pixel_scale * (1-kappa) * R(rho) @ S(g1, g2)

    where R is a rotation matrix and S is the lensing shear matrix
    (see jacobian_reconstruction for definitions). Positive rho
    corresponds to a clockwise rotation in the (u, v) tangent-plane.

    Parameters
    ----------
    jac : 2x2 array
        Jacobian matrix [[du/dx, du/dy], [dv/dx, dv/dy]] in arcsec/pixel,
        using GalSim tangent-plane convention (u=West, v=North).
    pixel_scale : float
        Pixel scale in arcsec/pixel.

    Returns
    -------
    g1, g2 : float
        Reduced shear components in lensing convention (dimensionless).
    rho : float
        Rotation angle in radians.
    kappa : float
        Convergence (dimensionless).
    """
    m = jac / pixel_scale

    # R(rho) @ S has trace = 2*(1-kappa)*cos(rho)
    # and antisymmetric part m[0,1]-m[1,0] = 2*(1-kappa)*sin(rho)
    trace = m[0, 0] + m[1, 1]
    anti = m[0, 1] - m[1, 0]

    one_minus_kappa = np.sqrt(trace**2 + anti**2) / 2.0
    kappa = 1.0 - one_minus_kappa
    rho = np.arctan2(anti, trace)

    # Remove scale and convergence
    n = m / one_minus_kappa

    # Remove rotation: S = R(-rho)^T @ N = R(rho)_standard @ N
    cos_rho = np.cos(rho)
    sin_rho = np.sin(rho)
    s00 = cos_rho * n[0, 0] - sin_rho * n[1, 0]
    s01 = cos_rho * n[0, 1] - sin_rho * n[1, 1]
    s10 = sin_rho * n[0, 0] + cos_rho * n[1, 0]
    s11 = sin_rho * n[0, 1] + cos_rho * n[1, 1]

    g1 = (s11 - s00) / 2.0
    g2 = -(s01 + s10) / 2.0
    return g1, g2, rho, kappa


def make_jwcs(pixel_scale, g1, g2, rho):
    """Build a GalSim JacobianWCS with shear and rotation.

    Constructs a 2x2 Jacobian via :func:`jacobian_reconstruction`
    and wraps it in a ``galsim.JacobianWCS``.

    Parameters
    ----------
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    g1, g2 : float
        Reduced shear components in lensing convention.
    rho : float
        Rotation angle in radians.

    Returns
    -------
    galsim.JacobianWCS
    """
    jac = jacobian_reconstruction(pixel_scale, g1, g2, rho)
    return galsim.JacobianWCS(
        dudx=jac[0, 0],
        dudy=jac[0, 1],
        dvdx=jac[1, 0],
        dvdy=jac[1, 1],
    )


def extract_perturbation_jwcs(wcs, pixel_scale):
    """Extract (g1, g2, rho, kappa) from a GalSim JacobianWCS.

    Reads the 2x2 matrix from a ``galsim.JacobianWCS`` and decomposes
    it via :func:`jacobian_decomposition`.

    Parameters
    ----------
    wcs : galsim.JacobianWCS
        GalSim Jacobian WCS (u=West, v=North, arcsec/pixel).
    pixel_scale : float
        Reference pixel scale in arcsec/pixel.

    Returns
    -------
    g1, g2, rho, kappa : float
        Shear, rotation, and convergence perturbations.
    """
    return jacobian_decomposition(wcs.getMatrix(), pixel_scale)


def extract_perturbation_galsim_wcs(wcs_gs, pixel_point, pixel_scale):
    """Extract (g1, g2, rho, kappa) from a GalSim TanWCS at a pixel position.

    Evaluates the local Jacobian of the GalSim TanWCS at the given pixel
    position and decomposes it into shear, rotation, and convergence.

    Parameters
    ----------
    wcs_gs : galsim.TanWCS
        GalSim TAN WCS object.
    pixel_point : galsim.PositionD
        Pixel position at which to evaluate the local Jacobian.
    pixel_scale : float
        Pixel scale in arcsec/pixel.

    Returns
    -------
    g1, g2 : float
        Reduced shear components in lensing convention (dimensionless).
    rho : float
        Rotation angle in radians.
    kappa : float
        Convergence (dimensionless).
    """
    local_wcs = wcs_gs.local(image_pos=pixel_point)
    jac = np.array(
        [
            [local_wcs.dudx, local_wcs.dudy],
            [local_wcs.dvdx, local_wcs.dvdy],
        ]
    )
    return jacobian_decomposition(jac, pixel_scale)


def extract_perturbation_dm_wcs(wcs_dm, pixel_point, pixel_scale):
    """Extract (g1, g2, rho, kappa) from an LSST SkyWcs.

    Linearizes the LSST SkyWcs at the given pixel position and decomposes the
    local Jacobian into shear, rotation, and convergence.

    Convention difference from GalSim
    ----------------------------------
    Both LSST and GalSim use tangent-plane (intermediate world) coordinates,
    The cos(Dec_0) factor is absorbed in the projection,
    so the CD matrix entries do NOT scale with 1/cos(Dec).

    LSST CDMatrix (from linearizePixelToSky or base_LocalWcs_CDMatrix):
        [[du'/dx, du'/dy], [dv'/dx, dv'/dy]]
        u' ≈ (RA - RA_0) * cos(Dec_0) (East), v' ≈ Dec - Dec_0 (North).
        Units: radians/pixel.

    GalSim Jacobian:
        [[du/dx, du/dy], [dv/dx, dv/dy]]
        u ≈ -(RA - RA_0) * cos(Dec_0) (West), v ≈ Dec - Dec_0 (North).
        Units: arcsec/pixel.

    Since u = -u', conversion requires negating the first row and scaling
    from radians to arcsec.

    Parameters
    ----------
    wcs_dm : lsst.afw.geom.SkyWcs
        LSST sky WCS object.
    pixel_point : lsst.geom.Point2D
        Pixel position at which to linearize the WCS.
    pixel_scale : float
        Pixel scale in arcsec/pixel.

    Returns
    -------
    g1, g2 : float
        Shear distortion components (dimensionless).
    rho : float
        Rotation component in radians (dimensionless).
    kappa : float
        Convergence (dimensionless).

    Note
    ----
    The returned perturbations follow GalSim's (u=West, v=North) convention.
    The LSST CDMatrix is first converted to GalSim's tangent-plane convention
    before decomposition.
    """
    import lsst.geom as lgeom

    local_transform = wcs_dm.linearizePixelToSky(pixel_point, lgeom.radians)
    linear = local_transform.getLinear()
    cd_local = np.array(
        [
            [linear[0, 0], linear[0, 1]],
            [linear[1, 0], linear[1, 1]],
        ]
    )
    # Convert radians to arcsec, negate first row (u' East -> u West)
    rad_to_arcsec = (180.0 / np.pi) * 3600.0
    jac_arcsec = np.array(
        [
            [-cd_local[0, 0] * rad_to_arcsec, -cd_local[0, 1] * rad_to_arcsec],
            [cd_local[1, 0] * rad_to_arcsec, cd_local[1, 1] * rad_to_arcsec],
        ]
    )
    return jacobian_decomposition(jac_arcsec, pixel_scale)


def header_to_dm_wcs(header):
    """Build an LSST ``SkyWcs`` from a FITS WCS header.

    Lets an externally-produced mosaic (e.g. a Euclid MER tile, whose WCS is
    normally read with astropy) be carried through the same DM-WCS code paths
    as an LSST coadd -- ``sky_to_pixel``, ``extract_perturbation_dm_wcs``, and
    the ``mergeTask`` field-distortion correction -- instead of maintaining a
    parallel astropy branch.  Standard FITS conventions (1-based ``CRPIX``,
    ``CD`` or ``PC``+``CDELT``) are handled by ``makeSkyWcs``; pixels in the
    returned WCS are 0-based, matching ``astropy.wcs.WCS`` evaluated with
    ``origin=0`` (so ``sky_to_pixel`` agrees between the two backends).

    Parameters
    ----------
    header : astropy.io.fits.Header, fitsio.FITSHDR, mapping, or PropertyList
        FITS header carrying the WCS keywords.

    Returns
    -------
    lsst.afw.geom.SkyWcs
    """
    import lsst.daf.base as dafBase
    from lsst.afw.geom import makeSkyWcs

    if isinstance(header, dafBase.PropertyList):
        return makeSkyWcs(header)

    if hasattr(header, "cards"):  # astropy.io.fits.Header
        pairs = ((c.keyword, c.value) for c in header.cards)
    elif hasattr(header, "records"):  # fitsio.FITSHDR
        pairs = ((r["name"], r["value"]) for r in header.records())
    else:  # plain mapping
        pairs = header.items()

    ps = dafBase.PropertyList()
    for key, value in pairs:
        if not key or key in ("HISTORY", "COMMENT", "CONTINUE", "END"):
            continue
        if hasattr(value, "item"):  # numpy scalar -> python scalar
            value = value.item()
        try:
            ps.set(key, value)
        except Exception:
            pass
    return makeSkyWcs(ps)


def dm_wcs_to_header(wcs):
    """Serialize an LSST ``SkyWcs`` to a FITS WCS header (astropy ``Header``).

    Inverse companion of :func:`header_to_dm_wcs`: emits standard FITS WCS
    keywords so a DM ``SkyWcs`` can be written to disk, handed to astropy (e.g.
    for a WCSAxes plot), or round-tripped back through ``header_to_dm_wcs``.

    Parameters
    ----------
    wcs : lsst.afw.geom.SkyWcs
        The world coordinate system to serialize.

    Returns
    -------
    astropy.io.fits.Header
    """
    from astropy.io import fits

    md = wcs.getFitsMetadata()
    header = fits.Header()
    for name in md.names():
        header[name] = md.getScalar(name)
    return header


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
        dudx=-J_arcsec[0, 0],
        dudy=-J_arcsec[0, 1],
        dvdx=J_arcsec[1, 0],
        dvdy=J_arcsec[1, 1],
        origin=galsim.PositionD(x0, y0),
    )
    world_origin = galsim.CelestialCoord(
        sky_center.getRa().asRadians() * galsim.radians,
        sky_center.getDec().asRadians() * galsim.radians,
    )

    wcs_galsim = galsim.TanWCS(affine=aff, world_origin=world_origin, units=galsim.arcsec)
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

    if wcs_gs.wcs_type == "TAN":
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
        raise RuntimeError("Does not support wcs_gs type: %s" % wcs_gs.wcs_type)

    return wcs_dm


def make_tanwcs_galsim(pixel_scale, g1, g2, rho, ra_deg, dec_deg, x_pix, y_pix):
    """Build a GalSim TanWCS with shear (g1, g2) and rotation (rho).

    Parameters
    ----------
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    g1, g2 : float
        Reduced shear components in lensing convention.
    rho : float
        Rotation angle in radians.
    ra_deg : float
        Right ascension of the tangent point in degrees.
    dec_deg : float
        Declination of the tangent point in degrees.
    x_pix, y_pix : float
        Pixel coordinates of the tangent point.

    Returns
    -------
    galsim.TanWCS
    """
    jac = jacobian_reconstruction(pixel_scale, g1, g2, rho)
    affine = galsim.AffineTransform(
        dudx=jac[0, 0],
        dudy=jac[0, 1],
        dvdx=jac[1, 0],
        dvdy=jac[1, 1],
        origin=galsim.PositionD(x_pix, y_pix),
    )
    world_origin = galsim.CelestialCoord(
        ra_deg * galsim.degrees,
        dec_deg * galsim.degrees,
    )
    return galsim.TanWCS(
        affine=affine,
        world_origin=world_origin,
        units=galsim.arcsec,
    )


def make_tanwcs_dm(pixel_scale, g1, g2, rho, ra_deg, dec_deg, x_pix, y_pix):
    """Build an LSST SkyWcs with shear (g1, g2) and rotation (rho).

    Constructs a GalSim TanWCS via make_tanwcs_galsim and converts
    it to an LSST SkyWcs using tanwcs_galsim2dm.

    Parameters
    ----------
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    g1, g2 : float
        Reduced shear components in lensing convention.
    rho : float
        Rotation angle in radians.
    ra_deg : float
        Right ascension of the tangent point in degrees.
    dec_deg : float
        Declination of the tangent point in degrees.
    x_pix, y_pix : float
        Pixel coordinates of the tangent point.

    Returns
    -------
    lsst.afw.geom.SkyWcs
    """
    wcs_gs = make_tanwcs_galsim(
        pixel_scale,
        g1,
        g2,
        rho,
        ra_deg,
        dec_deg,
        x_pix,
        y_pix,
    )
    return tanwcs_galsim2dm(wcs_gs)


def correct_fpfs_spin2_wcs(data, g1, g2, rho, prefix="", flipu=False):
    """Correct FPFS linear shapelet moments for WCS shear and rotation.

    Transforms the spin-2 shapelet moments measured in pixel coordinates
    to sky coordinates by undoing the WCS-induced shear and rotation.

    The correction is applied in two steps:
    1. Undo shear: subtract the first-order shear response from each
       moment using the relations from the shapelet shear algebra
       (see notes/shapelets.tex).
    2. Undo rotation: rotate spin-s moments by -s*rho.

    Parameters
    ----------
    data : np.ndarray
        Structured array of FPFS shapelet moments with fields
        m00, m20, m22c, m22s, m40, m42c, m42s, m44c, m44s,
        m60, m64c, m64s (dtype ``FpfsShapelets``), optionally with
        a common prefix (e.g. ``"fpfs1_"``).
    g1, g2 : float
        WCS shear components in lensing convention.
    rho : float
        WCS rotation angle in radians.
    prefix : str, optional
        Column name prefix (e.g. ``"fpfs1_"``). Default is ``""``.
    flipu : bool, optional
        If True, flip the output to LSST/FITS convention (u=East)
        from the default GalSim convention (u=West).  This negates
        all sine components (m22s, m42s, m44s, m64s).

    Returns
    -------
    np.ndarray
        Corrected shapelet moments (same dtype as input).
    """
    out = data.copy()
    p = prefix

    # --- Step 1: undo shear ---
    # From shapelets.tex: M_corrected = M_observed - g1*R1 - g2*R2
    # where R1, R2 are the shear correction coefficients matching the
    # measure_shapelets_dg responses in AnaCal's catalog.h.
    sqrt2 = np.sqrt(2.0)
    sqrt3 = np.sqrt(3.0)
    sqrt5 = np.sqrt(5.0)
    sqrt6 = np.sqrt(6.0)

    m00 = data[f"{p}m00"]
    m20 = data[f"{p}m20"]
    m22c = data[f"{p}m22c"]
    m22s = data[f"{p}m22s"]
    m40 = data[f"{p}m40"]
    m42c = data[f"{p}m42c"]
    m42s = data[f"{p}m42s"]
    m44c = data[f"{p}m44c"]
    m44s = data[f"{p}m44s"]
    m60 = data[f"{p}m60"]
    m64c = data[f"{p}m64c"]
    m64s = data[f"{p}m64s"]

    # Shear correction coefficients (same as measure_shapelets_dg in C++)

    dm22c_dg1 = (1.0 / sqrt2) * (m00 - m40) - sqrt3 * m44c
    dm22c_dg2 = -sqrt3 * m44s
    dm22s_dg1 = -sqrt3 * m44s
    dm22s_dg2 = (1.0 / sqrt2) * (m00 - m40) + sqrt3 * m44c

    dm42c_dg1 = (sqrt6 / 2.0) * (m20 - m60) - sqrt5 * m64c
    dm42c_dg2 = -sqrt5 * m64s
    dm42s_dg1 = -sqrt5 * m64s
    dm42s_dg2 = (sqrt6 / 2.0) * (m20 - m60) + sqrt5 * m64c

    # Apply correction: M_corrected = M_observed - g1*dm_dg1 - g2*dm_dg2
    out[f"{p}m22c"] = m22c - g1 * dm22c_dg1 - g2 * dm22c_dg2
    out[f"{p}m22s"] = m22s - g1 * dm22s_dg1 - g2 * dm22s_dg2
    out[f"{p}m42c"] = m42c - g1 * dm42c_dg1 - g2 * dm42c_dg2
    out[f"{p}m42s"] = m42s - g1 * dm42s_dg1 - g2 * dm42s_dg2

    # --- Step 2: undo rotation ---
    # Spin-s moments rotate by exp(-i*s*rho):
    #   m'_c =  m_c * cos(s*rho) + m_s * sin(s*rho)
    #   m'_s = -m_c * sin(s*rho) + m_s * cos(s*rho)

    # Spin-2: m22, m42
    cos2 = np.cos(2.0 * rho)
    sin2 = np.sin(2.0 * rho)

    m22c_s = out[f"{p}m22c"].copy()
    m22s_s = out[f"{p}m22s"].copy()
    out[f"{p}m22c"] = m22c_s * cos2 + m22s_s * sin2
    out[f"{p}m22s"] = -m22c_s * sin2 + m22s_s * cos2

    m42c_s = out[f"{p}m42c"].copy()
    m42s_s = out[f"{p}m42s"].copy()
    out[f"{p}m42c"] = m42c_s * cos2 + m42s_s * sin2
    out[f"{p}m42s"] = -m42c_s * sin2 + m42s_s * cos2
    if flipu:
        out[f"{p}m22s"] = -out[f"{p}m22s"]
        out[f"{p}m42s"] = -out[f"{p}m42s"]
        out[f"{p}m44s"] = -out[f"{p}m44s"]
        out[f"{p}m64s"] = -out[f"{p}m64s"]
    return out


def correct_ellipticity_wcs(ells, g1, g2, rho, prefix="fpfs1_", flipu=False):
    """Correct FPFS ellipticities for WCS shear and rotation.

    Undoes the WCS-induced shear and rotation on the nonlinear
    ellipticity estimates (e1, e2) produced by
    :func:`anacal.fpfs.process_image`.

    The correction is applied in two steps:

    1. **Undo shear**: subtract the shear contribution using the
       diagonal shear response:
       ``e1_corr = e1 - g1 * de1/dg1``,
       ``e2_corr = e2 - g2 * de2/dg2``.
    2. **Undo rotation**: rotate by ``-2*rho``:
       ``e1' =  e1*cos(2*rho) + e2*sin(2*rho)``,
       ``e2' = -e1*sin(2*rho) + e2*cos(2*rho)``.
    3. flip u axis if required

    Parameters
    ----------
    ells : np.ndarray
        Structured array from :func:`anacal.fpfs.process_image` with
        fields ``{prefix}e1``, ``{prefix}e2``, ``{prefix}de1_dg1``,
        ``{prefix}de2_dg2``.
    g1, g2 : float
        WCS shear components in lensing convention.
    rho : float
        WCS rotation angle in radians.
    prefix : str, optional
        Column name prefix (e.g. ``"fpfs1_"``). Default is ``"fpfs1_"``.
    flipu : bool, optional
        If True, flip the output to LSST/FITS convention (u=East)
        from the default GalSim convention (u=West).  This negates
        e2.

    Returns
    -------
    e1_corrected, e2_corrected : np.ndarray
        Corrected ellipticity arrays.
    """
    p = prefix
    # Step 1: Undo shear using response
    e1_shear = ells[f"{p}e1"] - g1 * ells[f"{p}de1_dg1"]
    e2_shear = ells[f"{p}e2"] - g2 * ells[f"{p}de2_dg2"]

    # Step 2: Undo rotation (rotate by -2*rho)
    cos2rho = np.cos(2.0 * rho)
    sin2rho = np.sin(2.0 * rho)
    e1_corrected = e1_shear * cos2rho + e2_shear * sin2rho
    e2_corrected = -e1_shear * sin2rho + e2_shear * cos2rho
    if flipu:
        e2_corrected = -e2_corrected
    return e1_corrected, e2_corrected
