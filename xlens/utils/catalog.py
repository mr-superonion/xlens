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

import numpy as np


def getPatchInner(sources, patchInfo, pixel_scale):
    """Check if each source centroid is in the inner bbox of a patch.

    Uses ``patchInfo.getInnerBBox()`` from the **skymap** (not the
    exposure or MultipleCellCoadd). This is important because:

    - The skymap patch inner bbox defines the non-overlapping tiling of the
      sky. Each sky position belongs to exactly one patch's inner region.
    - Using the skymap inner bbox ensures correct deduplication across
      neighboring patches.

    Parameters
    ----------
    sources : structured array
        Catalog with ``x1``, ``x2`` columns in arcsec.
    patchInfo : `lsst.skymap.PatchInfo`
        Patch info from ``skyMap[tract][patch]``.
    pixel_scale : float
        Pixel scale in arcsec/pixel (to convert x1/x2 to pixel coords).

    Returns
    -------
    isPatchInner : array-like of `bool`
        ``True`` for each source whose centroid falls within the
        skymap's patch inner region.
    """
    from lsst.geom import Box2D

    # Use the skymap's patch inner bbox for deduplication.
    # Do NOT use MultipleCellCoadd.inner_bbox or exposure.getBBox(),
    # which cover the full outer extent including overlap regions.
    innerFloatBBox = Box2D(patchInfo.getInnerBBox())
    inInner = innerFloatBBox.contains(sources["x1"] / pixel_scale, sources["x2"] / pixel_scale)
    return inInner


def getTractInner(sources, tractInfo, skyMap):
    """Set a flag for each source that the skyMap includes in tractInfo.

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceCatalog with pre-calculated centroids.
    tractInfo : `lsst.skymap.TractInfo`
        Tract object
    skyMap : `lsst.skymap.BaseSkyMap`
        Sky tessellation object

    Returns
    -------
    isTractInner : array-like of `bool`
        True if the skyMap.findTract method returns
        the same tract as tractInfo.
    """
    isTractInner = (
        skyMap.findTractIdArray(
            sources["ra"],
            sources["dec"],
            degrees=True,
        )
        == tractInfo.getId()
    )
    return isTractInner


def set_isPrimary(sources, skyMap, tractInfo, patchInfo, pixel_scale):
    """Set isPrimary and related flags on sources.

    For coadded imaging, the `isPrimary` flag returns True when an object is in
    the inner region of a coadd patch, is in the inner region of a coadd tract

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceTable. Reads in centroid fields and an nChild field.
        Writes is-patch-inner, is-tract-inner, and is-primary flags.
    skyMap : `lsst.skymap.BaseSkyMap`
        Sky tessellation object
    tractInfo : `lsst.skymap.TractInfo`, optional
        Tract object; required if ``self.isSingleFrame`` is False.
    patchInfo : `lsst.skymap.PatchInfo`
        Patch object; required if ``self.isSingleFrame`` is False.
    pixel_scale: `float`
        pixel scale
    """
    # Mark whether sources are contained within the inner regions of the
    # given tract/patch
    isPatchInner = getPatchInner(sources, patchInfo, pixel_scale)
    isTractInner = getTractInner(sources, tractInfo, skyMap)
    sources["is_primary"] = isTractInner & isPatchInner
    return


def euler_rotation_matrix(alpha, beta, gamma):
    """Intrinsic ZYZ Euler rotation matrix ``R = Rz(alpha) Ry(beta) Rz(gamma)``.

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles in **degrees** (Z, Y, Z respectively).

    Returns
    -------
    R : `numpy.ndarray`, shape (3, 3)
        Rotation matrix acting on unit column vectors: ``v_rot = R @ v``.
    """
    a, b, g = np.radians([alpha, beta, gamma])
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    rz1 = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rz2 = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
    return rz1 @ ry @ rz2


def rotate_ra_dec(ra, dec, alpha, beta=0.0, gamma=0.0, inverse=False):
    """Rotate sky coordinates by ZYZ Euler angles.

    Converts ``(ra, dec)`` to unit vectors, applies the rotation
    ``R = Rz(alpha) Ry(beta) Rz(gamma)`` (``v_rot = R @ v``), and converts
    back. With a single angle (``beta = gamma = 0``) this is a rotation about
    the pole by ``alpha`` in RA. Pass ``inverse=True`` to apply ``R.T`` (the
    exact inverse of the same angles), so
    ``rotate_ra_dec(*rotate_ra_dec(ra, dec, a, b, g), a, b, g, inverse=True)``
    returns the input.

    Parameters
    ----------
    ra, dec : array-like or float
        Sky coordinates in **degrees**.
    alpha, beta, gamma : float
        ZYZ Euler angles in **degrees** (``beta``, ``gamma`` default to 0).
    inverse : bool, optional
        If True apply the inverse rotation (``R.T``).

    Returns
    -------
    ra_rot, dec_rot : `numpy.ndarray`
        Rotated coordinates in degrees; ``ra_rot`` in ``[0, 360)``.
    """
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    lon, lat = np.radians(ra), np.radians(dec)
    cd = np.cos(lat)
    # unit vectors (same convention as healpy ang2vec(..., lonlat=True))
    v = np.stack([cd * np.cos(lon), cd * np.sin(lon), np.sin(lat)], axis=-1)
    R = euler_rotation_matrix(alpha, beta, gamma)
    if inverse:
        R = R.T
    w = v @ R.T  # rotate each row vector: w_row = R @ v_row
    ra_rot = np.degrees(np.arctan2(w[..., 1], w[..., 0])) % 360.0
    dec_rot = np.degrees(np.arcsin(np.clip(w[..., 2], -1.0, 1.0)))
    return ra_rot, dec_rot
