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

"""GAIA bright-star masking helpers.

Two-stage pipeline shared by every task that needs to write a GAIA
bright-star halo mask into an ``anacal.mask`` mask plane:

1. :func:`get_gaia_table` turns a raw GAIA reference catalog (the
   :class:`~lsst.afw.table.SimpleCatalog` returned by
   :meth:`~lsst.meas.algorithms.ReferenceObjectLoader.loadPixelBox`)
   into a wide structured array
   (:data:`GAIA_TABLE_DTYPE`: ``x_in_tract, y_in_tract, gaia_g_mag,
   gaia_source_id, ra, dec``). Doubles as the per-patch GAIA-stars
   data product written by callers that emit one.

2. :func:`build_gaia_xyr` consumes that wide table plus a bounding box
   and returns the ``(x, y, r)`` structured array that
   :func:`anacal.mask.add_bright_star_mask` expects: pixel coordinates
   are made bbox-local and the halo radius per star comes from a
   pluggable ``r(mag)`` model selected by name from
   :data:`STAR_MASK_RADIUS_FUNCS`.  The models return radii in
   **arcsec** so that one model serves surveys with different pixel
   scales; :func:`build_gaia_xyr` divides by the image's pixel scale.

Adding a new model means defining a function ``f(mag_array) ->
radius_arcsec_array`` and registering it::

    from xlens.utils.mask import STAR_MASK_RADIUS_FUNCS

    def isophotal(mag):
        ...

    STAR_MASK_RADIUS_FUNCS["isophotal"] = isophotal

Then any task can opt in via its ``starMaskType: "isophotal"`` config.
Stars with ``r <= 0`` are dropped, so a model can express its own
magnitude cut by returning ``0`` for stars it does not want to mask.
"""
from __future__ import annotations

from typing import Any, Callable

import astropy.units as au
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GAIA_TABLE_DTYPE",
    "default_gaia_radius",
    "DP2_nosim_gaia_radius",
    "no_mask_gaia_radius",
    "STAR_MASK_RADIUS_FUNCS",
    "get_gaia_table",
    "build_gaia_xyr",
    "add_galaxy_ellipse_mask",
]


GAIA_TABLE_DTYPE = np.dtype(
    [
        ("x_in_tract", np.float64),
        ("y_in_tract", np.float64),
        ("gaia_g_mag", np.float64),
        ("gaia_source_id", np.int64),
        ("ra", np.float64),
        ("dec", np.float64),
    ]
)


def default_gaia_radius(mag: NDArray) -> NDArray:
    """Step function ``r(mag)`` in arcsec, tuned on HSC (0.168"/px).

    ``r = 450`` px for ``mag <= 11``, ``200`` px for ``11 < mag <= 14``,
    ``100`` px for ``14 < mag <= 20`` on the HSC grid, i.e. 75.6, 33.6
    and 16.8 arcsec; ``0`` otherwise (those stars are dropped by
    :func:`build_gaia_xyr`).
    """
    mag = np.asarray(mag, dtype=np.float64)
    # step defined in HSC pixels (0.168 arcsec/pixel) -> arcsec
    return 0.168 * np.select(
        [mag <= 11.0, mag <= 14.0, mag <= 20.0],
        [450.0, 200.0, 100.0],
        default=0.0,
    )


def DP2_nosim_gaia_radius(mag: NDArray) -> NDArray:
    """Piecewise power-law fit to DP2 data, ``r(mag)`` in arcsec.

    Fitted in LSST pixels (0.2"/px): ``log10 r = -0.12 mag + 3.8389``
    for ``mag <= 15.5``, ``-0.0767 mag + 2.772`` for
    ``15.5 < mag <= 20``, ``0`` otherwise (those stars are dropped by
    :func:`build_gaia_xyr`).
    """
    mag = np.asarray(mag, dtype=np.float64)
    # fit defined in LSST pixels (0.2 arcsec/pixel) -> arcsec
    return 0.2 * np.select(
        [mag <= 15.5, mag <= 20.0],
        [10 ** (-0.12 * mag + 3.8389), 10 ** (-0.0767 * mag + 2.772)],
        default=0.0,
    )


def no_mask_gaia_radius(mag: NDArray) -> NDArray:
    """Flat ``r = 2`` arcsec (10 LSST px) for every GAIA star with
    ``mag <= 20``; ``0`` otherwise (those stars are dropped).

    Practically disables bright-star wing masking but still drops a
    small placeholder halo at every star position.
    """
    mag = np.asarray(mag, dtype=np.float64)
    return np.where(mag <= 20.0, 2.0, 0.0)


STAR_MASK_RADIUS_FUNCS: dict[str, Callable[[NDArray], NDArray]] = {
    "default": default_gaia_radius,
    "DP2_nosim": DP2_nosim_gaia_radius,
    "no_mask": no_mask_gaia_radius,
}


def get_gaia_table(gaia_catalog: Any, wcs: Any) -> NDArray:
    """Raw GAIA refCat → wide :data:`GAIA_TABLE_DTYPE` structured array.

    Pixel coordinates ``(x_in_tract, y_in_tract)`` come from
    ``wcs.skyToPixelArray``, so they live on whatever pixel grid
    ``wcs`` defines (usually the tract grid for a patch-level WCS).
    ``ra``/``dec`` are returned in degrees (refcat
    ``coord_ra``/``coord_dec`` are stored as Angle/radians, so we
    convert).
    """
    gaia_astropy = gaia_catalog.asAstropy()
    flux = gaia_astropy["phot_g_mean_flux"]
    mag = (np.asarray(flux) * au.nJy).to_value(au.ABmag)
    ra_deg = np.asarray(gaia_astropy["coord_ra"], dtype=np.float64) * (180.0 / np.pi)
    dec_deg = np.asarray(gaia_astropy["coord_dec"], dtype=np.float64) * (180.0 / np.pi)
    x, y = wcs.skyToPixelArray(ra=ra_deg, dec=dec_deg, degrees=True)
    out = np.empty(len(mag), dtype=GAIA_TABLE_DTYPE)
    out["x_in_tract"] = np.asarray(x, dtype=np.float64)
    out["y_in_tract"] = np.asarray(y, dtype=np.float64)
    out["gaia_g_mag"] = mag
    out["gaia_source_id"] = np.asarray(gaia_astropy["id"], dtype=np.int64)
    out["ra"] = ra_deg
    out["dec"] = dec_deg
    return out


def build_gaia_xyr(
    gaia_table: NDArray,
    bbox: Any,
    *,
    pixel_scale: float,
    star_mask_type: str = "default",
    mag_max: float | None = None,
) -> NDArray | None:
    """Wide ``gaia_table`` + bbox → ``(x, y, r)`` structured array for
    :func:`anacal.mask.add_bright_star_mask`.

    Pixel coordinates are made bbox-local (``bbox.getBeginX/Y`` is
    subtracted from ``x_in_tract`` / ``y_in_tract``). The per-star
    halo radius is ``STAR_MASK_RADIUS_FUNCS[star_mask_type](mag)``,
    which is in arcsec, divided by ``pixel_scale`` (arcsec/pixel) so
    that ``r`` is in pixels of the image being masked; rows with
    ``r <= 0`` are dropped, so each radius function can encode its
    own magnitude cut by returning ``0`` for stars it does not want to
    mask.

    ``mag_max`` tightens that cut without touching the radius model:
    stars fainter than it are dropped whatever radius the model gave
    them. ``None`` (the default) leaves the model's own limit in force
    -- 20 for ``'default'``.

    Returns ``None`` when no stars pass the radius cut.
    """
    try:
        radius_func = STAR_MASK_RADIUS_FUNCS[star_mask_type]
    except KeyError as exc:
        raise ValueError(
            f"unknown star_mask_type {star_mask_type!r}; "
            f"available: {sorted(STAR_MASK_RADIUS_FUNCS)}"
        ) from exc

    if not pixel_scale > 0.0:
        raise ValueError(f"pixel_scale must be positive, got {pixel_scale}")
    mag = gaia_table["gaia_g_mag"]
    r = radius_func(mag) / float(pixel_scale)
    if mag_max is not None:
        r = np.where(mag > mag_max, 0.0, r)
    keep = r > 0
    if not np.any(keep):
        return None

    x = gaia_table["x_in_tract"][keep] - bbox.getBeginX()
    y = gaia_table["y_in_tract"][keep] - bbox.getBeginY()
    out = np.zeros(int(keep.sum()),
                   dtype=np.dtype([("x", float), ("y", float), ("r", float)]))
    out["x"] = x
    out["y"] = y
    out["r"] = r[keep]
    return out


def add_galaxy_ellipse_mask(
    mask_array: NDArray,
    *,
    ra: NDArray,
    dec: NDArray,
    d25_arcmin: NDArray,
    logr25: NDArray,
    pa: NDArray,
    wcs: Any,
    bbox: Any,
    scale: float = 2.5,
    d25_min: float = 0.3,
    rmax_arcsec: float = 300.0,
    value: int = 1,
) -> int:
    """OR large-galaxy ellipses into ``mask_array`` (bbox-local), in place.

    The catalog side of the lsst-starsub large-galaxy mask (E. Sheldon,
    ``lsst_starsub/galaxies.py``): each galaxy of ``d25_arcmin >= d25_min``
    is masked by its D25 ellipse -- semi-major axis ``scale * D25 / 2``
    (capped at ``rmax_arcsec``), axis ratio ``10**-logr25`` (1 when NaN,
    floored at 0.05), position angle ``pa`` east of north (0 when NaN).
    lsst-starsub sizes the mask from the image segmentation instead; with
    no segmentation the D25 ellipse is scaled (2.5: the blend groups of
    its DP2 test region reached ~2.5 x D25/2).  The ellipse is drawn in
    the galaxy's local tangent plane through ``wcs`` (east and north taken
    from the WCS), so it does not depend on the image orientation.

    Parameters
    ----------
    mask_array : (ny, nx) integer array
        Indexed from ``bbox``'s minimum corner; ``value`` is OR-ed in.
    ra, dec : arrays, degrees
    d25_arcmin, logr25, pa : arrays
    wcs : lsst.afw.geom.SkyWcs
    bbox : lsst.geom.Box2I
    scale, d25_min, rmax_arcsec : float

    Returns
    -------
    ngal : int
        Galaxies that masked at least one pixel.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    d25 = np.asarray(d25_arcmin, dtype=np.float64)
    keep = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(d25) & (d25 >= d25_min)
    if not keep.any():
        return 0
    ra, dec, d25 = ra[keep], dec[keep], d25[keep]
    logr = np.asarray(logr25, dtype=np.float64)[keep]
    q = np.clip(np.where(np.isfinite(logr), 10.0 ** (-logr), 1.0), 0.05, 1.0)
    pa = np.asarray(pa, dtype=np.float64)[keep]
    pa = np.deg2rad(np.where(np.isfinite(pa), pa, 0.0))
    ny, nx = mask_array.shape
    x0, y0 = bbox.getMinX(), bbox.getMinY()
    a_arc = np.minimum(scale * d25 * 30.0, rmax_arcsec)  # D25/2 in arcsec x scale
    # local Jacobian from 1 arcsec steps east and north
    step = 1.0 / 3600.0
    cx, cy = wcs.skyToPixelArray(ra, dec, degrees=True)
    ex, ey = wcs.skyToPixelArray(
        ra + step / np.cos(np.deg2rad(dec)), dec, degrees=True
    )
    nx_, ny_ = wcs.skyToPixelArray(ra, dec + step, degrees=True)
    ex, ey, nx_, ny_ = ex - cx, ey - cy, nx_ - cx, ny_ - cy
    pix_per_arcsec = np.sqrt(np.abs(ex * ny_ - ey * nx_))
    reach = a_arc * pix_per_arcsec * 1.01 + 1.0
    lx, ly = cx - x0, cy - y0
    near = (
        (lx + reach >= 0) & (lx - reach <= nx - 1)
        & (ly + reach >= 0) & (ly - reach <= ny - 1)
    )
    ngal = 0
    for i in np.flatnonzero(near):
        jac = np.array([[ex[i], nx_[i]], [ey[i], ny_[i]]])  # (dx,dy)=J(xi,eta)
        jinv = np.linalg.inv(jac)
        ix0 = int(max(np.floor(lx[i] - reach[i]), 0))
        ix1 = int(min(np.ceil(lx[i] + reach[i]) + 1, nx))
        iy0 = int(max(np.floor(ly[i] - reach[i]), 0))
        iy1 = int(min(np.ceil(ly[i] + reach[i]) + 1, ny))
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        dx = np.arange(ix0, ix1, dtype=np.float64)[None, :] - lx[i]
        dy = np.arange(iy0, iy1, dtype=np.float64)[:, None] - ly[i]
        xi = jinv[0, 0] * dx + jinv[0, 1] * dy    # arcsec east
        eta = jinv[1, 0] * dx + jinv[1, 1] * dy   # arcsec north
        sp, cp = np.sin(pa[i]), np.cos(pa[i])
        u = xi * sp + eta * cp                    # along the major axis
        v = xi * cp - eta * sp
        a = a_arc[i]
        b = a * q[i]
        inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
        if inside.any():
            sub = mask_array[iy0:iy1, ix0:ix1]
            sub[inside] |= value
            ngal += 1
    return ngal
