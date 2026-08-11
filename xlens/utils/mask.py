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
   :data:`STAR_MASK_RADIUS_FUNCS`.

Adding a new model means defining a function ``f(mag_array) ->
radius_array`` and registering it::

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
    "no_mask_gaia_radius",
    "STAR_MASK_RADIUS_FUNCS",
    "get_gaia_table",
    "build_gaia_xyr",
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
    """Step function ``r(mag)`` in pixels.

    ``r = 450`` for ``mag <= 11``, ``200`` for ``11 < mag <= 14``,
    ``100`` for ``14 < mag <= 20``, ``0`` otherwise (those stars are
    dropped by :func:`build_gaia_xyr`).
    """
    mag = np.asarray(mag, dtype=np.float64)
    return np.select(
        [mag <= 11.0, mag <= 14.0, mag <= 20.0],
        [450.0, 200.0, 100.0],
        default=0.0,
    )


def no_mask_gaia_radius(mag: NDArray) -> NDArray:
    """Flat ``r = 10`` px for every GAIA star with ``mag <= 20``;
    ``0`` otherwise (those stars are dropped).

    Practically disables bright-star wing masking but still drops a
    small placeholder halo at every star position.
    """
    mag = np.asarray(mag, dtype=np.float64)
    return np.where(mag <= 20.0, 10.0, 0.0)


STAR_MASK_RADIUS_FUNCS: dict[str, Callable[[NDArray], NDArray]] = {
    "default": default_gaia_radius,
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
    star_mask_type: str = "default",
    mag_max: float | None = None,
) -> NDArray | None:
    """Wide ``gaia_table`` + bbox → ``(x, y, r)`` structured array for
    :func:`anacal.mask.add_bright_star_mask`.

    Pixel coordinates are made bbox-local (``bbox.getBeginX/Y`` is
    subtracted from ``x_in_tract`` / ``y_in_tract``). The per-star
    halo radius comes from
    ``STAR_MASK_RADIUS_FUNCS[star_mask_type](mag)``; rows with
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

    mag = gaia_table["gaia_g_mag"]
    r = radius_func(mag)
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
