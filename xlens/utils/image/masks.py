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

"""Pixel-mask building for the systematics tasks: the default
bad-mask-plane list and the per-band mask builder.

Mask building happens ONLY in the systematics tasks
(``BuildSystematicsTask`` / ``BuildCellSystematicsTask``); the measurement
``prepare_data*`` functions read the resulting ``mask_array`` and never
build masks themselves.

Split out of ``xlens.utils.image``, which re-exports every public name here
for backward compatibility.
"""

from typing import List

import lsst.pex.exceptions as pexExcept
import numpy as np
from numpy.typing import NDArray

badMaskDefault = [
    "BAD",
    "SAT",
    "CR",
    "NO_DATA",
    "UNMASKEDNAN",
    "CROSSTALK",
    "INTRP",
    "STREAK",
    "VIGNETTED",
    "CLIPPED",
]


def prepare_mask(
    image_array: NDArray,
    mask_object,
    variance_array: NDArray,
    badMaskPlanes: List[str],
    original_mask_array: NDArray | None = None,
) -> NDArray:
    """Build a combined mask from raw mask, variance, and image planes.

    Resolves ``badMaskPlanes`` against ``mask_object`` (an LSST
    ``Mask`` instance) to compute the bitmask, OR-flags any pixel
    matching that bitmask, and additionally flags pixels whose image
    value drops below -6 sigma (using the variance plane).

    If ``original_mask_array`` is provided (e.g., a pre-existing
    caller-supplied mask), the freshly computed mask is OR-ed with
    it; otherwise the freshly computed mask is returned as-is.

    Parameters
    ----------
    image_array : NDArray
        Science image array.
    mask_object : lsst.afw.image.Mask
        Mask object exposing ``.array`` and ``getPlaneBitMask(plane)``.
        Plane names absent from this mask are silently skipped.
    variance_array : NDArray
        Variance plane.
    badMaskPlanes : list of str
        Mask plane names to flag.
    original_mask_array : NDArray or None, optional
        Pre-existing mask to OR into the result.  ``None`` (default)
        means start from scratch.

    Returns
    -------
    NDArray
        Combined 0/1 mask as uint8 (bit 0 of the anacal mask
        convention; anacal's in-place mask functions require uint8).
    """
    bitv = 0
    for plane in badMaskPlanes:
        try:
            bitv |= mask_object.getPlaneBitMask(plane)
        except pexExcept.InvalidParameterError:
            pass
    mask_array_raw = mask_object.array
    new_mask = (
        ((mask_array_raw & bitv) != 0)
        | (image_array < (-6.0 * np.sqrt(np.where(variance_array < 0, 0, variance_array))))
    ).astype(np.uint8)
    if original_mask_array is None:
        return new_mask
    return (
        new_mask | (np.asarray(original_mask_array) != 0).astype(np.uint8)
    ).astype(np.uint8)


# ----------------------------------------------------------------------
# Run-length encoding for small-valued patch masks
# ----------------------------------------------------------------------
# A systematics mask stores a couple of bits per pixel but was written
# as a full int32 image: 67 MB per patch, 2.7 TB for an HSC-wide run,
# almost all of it zeros. RLE stores one row per CONTIGUOUS
# constant-value run instead -- (y, x_start, x_end, value) -- which for
# the typical bright-star-halo morphology is ~1000x smaller.
#
# Convention: ``x_end`` is EXCLUSIVE, i.e. the run is
# ``mask[y, x_start:x_end]``, matching numpy slicing. ``value`` is the
# (nonzero) pixel value of the run, so a single table carries the whole
# combined anacal bitmask (bit 0 = masked, bit 1 = discontinuity;
# values 1..3). Tables written before the value column existed decode
# as 0/1.
RLE_DTYPE = np.dtype(
    [
        ("y", np.int32),
        ("x_start", np.int32),
        ("x_end", np.int32),
        ("value", np.uint8),
    ]
)


def mask_to_rle(mask_array: NDArray) -> NDArray:
    """Run-length encode a 2-D mask, preserving pixel values.

    Returns a :data:`RLE_DTYPE` structured array with one row per
    contiguous constant-value nonzero run; ``x_end`` is exclusive. The
    image shape is NOT stored here -- carry it alongside (see
    :func:`mask_to_rle_table`).
    """
    arr = np.asarray(mask_array)
    if arr.ndim != 2:
        raise ValueError("mask_to_rle expects a 2-D array")
    if arr.dtype.kind == "f" or (arr.size and (arr.min() < 0 or arr.max() > 255)):
        raise ValueError("mask_to_rle expects integer values in [0, 255]")
    arr = arr.astype(np.uint8, copy=False)
    ny, nx = arr.shape
    padded = np.zeros((ny, nx + 2), dtype=np.int16)
    padded[:, 1:-1] = arr
    step = np.diff(padded, axis=1)
    # Boundary at step index x = value change entering image pixel x
    # (x == nx is the trailing pad, only ever the end of a run). Both
    # pads are 0, so every row's boundaries open and close within the
    # row and row-major order pairs each nonzero-entry boundary with
    # the NEXT boundary as its exclusive end.
    ys, xs = np.nonzero(step != 0)
    vals = np.where(xs < nx, arr[ys, np.minimum(xs, nx - 1)], 0)
    starts = np.nonzero(vals != 0)[0]
    out = np.empty(len(starts), dtype=RLE_DTYPE)
    out["y"] = ys[starts]
    out["x_start"] = xs[starts]
    out["x_end"] = xs[starts + 1]
    out["value"] = vals[starts]
    return out


def rle_to_mask(rle: NDArray, shape, dtype=np.uint8) -> NDArray:
    """Decode :func:`mask_to_rle` output back to a mask image.

    Vectorised via a flattened difference array: +value at each run
    start, -value at each (exclusive) end, cumulative sum restores the
    mask (runs never overlap, so values add back exactly). An end at
    ``x_end == nx`` lands exactly on the next row's first element and
    cancels there before any of that row's own starts, so runs never
    bleed across rows. Input lacking a ``value`` field (the original
    binary encoding) decodes as 0/1.
    """
    ny, nx = shape
    if rle.dtype.names and "value" in rle.dtype.names:
        weights = np.asarray(rle["value"], dtype=np.int32)
    else:
        weights = np.ones(len(rle), dtype=np.int32)
    diff = np.zeros(ny * nx + 1, dtype=np.int32)
    y = np.asarray(rle["y"], dtype=np.int64)
    np.add.at(
        diff, y * nx + np.asarray(rle["x_start"], dtype=np.int64), weights
    )
    np.add.at(
        diff, y * nx + np.asarray(rle["x_end"], dtype=np.int64), -weights
    )
    return np.cumsum(diff[:-1]).reshape(ny, nx).astype(dtype)


def mask_to_rle_table(mask_array: NDArray, x0: int = 0, y0: int = 0):
    """RLE-encode into an ``astropy`` Table carrying shape and origin.

    ``meta['MASK_NY'] / meta['MASK_NX']`` record the shape and
    ``meta['MASK_X0'] / meta['MASK_Y0']`` the pixel origin (XY0 of the
    source ``MaskX``; the cell-coadd chain needs it to place the
    stitched mask on the tract grid), so :func:`rle_table_to_mask` and
    :func:`rle_table_origin` can reconstruct everything without
    external knowledge. This is the payload of the
    ``*_systematics_mask_rle`` butler datasets and of the file-based
    mask products.
    """
    from astropy.table import Table

    rle = mask_to_rle(mask_array)
    tab = Table(rle)
    tab.meta["MASK_NY"] = int(mask_array.shape[0])
    tab.meta["MASK_NX"] = int(mask_array.shape[1])
    tab.meta["MASK_X0"] = int(x0)
    tab.meta["MASK_Y0"] = int(y0)
    return tab


def rle_table_origin(table) -> tuple[int, int]:
    """(x0, y0) origin stored by :func:`mask_to_rle_table` (0,0 if absent)."""
    meta = {k.upper(): v for k, v in table.meta.items()}
    return int(meta.get("MASK_X0", 0)), int(meta.get("MASK_Y0", 0))


def rle_table_to_mask(table, dtype=np.uint8) -> NDArray:
    """Decode a :func:`mask_to_rle_table` Table back to a mask image.

    Tables written before the ``value`` column existed decode as 0/1.
    """
    meta = {k.upper(): v for k, v in table.meta.items()}
    shape = (int(meta["MASK_NY"]), int(meta["MASK_NX"]))
    if len(table) == 0:
        return np.zeros(shape, dtype=dtype)
    rle = np.empty(len(table), dtype=RLE_DTYPE)
    for col in ("y", "x_start", "x_end"):
        rle[col] = np.asarray(table[col])
    rle["value"] = (
        np.asarray(table["value"]) if "value" in table.colnames else 1
    )
    return rle_to_mask(rle, shape, dtype=dtype)
