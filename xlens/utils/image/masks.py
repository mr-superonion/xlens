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

"""Pixel-mask preparation for AnaCal measurements: the default
bad-mask-plane list, per-band mask building, and the
multi-band union mask.

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
        Combined mask as int16 array.
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
    ).astype(np.int16)
    if original_mask_array is None:
        return new_mask
    return (new_mask | original_mask_array.astype(np.int16)).astype(np.int16)


def _union_mask(image_arrays, mask_objects, variance_arrays, badMaskPlanes,
                mask_array=None):
    """Mask that flags a pixel bad if it is bad in ANY band.

    Every band is masked with this same union, so a pixel that one band
    cannot be trusted on does not leak into the coadd through the others.
    """
    out = mask_array
    for image_array, mask_object, variance_array in zip(
        image_arrays, mask_objects, variance_arrays
    ):
        out = prepare_mask(
            image_array,
            mask_object,
            variance_array,
            badMaskPlanes,
            original_mask_array=out,
        )
    return out
