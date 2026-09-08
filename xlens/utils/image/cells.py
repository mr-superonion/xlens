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

"""Measurement-input preparation for cell-based coadds
(``SingleCellCoadd``), single- and multi-band.

Split out of ``xlens.utils.image``, which re-exports every public name here
for backward compatibility.
"""

from typing import List, Sequence

import anacal
import astropy
import numpy as np
from numpy.typing import NDArray

from ..constants import MAG_ZERO_AB
from .noise import estimate_noise_variance, prepare_noise_array
from .prepare import _stack_bands, prepare_detection
from .psf import prepare_psf_array_cell


def prepare_data_one_cell(
    *,
    cell,
    band: str | None,
    seed: int,
    mag_zero: float,
    npix: int = 64,
    do_noise_bias_correction: bool = True,
    skyMap=None,
    tract: int = 0,
    patch: int = 0,
    star_cat: NDArray | None = None,
    psf_array: NDArray | None = None,
    mask_array: NDArray | None = None,
    noise_array: NDArray | None = None,
    detection: astropy.table.Table | None = None,
    cells: List | None = None,
    survey: str | None = None,
    **kwargs,
):
    """Collect metadata and auxiliary arrays from a SingleCellCoadd.

    ``mask_array`` is consumed as-is: mask building lives in
    ``BuildCellSystematicsTask``, and callers pass the cell's slice of
    its stitched output here.  ``None`` means no masking.

    ``survey`` (when given) makes the noise seed survey-aware and sets the
    ``{survey}_{band}_`` output-column prefix.
    """
    outer = cell.outer
    wcs = cell.wcs
    pixel_scale = float(wcs.getPixelScale().asArcseconds())

    bbox = outer.bbox
    # float32 and a real copy, for the reasons given in prepare_data
    gal_array = np.array(outer.image.array, dtype=np.float32)

    if psf_array is None:
        psf_array = prepare_psf_array_cell(cell, npix)

    # Private uint8 copy: mask_galaxy_image can write bright-star halos into
    # the mask it is given, and the caller's array must not be mutated.
    if mask_array is None:
        mask_array = np.zeros(gal_array.shape, dtype=np.uint8)
    else:
        mask_array = np.array(mask_array, dtype=np.uint8)

    anacal.mask.mask_galaxy_image(gal_array, mask_array, star_cat)

    noise_variance = estimate_noise_variance(
        outer.variance.array,
        outer.mask,
        mask_array,
    )

    # The noise plane must be the cell coadd's own stored realization, built
    # by the DM pipeline from the input visits.  Never fall back to generating
    # pure noise here: a generated plane would not carry the coadd's noise
    # correlations, and the patch-level seed would hand every cell in the
    # patch the same realization.
    if do_noise_bias_correction and noise_array is None:
        noise_reals = outer.noise_realizations
        if len(noise_reals) == 0:
            raise RuntimeError(
                "noise bias correction needs a noise realization stored in "
                "the cell coadd, but this cell has none; build the cell "
                "coadds with noise realizations or set "
                "do_noise_bias_correction=False"
            )
        noise_array = np.array(
            noise_reals[0].array,
            dtype=np.float32,
        )

    noise_array = prepare_noise_array(
        noise_array=noise_array,
        do_noise_bias_correction=do_noise_bias_correction,
        gal_shape=gal_array.shape,
        pixel_scale=pixel_scale,
        seed=seed,
        band=band,
        noise_variance=noise_variance,
        mask_array=mask_array,
        star_cat=star_cat,
        survey=survey,
    )

    if skyMap is not None:
        tractInfo = skyMap[tract]
        patchInfo = tractInfo[patch]
    else:
        tractInfo = None
        patchInfo = None

    beginx = bbox.getMinX()
    beginy = bbox.getMinY()
    detection = prepare_detection(
        detection,
        pixel_scale=pixel_scale,
        beginx=beginx,
        beginy=beginy,
        cells=cells,
    )

    # Normalize the image onto the fixed AB zeropoint; the measurement
    # then runs at mag_zero=MAG_ZERO_AB independent of the cell's native
    # mag_zero. No-op (no allocation) for a native-31.4 coadd.
    gal_array, noise_array, noise_variance = anacal.utils.rescale_image_to_zeropoint(
        gal_array, noise_array, noise_variance, mag_zero, MAG_ZERO_AB,
    )

    return {
        "pixel_scale": pixel_scale,
        "mag_zero": MAG_ZERO_AB,
        "noise_variance": noise_variance,
        "gal_array": gal_array,
        "psf_array": psf_array,
        "mask_array": mask_array,
        "noise_array": noise_array,
        "begin_x": beginx,
        "begin_y": beginy,
        "wcs": wcs,
        "skyMap": skyMap,
        "tractInfo": tractInfo,
        "patchInfo": patchInfo,
        "detection": detection,
        "cells": cells,
        "psf_object": None,
        "lsst_psf": None,
        "base_column_name": (
            None if band is None
            else (f"{survey}_{band}_" if survey is not None else band + "_")
        ),
    }


def prepare_data_one_cell_multiband(
    *,
    bands: Sequence[str],
    lsst_cells: dict,
    seed: int,
    mag_zeros: dict,
    npix: int = 64,
    do_noise_bias_correction: bool = True,
    skyMap=None,
    tract: int = 0,
    patch: int = 0,
    star_cat: NDArray | None = None,
    mask_array: NDArray | None = None,
    detection: astropy.table.Table | None = None,
    cells: List | None = None,
    survey: str | None = None,
    **kwargs,
):
    """Cell-coadd counterpart of :func:`prepare_data_multiband`.

    ``lsst_cells`` maps band to the ``SingleCellCoadd`` at the same cell
    id, and ``mag_zeros`` gives that band coadd's native zeropoint;
    ``cells`` is the AnaCal cell list for the measurement.

    ``mask_array`` (the cell's slice of the stitched systematics mask,
    already a union over ALL bands plus bright stars) is handed unchanged
    to every band, so every band is masked on the same pixel footprint.
    """
    bands = list(bands)
    if len(bands) == 0:
        raise ValueError("prepare_data_one_cell_multiband needs at least one band")
    if len(set(bands)) != len(bands):
        raise ValueError(f"duplicate entries in bands: {bands}")
    missing = [b for b in bands if b not in lsst_cells]
    if missing:
        raise KeyError(f"no cell for band(s) {missing}")

    def per_band():
        for band in bands:
            yield prepare_data_one_cell(
                cell=lsst_cells[band],
                band=band,
                seed=seed,
                mag_zero=mag_zeros[band],
                npix=npix,
                do_noise_bias_correction=do_noise_bias_correction,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                star_cat=star_cat,
                mask_array=mask_array,
                detection=detection,
                cells=cells,
                survey=survey,
            )

    return _stack_bands(per_band(), bands)
