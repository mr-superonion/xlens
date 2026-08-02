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

"""Measurement-input preparation for exposures and patch coadds:
block lists, detection-catalog preparation, and the single- and
multi-band ``prepare_data`` entry points.

Split out of ``xlens.utils.image``, which re-exports every public name here
for backward compatibility.
"""

from typing import List, Sequence

import anacal
import astropy
import lsst.geom as lsst_geom
import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..constants import MAG_ZERO_AB
from .masks import _union_mask, badMaskDefault, prepare_mask
from .noise import estimate_noise_variance, prepare_noise_array
from .psf import prepare_psf_array, resize_array


def get_blocks(*, lsst_psf, lsst_bbox, pixel_scale, npix, psf_array):
    min_corner = lsst_bbox.getMin()
    x_min, y_min = min_corner.getX(), min_corner.getY()
    width, height = lsst_bbox.getWidth(), lsst_bbox.getHeight()
    # Create blocks
    blocks = anacal.geometry.get_block_list(
        img_ny=height,
        img_nx=width,
        block_nx=250,
        block_ny=250,
        block_overlap=80,
        scale=pixel_scale,
    )
    new_blocks = []
    for bb in blocks:
        # Center of the block
        x0 = int(np.clip(bb.xcen, 0, width - 1))
        y0 = int(np.clip(bb.ycen, 0, height - 1))
        try:
            this_psf = lsst_psf.computeImage(lsst_geom.Point2D(x_min + x0, y_min + y0)).getArray()
            bb.psf_array = resize_array(this_psf, (npix, npix))
        except Exception:
            continue
        new_blocks.append(bb)
    return new_blocks


def get_blocks_multiband(*, lsst_psfs, lsst_bbox, pixel_scale, npix):
    """Blocks whose PSF stamp is a ``(nband, npix, npix)`` stack.

    Same geometry as :func:`get_blocks`; the only difference is that each
    block carries one PSF per band instead of one.  A block is dropped
    unless *every* band can supply a PSF at its centre, so all bands are
    detected on exactly the same set of blocks.

    Parameters
    ----------
    lsst_psfs : sequence
        One LSST PSF model per band, in the same order as the image stack
        that will be handed to anacal.
    """
    if len(lsst_psfs) == 0:
        raise ValueError("get_blocks_multiband needs at least one PSF")

    min_corner = lsst_bbox.getMin()
    x_min, y_min = min_corner.getX(), min_corner.getY()
    width, height = lsst_bbox.getWidth(), lsst_bbox.getHeight()
    blocks = anacal.geometry.get_block_list(
        img_ny=height,
        img_nx=width,
        block_nx=250,
        block_ny=250,
        block_overlap=80,
        scale=pixel_scale,
    )
    new_blocks = []
    for bb in blocks:
        x0 = int(np.clip(bb.xcen, 0, width - 1))
        y0 = int(np.clip(bb.ycen, 0, height - 1))
        point = lsst_geom.Point2D(x_min + x0, y_min + y0)
        stamps = []
        try:
            for lsst_psf in lsst_psfs:
                stamps.append(
                    resize_array(
                        lsst_psf.computeImage(point).getArray(),
                        (npix, npix),
                    )
                )
        except Exception:
            continue
        bb.psf_array = np.asarray(stamps, dtype=np.float64)
        new_blocks.append(bb)
    return new_blocks


def combine_sim_exposures(
    exposures: Sequence,
    noises: Sequence[NDArray],
):
    """Combine simulated exposures using inverse-variance weights."""

    if len(exposures) != len(noises):
        raise ValueError("exposure and noises should have the same length")
    if len(exposures) <= 0:
        raise ValueError("no elements in the input list")

    reference_shape = exposures[0].getMaskedImage().image.array.shape
    combined_image = np.zeros(reference_shape, dtype=np.float32)
    combined_noise = np.zeros(reference_shape, dtype=noises[0].dtype)
    total_weight = 0.0

    for exposure, noise_array in zip(exposures, noises):
        image = exposure.getMaskedImage().image.array
        variance = exposure.getMaskedImage().variance.array

        if image.shape != reference_shape:
            raise ValueError("All exposures must share the same image shape")

        finite_variance = variance[np.isfinite(variance)]
        if finite_variance.size == 0:
            raise ValueError("Variance plane must contain at least one finite value")
        variance_value = float(np.nanmean(variance))
        if not np.isfinite(variance_value):
            raise ValueError("Variance mean must be finite")
        if variance_value <= 0:
            raise ValueError("Variance values must be positive")

        weight = 1.0 / variance_value
        combined_image += weight * image
        combined_noise += weight * noise_array
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("Total weight must be positive")

    combined_image = combined_image / total_weight
    combined_noise = combined_noise / total_weight
    combined_variance = 1.0 / total_weight

    combined_exposure = exposures[0].clone()
    image_plane = combined_exposure.getMaskedImage().image.array
    variance_plane = combined_exposure.getMaskedImage().variance.array

    image_plane[:, :] = combined_image.astype(image_plane.dtype, copy=False)
    variance_plane[:, :] = combined_variance

    return combined_exposure, combined_noise


def prepare_detection(
    detection: astropy.table.Table | NDArray | None,
    *,
    pixel_scale: float,
    beginx: int,
    beginy: int,
    blocks: List | None = None,
) -> NDArray | None:
    """Prepare a detection catalog for forced measurement.

    Copies the catalog and selects anacal columns.  Block ownership is no
    longer assigned here: AnaCal recomputes block_id internally from
    x1_det/x2_det (Task::assign_block_ids), and block_id is not a catalog
    column any more.

    Parameters
    ----------
    detection : astropy.table.Table, NDArray, or None
        Input detection catalog.
    pixel_scale : float
        Pixel scale in arcseconds.  Unused, kept for caller compatibility.
    beginx, beginy : int
        Image origin in pixel coordinates.  Unused, kept for caller
        compatibility.
    blocks : list or None
        Unused, kept for caller compatibility.

    Returns
    -------
    NDArray or None
        Prepared detection catalog, or None if input is None.
    """
    if detection is None:
        return None
    if isinstance(detection, astropy.table.Table):
        detection = detection.copy().as_array()
    elif isinstance(detection, np.ndarray):
        detection = detection.copy()
    detection = rfn.repack_fields(detection[list(anacal.table.column_names())])
    return detection


def prepare_data(
    *,
    band: str | None,
    exposure,
    seed: int,
    noiseId: int = 0,
    rotId: int = 0,
    npix: int = 64,
    noise_corr: NDArray | None = None,
    do_noise_bias_correction: bool = True,
    badMaskPlanes: List[str] = badMaskDefault,
    skyMap=None,
    tract: int = 0,
    patch: int = 0,
    star_cat: NDArray | None = None,
    psf_array: NDArray | None = None,
    mask_array: NDArray | None = None,
    noise_array: NDArray | None = None,
    detection: astropy.table.Table | None = None,
    blocks: List | None = None,
    survey: str | None = None,
    **kwargs,
):
    """Collect metadata and auxiliary arrays from an LSST ExposureF.

    ``survey`` (when given) makes the noise-realisation seed
    survey-aware and is used by callers to build the
    ``{survey}_{band}_`` output-column prefix.
    """
    pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
    mag_zero = np.log10(exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()) / 0.4
    wcs = exposure.getWcs()
    lsst_bbox = exposure.getBBox()

    if psf_array is None:
        psf_array = prepare_psf_array(exposure, npix)

    # float32, the dtype the coadd is already stored in (ExposureF). AnaCal
    # takes the science and noise planes at single precision and widens to
    # double once, when the pixels are copied into the FFT buffer, so a float64
    # copy of the whole patch here would cost memory and buy no accuracy.
    # ``np.array`` (not ``asarray``) because the masking below writes into it.
    gal_array = np.array(exposure.image.array, dtype=np.float32)

    mask_array = prepare_mask(
        exposure.image.array,
        exposure.mask,
        exposure.variance.array,
        badMaskPlanes,
        original_mask_array=mask_array,
    )

    anacal.mask.mask_galaxy_image(gal_array, mask_array, False, star_cat)

    noise_variance = estimate_noise_variance(
        exposure.variance.array,
        exposure.mask,
        mask_array,
    )

    noise_array = prepare_noise_array(
        noise_array=noise_array,
        do_noise_bias_correction=do_noise_bias_correction,
        gal_shape=gal_array.shape,
        pixel_scale=pixel_scale,
        seed=seed,
        band=band,
        noise_variance=noise_variance,
        noise_corr=noise_corr,
        noiseId=noiseId,
        rotId=rotId,
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

    beginx = lsst_bbox.beginX
    beginy = lsst_bbox.beginY
    detection = prepare_detection(
        detection,
        pixel_scale=pixel_scale,
        beginx=beginx,
        beginy=beginy,
        blocks=blocks,
    )

    # Normalize the image onto the fixed AB zeropoint so the measured
    # moments/fluxes are independent of the coadd's native mag_zero. Both FPFS
    # paths (detection Task + fpfs.process_image) then receive this 31.4 image
    # with mag_zero=MAG_ZERO_AB. No-op (no allocation) for a native-31.4 coadd.
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
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Multi-band detection input
#
# Detection can use several bands at once.  Each band is prepared exactly as
# it would be on its own -- masked, noise-matched, normalised onto
# MAG_ZERO_AB -- and the per-band planes are then stacked into
# ``(nband, ny, nx)`` arrays.  anacal removes each band's own PSF before
# combining them, so nothing here needs to know about PSF matching; it only
# has to keep the bands aligned and hand over one noise variance per band.
# ---------------------------------------------------------------------------


def _stack_bands(per_band_data, bands, keys=("gal_array", "noise_array")):
    """Move each band's planes into one preallocated stack.

    ``per_band_data`` is a generator of single-band dicts.  The planes are
    copied into the stack and dropped from the dict as soon as they land, so
    at most one band's worth of extra memory is held on top of the stack --
    ``np.stack`` on a list of all the bands would hold two full copies.
    """
    SHARED = ("pixel_scale", "begin_x", "begin_y", "mag_zero")

    stacks: dict = {}
    psf_stack = None
    variances: list[float] = []
    first: dict | None = None
    # Kept separately: the planes are emptied out of ``first`` as they are
    # moved into the stack, so the dict cannot be used as the reference.
    ref_shape: tuple | None = None
    ref_shared: dict = {}

    for iband, data in enumerate(per_band_data):
        if first is None:
            first = data
            nband = len(bands)
            ref_shape = data["gal_array"].shape
            ref_shared = {name: data[name] for name in SHARED}
            ny, nx = ref_shape
            for key in keys:
                if data.get(key) is None:
                    stacks[key] = None
                else:
                    stacks[key] = np.empty(
                        (nband, ny, nx), dtype=data[key].dtype,
                    )
            npsf = data["psf_array"].shape[-1]
            psf_stack = np.empty((nband, npsf, npsf), dtype=np.float64)
        else:
            if data["gal_array"].shape != ref_shape:
                raise ValueError(
                    f"band '{bands[iband]}' has image shape "
                    f"{data['gal_array'].shape}, expected {ref_shape}"
                )
            for name in SHARED:
                if data[name] != ref_shared[name]:
                    raise ValueError(
                        f"band '{bands[iband]}' has {name}={data[name]}, "
                        f"but band '{bands[0]}' has {ref_shared[name]}"
                    )

        for key in keys:
            if (stacks[key] is None) != (data.get(key) is None):
                raise ValueError(
                    f"band '{bands[iband]}' disagrees with band "
                    f"'{bands[0]}' on whether '{key}' is present"
                )
            if stacks[key] is not None:
                stacks[key][iband] = data[key]
                data[key] = None
        psf_stack[iband] = data["psf_array"]
        variances.append(float(data["noise_variance"]))

    out = dict(first)
    out.update(stacks)
    out["psf_array"] = psf_stack
    out["noise_variance"] = variances
    if "base_column_name" in out:
        # The coadd belongs to no single band, so it gets no band prefix.
        out["base_column_name"] = None
    return out


def prepare_data_multiband(
    *,
    bands: Sequence[str],
    exposures: dict,
    seed: int,
    noiseId: int = 0,
    rotId: int = 0,
    npix: int = 64,
    noise_corrs: dict | None = None,
    do_noise_bias_correction: bool = True,
    badMaskPlanes: List[str] = badMaskDefault,
    skyMap=None,
    tract: int = 0,
    patch: int = 0,
    star_cat: NDArray | None = None,
    mask_array: NDArray | None = None,
    detection: astropy.table.Table | None = None,
    blocks: List | None = None,
    survey: str | None = None,
    **kwargs,
):
    """Stack several bands' exposures into one anacal detection input.

    Returns the same dict as :func:`prepare_data`, except that
    ``gal_array``, ``noise_array`` and ``psf_array`` gain a leading band
    axis and ``noise_variance`` is a list with one entry per band.
    """
    bands = list(bands)
    if len(bands) == 0:
        raise ValueError("prepare_data_multiband needs at least one band")
    if len(set(bands)) != len(bands):
        raise ValueError(f"duplicate entries in bands: {bands}")
    missing = [b for b in bands if b not in exposures]
    if missing:
        raise KeyError(f"no exposure for band(s) {missing}")

    exps = [exposures[b] for b in bands]
    union = _union_mask(
        [e.image.array for e in exps],
        [e.mask for e in exps],
        [e.variance.array for e in exps],
        badMaskPlanes,
        mask_array=mask_array,
    )
    noise_corrs = noise_corrs or {}

    def per_band():
        for band, exposure in zip(bands, exps):
            yield prepare_data(
                band=band,
                exposure=exposure,
                seed=seed,
                noiseId=noiseId,
                rotId=rotId,
                npix=npix,
                noise_corr=noise_corrs.get(band, None),
                do_noise_bias_correction=do_noise_bias_correction,
                badMaskPlanes=badMaskPlanes,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                star_cat=star_cat,
                mask_array=union,
                detection=detection,
                blocks=blocks,
                survey=survey,
            )

    return _stack_bands(per_band(), bands)
