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

"""Image utilities for working with LSST exposures and PSF models.

This module collects helper routines that are repeatedly used across
``xlens`` when generating or post-processing simulated images.  The
implementations originate from the LSST Science Pipelines, and the
docstrings have been expanded here to clarify how they interact with the
rest of ``xlens``.
"""


from typing import Any, List

import anacal
import astropy
import lsst.geom as lsst_geom
import numpy as np
from lsst.afw.image import ExposureF, MaskX
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


def subpixel_shift(image: NDArray, shift_x: float, shift_y: float) -> NDArray:
    """Shift an image by arbitrary subpixel offsets using Fourier methods.

    Parameters
    ----------
    image
        Two-dimensional array containing the image that should be shifted.
    shift_x
        Desired shift in the x-direction, expressed in pixel units.  The
        value can be any real number; positive values move the image towards
        larger x.
    shift_y
        Desired shift in the y-direction, expressed in pixel units.  Positive
        values move the image towards larger y.

    Returns
    -------
    numpy.ndarray
        The shifted image.  The output has the same shape as the input and is
        guaranteed to be real-valued.
    """
    # Get the image size
    ny, nx = image.shape

    # Create a grid of coordinates in the frequency domain
    x = np.fft.fftfreq(nx)
    y = np.fft.fftfreq(ny)
    X, Y = np.meshgrid(x, y)

    # Fourier transform of the image
    f_image = np.fft.fft2(image)

    # Create the shift phase factor
    phase_shift = np.exp(-2j * np.pi * (shift_x * X + shift_y * Y))

    # Apply the shift in the frequency domain
    f_image_shifted = f_image * phase_shift

    # Inverse Fourier transform to get the shifted image
    shifted_image = np.fft.ifft2(f_image_shifted)

    # Take the real part of the shifted image
    shifted_image = np.real(shifted_image)

    return shifted_image


def resize_array(
    array: NDArray[Any],
    target_shape: tuple[int, int] = (64, 64),
    truth_catalog=None,
):
    """Resize an image-like array to a square target shape.

    The function first crops the array symmetrically if it is larger than the
    requested output size and then applies zero-padding when the array is too
    small.  When a truth catalog is provided, its pixel coordinates are
    updated so they remain consistent with the resized image.

    Parameters
    ----------
    array
        Input array to resize.  The array is assumed to be two-dimensional.
    target_shape
        Tuple of ``(height, width)`` describing the requested output shape.
    truth_catalog
        Optional truth catalog whose ``image_*`` and ``prelensed_image_*``
        columns are updated in place so that they refer to the resized image.

    Returns
    -------
    numpy.ndarray or tuple
        The resized array.  If ``truth_catalog`` was provided, the function
        returns a tuple ``(array, truth_catalog)`` with the mutated catalog as
        the second element.
    """
    target_height, target_width = target_shape
    input_height, input_width = array.shape

    # Crop if larger
    if input_height > target_height:
        start_h = (input_height - target_height) // 2
        array = array[start_h : start_h + target_height, :]
        if truth_catalog is not None:
            truth_catalog["image_y"] = truth_catalog["image_y"] - start_h
            truth_catalog["prelensed_image_y"] = (
                truth_catalog["prelensed_image_y"] - start_h
            )
    if input_width > target_width:
        start_w = (input_width - target_width) // 2
        array = array[:, start_w : start_w + target_width]
        if truth_catalog is not None:
            truth_catalog["image_x"] = truth_catalog["image_x"] - start_w
            truth_catalog["prelensed_image_x"] = (
                truth_catalog["prelensed_image_x"] - start_w
            )

    # Pad with zeros if smaller
    if input_height < target_height:
        pad_height = target_height - input_height
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        array = np.pad(
            array,
            ((pad_bottom, pad_top), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

    if input_width < target_width:
        pad_width = target_width - input_width
        pad_right = pad_width // 2
        pad_left = pad_width - pad_right
        array = np.pad(
            array,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
        )
    if truth_catalog is not None:
        return array, truth_catalog
    else:
        return array


class LsstPsf(anacal.psf.BasePsf):
    """Adapter that exposes an LSST PSF model with an ``anacal`` interface."""

    def __init__(self, psf, npix, lsst_bbox=None):
        super().__init__()
        self.psf = psf
        self.shape = (npix, npix)

        if lsst_bbox is None:
            self.x_min = 0.0
            self.y_min = 0.0
        else:
            min_corner = lsst_bbox.getMin()
            # Get the x_min and y_min
            self.x_min = min_corner.getX()
            self.y_min = min_corner.getY()

    def draw(self, x, y):
        """Evaluate the PSF image centered on the requested pixel position."""
        this_psf = self.psf.computeImage(
            lsst_geom.Point2D(x + self.x_min, y + self.y_min)
        ).getArray()
        this_psf = resize_array(this_psf, self.shape)
        return this_psf


def truncate_square(arr: NDArray, rcut: int) -> None:
    """Zero out pixels outside a centred square support region.

    The function is primarily used when constructing PSF postage stamps.  It
    enforces a compact support by setting all pixels farther than ``rcut``
    from the stamp centre to zero while leaving the inner region untouched.

    Parameters
    ----------
    arr : numpy.ndarray
        Square, two-dimensional array to modify in place.
    rcut : int
        Half-width of the square region that should be kept.  The resulting
        mask spans ``2 * rcut + 1`` pixels in both directions.

    Raises
    ------
    ValueError
        If ``arr`` is not a square 2-D array or if ``rcut`` is too large for
        the provided array size.
    """
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input array must be a 2D square array")
    npix = arr.shape[0]
    npix2 = npix // 2
    assert rcut < npix2, "truncation radius too large."
    if rcut < npix2 - 1:
        arr[: npix2 - rcut, :] = 0
        arr[npix2 + rcut + 1 :, :] = 0
        arr[:, : npix2 - rcut] = 0
        arr[:, npix2 + rcut + 1 :] = 0
    return


def get_psf_array(
    *,
    lsst_psf,
    lsst_bbox,
    npix: int,
    dg: int = 250,
    lsst_mask: None | MaskX = None,
):
    """Compute an average PSF image over a regular grid.

    The function samples the provided LSST PSF model at a grid of points
    across the bounding box and averages the resulting images.  Pixels that
    are flagged as ``INEXACT_PSF`` in the optional mask are excluded from the
    average, mimicking the behaviour in the LSST pipelines.

    Parameters
    ----------
    lsst_psf : lsst.meas.algorithms.Psf
        LSST PSF model.
    lsst_bbox : lsst.geom.Box2I
        Bounding box defining the region to evaluate the PSF.
    npix : int
        Target shape (npix, npix) to which each PSF will be resized.
    dg : int, optional
        Grid spacing in pixels (default is 250).
    lsst_mask : MaskX or None, optional
        LSST mask image. If provided, pixels with INEXACT_PSF will be skipped.

    Returns
    -------
    out : numpy.ndarray
        Averaged PSF as a 2D array of shape ``(npix, npix)``.
    """
    x_min, y_min = lsst_bbox.getMin().getX(), lsst_bbox.getMin().getY()
    x_max, y_max = lsst_bbox.getMax().getX(), lsst_bbox.getMax().getY()

    # Ensure grid stays within the bbox and aligned with step size
    width = (x_max - x_min) // dg * dg
    height = (y_max - y_min) // dg * dg

    x_array = np.arange(x_min + 20, x_min + width - 20, dg, dtype=int)
    y_array = np.arange(y_min + 20, y_min + height - 20, dg, dtype=int)

    # Build INEXACT_PSF mask if needed
    if lsst_mask is not None and "INEXACT_PSF" in lsst_mask.getMaskPlaneDict():
        bitmask = lsst_mask.getPlaneBitMask("INEXACT_PSF")
        mask_array = (bitmask & lsst_mask.array) > 0
    else:
        mask_array = None

    out = np.zeros(shape=(npix, npix), dtype=np.float32)
    ncount = 0

    for yc in y_array:
        for xc in x_array:
            yim, xim = yc - y_min, xc - x_min
            if mask_array is not None and mask_array[yim, xim]:
                continue
            try:
                psf_img = lsst_psf.computeImage(
                    lsst_geom.Point2D(xc, yc)
                ).getArray()
                out += resize_array(psf_img, (npix, npix))
                ncount += 1
            except Exception:
                continue

    if ncount < 2:
        raise ValueError("Could not find enough valid PSF samples to average.")

    out /= ncount
    psf_rcut = npix // 2 - 2
    truncate_square(out, psf_rcut)
    return out


def get_blocks(
    *, lsst_psf, lsst_bbox, lsst_mask, pixel_scale, npix, psf_array
):
    min_corner = lsst_bbox.getMin()
    x_min, y_min = min_corner.getX(), min_corner.getY()
    width, height = lsst_bbox.getWidth(), lsst_bbox.getHeight()

    # Build mask array: True = masked
    if lsst_mask is not None and "INEXACT_PSF" in lsst_mask.getMaskPlaneDict():
        bitv = lsst_mask.getPlaneBitMask("INEXACT_PSF")
        mask_array = (bitv & lsst_mask.array) > 0
    else:
        mask_array = np.zeros((height, width), dtype=bool)

    # Create blocks
    blocks = anacal.geometry.get_block_list(
        img_ny=height,
        img_nx=width,
        block_nx=250,
        block_ny=250,
        block_overlap=80,
        scale=pixel_scale,
    )

    for bb in blocks:
        # Center of the block
        x0 = int(np.clip(bb.xcen, 0, width - 1))
        y0 = int(np.clip(bb.ycen, 0, height - 1))
        # Define 21x21 local box
        x_start = max(x0 - 10, 0)
        x_end   = min(x0 + 11, width)
        y_start = max(y0 - 10, 0)
        y_end   = min(y0 + 11, height)
        # Get unmasked local pixels
        local_mask = mask_array[y_start:y_end, x_start:x_end]
        local_yx = np.argwhere(~local_mask)  # shape (N, 2)
        if local_yx.shape[0] == 0:
            bb.psf_array = psf_array
            continue
        # Compute squared distances to block center
        local_coords = local_yx + np.array([y_start, x_start])
        dy = local_coords[:, 0] - y0
        dx = local_coords[:, 1] - x0
        dist2 = dx**2 + dy**2
        # Sort and try the 5 closest
        sorted_idx = np.argsort(dist2)[:5]
        found = False

        for idx in sorted_idx:
            yy, xx = local_coords[idx]
            try:
                this_psf = lsst_psf.computeImage(
                    lsst_geom.Point2D(x_min + xx, y_min + yy)
                ).getArray()
                bb.psf_array = resize_array(this_psf, (npix, npix))
                found = True
                break
            except Exception:
                continue
        if not found:
            bb.psf_array = psf_array
    return blocks


def prepare_data(
    *,
    band: str,
    exposure: ExposureF,
    seed: int,
    noiseId: int = 0,
    rotId: int = 0,
    npix: int = 32,
    noise_corr: NDArray | None = None,
    do_noise_bias_correction: bool = True,
    badMaskPlanes: List[str] = badMaskDefault,
    skyMap=None,
    tract: int = 0,
    patch: int = 0,
    star_cat: NDArray | None = None,
    mask_array: NDArray | None = None,
    detection: astropy.table.Table | None = None,
    **kwargs,
):
    """Collect metadata and auxiliary arrays for shear measurement tasks.

    The routine orchestrates several helper utilities in this module to build
    a dictionary consumed by the analysis pipeline.  It extracts PSF postage
    stamps, prepares the galaxy image data, and computes deterministic random
    seeds used when adding synthetic noise.

    Parameters
    ----------
    band : str
        Photometric band label used to tag the output dictionary.
    exposure : lsst.afw.image.ExposureF
        LSST exposure containing the science image and its associated PSF and
        mask information.
    seed : int
        Base seed that, together with ``noiseId`` and ``rotId``, controls the
        stochastic components of the processing.
    noiseId : int, optional
        Identifier for the noise realisation.  Defaults to ``0``.
    rotId : int, optional
        Identifier for the rotation realisation.  Defaults to ``0``.
    npix : int, optional
        Target size of the PSF postage stamp in pixels.  Defaults to ``32``.
    noise_corr : numpy.ndarray, optional
        Noise correlation function sampled on the same grid as the PSF stamp.
    do_noise_bias_correction : bool, optional
        If ``True`` (default) include the per-block noise-bias correction
        arrays in the output payload.
    badMaskPlanes : list of str, optional
        Collection of mask plane names that should be treated as invalid.
    skyMap : optional
        Sky-map descriptor propagated to the output dictionary unchanged.
    tract, patch : int, optional
        Identifiers for the tract and patch associated with ``exposure``.
    star_cat : numpy.ndarray, optional
        Catalogue of reference stars used for PSF modelling.
    mask_array : numpy.ndarray, optional
        Pre-computed boolean mask array.  If ``None`` the mask is built from
        ``exposure`` directly.
    detection : astropy.table.Table, optional
        Detection catalogue that provides initial estimates for source
        properties.
    **kwargs
        Additional keyword arguments propagated to downstream consumers.

    Returns
    -------
    dict
        A dictionary containing harmonised image data, PSF information, and
        metadata ready for the ``anacal`` measurement pipeline.
    """
    from .random import get_noise_seed

    pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
    mag_zero = (
        np.log10(exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()) / 0.4
    )
    wcs = exposure.getWcs()

    lsst_bbox = exposure.getBBox()
    lsst_psf = exposure.getPsf()
    psf_array = np.asarray(
        get_psf_array(
            lsst_psf=lsst_psf,
            lsst_bbox=lsst_bbox,
            npix=npix,
            dg=250,
            lsst_mask=exposure.mask,
        ),
        dtype=np.float64,
    )
    gal_array = np.asarray(
        exposure.image.array,
        dtype=np.float64,
    )

    if mask_array is None:
        bitv = exposure.mask.getPlaneBitMask(badMaskPlanes)
        mask_array = (
            ((exposure.mask.array & bitv) != 0)
            | (
                exposure.image.array
                < (
                    -6.0
                    * np.sqrt(
                        np.where(
                            exposure.variance.array < 0,
                            0, exposure.variance.array,
                        )
                    )
                )
            )
        ).astype(np.int16)
    # Set the value inside star mask to zero
    anacal.mask.mask_galaxy_image(
        gal_array,
        mask_array,
        False,  # extend mask
        star_cat,
    )

    mm = (
        (exposure.variance.array < 1e9) &
        (exposure.mask.array == 0) &
        (mask_array == 0)
    )
    if np.sum(mm) < 10:
        raise ValueError(
            "Do not have enough valid pixels"
        )
    noise_variance = np.nanmean(
        exposure.variance.array[mm],
    )
    del mm
    if (noise_variance < 1e-10) | (np.isnan(noise_variance)):
        raise ValueError(
            "the estimated image noise variance should be positive."
        )
    noise_std = np.sqrt(noise_variance)

    if do_noise_bias_correction:
        noise_seed = get_noise_seed(
            galaxy_seed=seed,
            noiseId=noiseId,
            rotId=rotId,
            band=band,
            is_sim=False,
        )
        ny, nx = gal_array.shape
        if noise_corr is None:
            noise_array = (
                np.random.RandomState(noise_seed)
                .normal(
                    scale=noise_std,
                    size=(ny, nx),
                )
                .astype(np.float64)
            )
        else:
            noise_corr = np.rot90(m=noise_corr, k=-1)
            noise_array = (
                anacal.noise.simulate_noise(
                    seed=noise_seed,
                    correlation=noise_corr,
                    nx=nx,
                    ny=ny,
                    scale=pixel_scale,
                )
                * noise_std
            )
        # Also do it for pure noise image
        anacal.mask.mask_galaxy_image(
            noise_array,
            mask_array,
            False,  # extend mask
            star_cat,
        )
    else:
        noise_array = None

    if skyMap is not None:
        tractInfo = skyMap[tract]
        patchInfo = tractInfo[patch]
    else:
        tractInfo = None
        patchInfo = None
    if detection is not None:
        if isinstance(detection, astropy.table.Table):
            detection = detection.copy().as_array()
        elif isinstance(detection, np.ndarray):
            detection = detection.copy()

    return {
        "pixel_scale": pixel_scale,
        "mag_zero": mag_zero,
        "noise_variance": noise_variance,
        "gal_array": gal_array,
        "psf_array": psf_array,
        "mask_array": mask_array,
        "noise_array": noise_array,
        "begin_x": lsst_bbox.beginX,
        "begin_y": lsst_bbox.beginY,
        "wcs": wcs,
        "skyMap": skyMap,
        "tractInfo": tractInfo,
        "patchInfo": patchInfo,
        "detection": detection,
    }
