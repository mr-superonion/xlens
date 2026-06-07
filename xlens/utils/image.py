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


from typing import Any, List, Sequence

import anacal
import astropy
import lsst.geom as lsst_geom
import lsst.pex.exceptions as pexExcept
import numpy as np
from numpy.lib import recfunctions as rfn
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
):
    """Resize an image-like array to a square target shape.

    The function first crops the array symmetrically if it is larger than the
    requested output size and then applies zero-padding when the array is too
    small.

    Parameters
    ----------
    array
        Input array to resize.  The array is assumed to be two-dimensional.
    target_shape
        Tuple of ``(height, width)`` describing the requested output shape.

    Returns
    -------
    numpy.ndarray
        The resized array.
    """
    target_height, target_width = target_shape
    input_height, input_width = array.shape

    # Crop if larger
    if input_height > target_height:
        start_h = (input_height - target_height) // 2
        array = array[start_h : start_h + target_height, :]
    if input_width > target_width:
        start_w = (input_width - target_width) // 2
        array = array[:, start_w : start_w + target_width]

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
    lsst_mask=None,
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

    # Ensure grid stays within the bbox and aligned with step size.
    # For patches small enough that the strided grid is empty, fall
    # back to a single center sample so we still get one PSF estimate.
    width = (x_max - x_min) // dg * dg
    height = (y_max - y_min) // dg * dg

    x_array = np.arange(x_min + 20, x_min + width - 20, dg, dtype=int)
    y_array = np.arange(y_min + 20, y_min + height - 20, dg, dtype=int)
    if len(x_array) == 0:
        x_array = np.array([(x_min + x_max) // 2], dtype=int)
    if len(y_array) == 0:
        y_array = np.array([(y_min + y_max) // 2], dtype=int)

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

    if ncount < 1:
        raise ValueError("Could not find any valid PSF sample.")

    out /= ncount
    psf_rcut = npix // 2 - 2
    truncate_square(out, psf_rcut)
    return out


def get_blocks(
    *, lsst_psf, lsst_bbox, pixel_scale, npix, psf_array
):
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
            this_psf = lsst_psf.computeImage(
                lsst_geom.Point2D(x_min + x0, y_min + y0)
            ).getArray()
            bb.psf_array = resize_array(this_psf, (npix, npix))
        except Exception:
            continue
        new_blocks.append(bb)
    return new_blocks


def stack_psfs_cells(
    *, cell_coadd, npix
):
    psf_array = np.zeros((npix, npix))
    npsf = 0.0
    for cell in cell_coadd.cells.values():
        p0 = None
        psf_image = getattr(cell, "psf_image", None)
        if psf_image is not None:
            p0 = getattr(psf_image, "array", None)
        if (p0 is not None) and np.isfinite(p0).all():
            psf_array = psf_array + resize_array(
                p0,
                (npix, npix),
            )
            npsf += 1
    psf_array = psf_array / npsf
    return psf_array


def combine_sim_exposures(
    exposures: Sequence,
    noises: Sequence[NDArray],
):
    """Combine simulated exposures using inverse-variance weights.
    """

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
            raise ValueError(
                "Variance plane must contain at least one finite value"
            )
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


def rotate_noise_corr(noise_corr):
    noise_max = np.amax(noise_corr)
    noise_corr = noise_corr / noise_max
    ny2, nx2 = noise_corr.shape
    if ny2 % 2 != 1 or nx2 % 2 != 1:
        raise ValueError(
            f"noise correlation must have odd dimensions; got {ny2}x{nx2}"
        )
    if noise_corr[ny2 // 2, nx2 // 2] != 1:
        raise RuntimeError(
            "noise correlation peak is not at the center pixel"
        )
    return np.rot90(m=noise_corr, k=-1)


def generate_pure_noise(
    *,
    ny: int,
    nx: int,
    pixel_scale: float,
    seed: int,
    band: str | None,
    noise_variance: float,
    noise_corr=None,
    noiseId: int = 0,
    rotId: int = 0,
):
    from .random import get_noise_seed
    noise_std = np.sqrt(noise_variance)
    noise_seed = get_noise_seed(
        galaxy_seed=seed,
        noiseId=noiseId,
        rotId=rotId,
        band=band,
        is_sim=False,
    )
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
        noise_corr = rotate_noise_corr(noise_corr)
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
    return noise_array


def estimate_noise_variance(
    variance_array: NDArray,
    mask,
    mask_array: NDArray | None = None,
) -> float:
    """Estimate noise variance from the variance plane.

    Parameters
    ----------
    variance_array : NDArray
        Variance plane of the image.
    mask : lsst.afw.image.Mask
        Raw mask object (e.g., ``exposure.mask``). Used both for its
        pixel array and to look up the DETECTED / DETECTED_NEGATIVE bit
        positions via the exposure's own mask-plane dict — so we do not
        hard-code bit indices that may differ across cameras / stack
        versions.
    mask_array : NDArray or None
        Processed mask (bad pixels, bright stars, etc.).
        If None, only ``mask`` is used for pixel selection.

    Returns
    -------
    float
        Median noise variance over valid pixels.
    """
    detect_bits = mask.getPlaneBitMask(["DETECTED", "DETECTED_NEGATIVE"])
    mask_raw = mask.array
    mm = (variance_array < 1e5) & ((mask_raw & detect_bits) == 0)
    if mask_array is not None:
        mm &= (mask_array == 0)
    if np.sum(mm) < 10:
        raise ValueError("Not enough valid pixels for noise estimation")
    noise_variance = float(np.nanmedian(variance_array[mm]))
    if noise_variance < 1e-10 or np.isnan(noise_variance):
        raise ValueError("Estimated noise variance must be positive")
    return noise_variance


def prepare_psf_array(
    exposure,
    npix: int,
) -> NDArray:
    """Compute average PSF array from an LSST exposure."""
    psf_array = np.asarray(
        get_psf_array(
            lsst_psf=exposure.getPsf(),
            lsst_bbox=exposure.getBBox(),
            npix=npix,
            dg=250,
            lsst_mask=exposure.mask,
        ),
        dtype=np.float64,
    )
    return psf_array


def prepare_psf_array_cell(
    cell,
    npix: int,
) -> NDArray:
    """Compute PSF array from a SingleCellCoadd's psf_image."""
    psf_img = cell.psf_image.array
    psf_array = np.asarray(
        resize_array(psf_img, (npix, npix)), dtype=np.float64,
    )
    psf_array /= np.sum(psf_array)
    psf_rcut = npix // 2 - 2
    truncate_square(psf_array, psf_rcut)
    return psf_array


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
        | (
            image_array
            < (
                -6.0
                * np.sqrt(
                    np.where(variance_array < 0, 0, variance_array)
                )
            )
        )
    ).astype(np.int16)
    if original_mask_array is None:
        return new_mask
    return (new_mask | original_mask_array.astype(np.int16)).astype(np.int16)


def prepare_noise_array(
    *,
    noise_array: NDArray | None,
    do_noise_bias_correction: bool,
    gal_shape: tuple[int, int],
    pixel_scale: float,
    seed: int,
    band: str | None,
    noise_variance: float,
    noise_corr: NDArray | None = None,
    noiseId: int = 0,
    rotId: int = 0,
    mask_array: NDArray | None = None,
    star_cat: NDArray | None = None,
) -> NDArray | None:
    """Prepare the noise array for noise bias correction.

    If ``noise_array`` is provided, it is rotated 90 degrees CCW
    to decorrelate noise from shear. If ``None``, a pure noise
    realisation is generated. The result is masked by ``mask_array``.

    Parameters
    ----------
    noise_array : NDArray or None
        Pre-existing noise array (e.g., from cell coadd noise
        realisations). Rotated 90 degrees if provided.
    do_noise_bias_correction : bool
        Whether to include noise bias correction.
    gal_shape : tuple of int
        ``(ny, nx)`` shape of the galaxy image.
    pixel_scale : float
        Pixel scale in arcseconds.
    seed : int
        Random seed for noise generation.
    band : str or None
        Photometric band label.
    noise_variance : float
        Estimated noise variance.
    noise_corr : NDArray or None
        Noise correlation function.
    noiseId, rotId : int
        Noise and rotation identifiers.
    mask_array : NDArray or None
        Mask to apply to the noise array.
    star_cat : NDArray or None
        Star catalogue for masking.

    Returns
    -------
    NDArray or None
        Prepared noise array, or None if correction is disabled.
    """
    if not do_noise_bias_correction:
        return None

    if noise_array is None:
        ny, nx = gal_shape
        noise_array = generate_pure_noise(
            ny=ny,
            nx=nx,
            pixel_scale=pixel_scale,
            seed=seed,
            band=band,
            noise_variance=noise_variance,
            noise_corr=noise_corr,
            noiseId=noiseId,
            rotId=rotId,
        )
    else:
        # Rotate noise image by 90 degrees CCW to remove anisotropy
        noise_array = np.rot90(noise_array, k=1)

    anacal.mask.mask_galaxy_image(
        noise_array,
        mask_array,
        False,
        star_cat,
    )
    return noise_array


def prepare_detection(
    detection: astropy.table.Table | NDArray | None,
    *,
    pixel_scale: float,
    beginx: int,
    beginy: int,
    blocks: List | None = None,
) -> NDArray | None:
    """Prepare a detection catalog for forced measurement.

    Copies the catalog, selects anacal columns, and assigns block IDs
    based on detection positions.

    Parameters
    ----------
    detection : astropy.table.Table, NDArray, or None
        Input detection catalog.
    pixel_scale : float
        Pixel scale in arcseconds.
    beginx, beginy : int
        Image origin in pixel coordinates.
    blocks : list or None
        Anacal block list for block ID assignment.

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
    detection = rfn.repack_fields(
        detection[list(anacal.table.column_names())]
    )
    if blocks is not None:
        for bb in blocks:
            mm = (
                (detection["x2_det"] / pixel_scale - beginy >= bb.ymin_in)
                & (detection["x2_det"] / pixel_scale - beginy < bb.ymax_in)
                & (detection["x1_det"] / pixel_scale - beginx >= bb.xmin_in)
                & (detection["x1_det"] / pixel_scale - beginx < bb.xmax_in)
            )
            detection["block_id"][mm] = bb.index
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
    **kwargs,
):
    """Collect metadata and auxiliary arrays from an LSST ExposureF."""
    pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
    mag_zero = (
        np.log10(exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()) / 0.4
    )
    wcs = exposure.getWcs()
    lsst_bbox = exposure.getBBox()

    if psf_array is None:
        psf_array = prepare_psf_array(exposure, npix)

    gal_array = np.asarray(exposure.image.array, dtype=np.float64)

    mask_array = prepare_mask(
        exposure.image.array, exposure.mask,
        exposure.variance.array, badMaskPlanes,
        original_mask_array=mask_array,
    )

    anacal.mask.mask_galaxy_image(gal_array, mask_array, False, star_cat)

    noise_variance = estimate_noise_variance(
        exposure.variance.array, exposure.mask, mask_array,
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

    return {
        "pixel_scale": pixel_scale,
        "mag_zero": mag_zero,
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


def prepare_data_one_cell(
    *,
    cell,
    band: str | None,
    seed: int,
    mag_zero: float,
    npix: int = 64,
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
    **kwargs,
):
    """Collect metadata and auxiliary arrays from a SingleCellCoadd."""
    outer = cell.outer
    wcs = cell.wcs
    pixel_scale = float(wcs.getPixelScale().asArcseconds())

    bbox = outer.bbox
    gal_array = np.asarray(outer.image.array, dtype=np.float64)

    if psf_array is None:
        psf_array = prepare_psf_array_cell(cell, npix)

    mask_array = prepare_mask(
        outer.image.array, outer.mask,
        outer.variance.array, badMaskPlanes,
        original_mask_array=mask_array,
    )

    anacal.mask.mask_galaxy_image(gal_array, mask_array, False, star_cat)

    noise_variance = estimate_noise_variance(
        outer.variance.array, outer.mask, mask_array,
    )

    # Extract noise from cell if available and not provided
    if do_noise_bias_correction and noise_array is None:
        noise_reals = outer.noise_realizations
        if len(noise_reals) > 0:
            noise_array = np.asarray(
                noise_reals[0].array, dtype=np.float64,
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
        blocks=blocks,
    )

    return {
        "pixel_scale": pixel_scale,
        "mag_zero": mag_zero,
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
        "psf_object": None,
        "lsst_psf": None,
        "base_column_name": (band + "_") if band is not None else None,
    }
