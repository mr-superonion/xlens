#
# LSST Data Management System
# Copyright 20082014 LSST Corpoalphan.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

from typing import Any, List

import anacal
import astropy
import lsst.geom as lsst_geom
import numpy as np
from astropy.stats import sigma_clip
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


def subpixel_shift(image: NDArray, shift_x: int, shift_y: int):
    """
    Perform a subpixel shift on a 2D image using the Fourier shift theorem.

    Args:
    image (NDArray): 2D numpy array representing the image to be shifted.
    shift_x (float): Subpixel shift in the x-direction.
    shift_y (float): Subpixel shift in the y-direction.

    Returns:
    shifted_image (NDArray): 2D numpy array representing the shifted image.
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
    """This is a util function to resize array to the target shape
    Args:
    array (NDArray): input array
    target_shape (tuple): output array's shape
    truth_catalog: truth catalog with image coordinates that need to be resized

    Returns:
    array (NDArray): output array with the target shape
    truth_catalog: resized truth catalog
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


class LsstPsf(anacal.psf.PyPsf):
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
        this_psf = self.psf.computeImage(
            lsst_geom.Point2D(x + self.x_min, y + self.y_min)
        ).getArray()
        this_psf = resize_array(this_psf, self.shape)
        return this_psf


def get_psf_array(
    *,
    lsst_psf,
    lsst_bbox,
    npix: int,
    dg: int = 250,
    lsst_mask: None | MaskX = None,
):
    """This function returns the average PSF model as numpy array
    Args:
    lsst_psf:  lsst PSF model
    lsst_bbox: lsst boundary box
    npix (int):  number of pixels for stamp
    dg (int): patch size
    lsst_mask (None | MaskX): mask object of LSST DM
    """
    width, height = lsst_bbox.getWidth(), lsst_bbox.getHeight()
    # Get the minimum corner
    min_corner = lsst_bbox.getMin()
    # Get the x_min and y_min
    x_min = min_corner.getX()
    y_min = min_corner.getY()

    max_corner = lsst_bbox.getMax()
    x_max = max_corner.getX()
    y_max = max_corner.getY()

    width = (width // dg) * dg - 1
    height = (height // dg) * dg - 1

    if lsst_mask is not None:
        if "INEXACT_PSF" in lsst_mask.getMaskPlaneDict().keys():
            bitv = lsst_mask.getPlaneBitMask("INEXACT_PSF")
            mask_array = bitv & lsst_mask.array
        else:
            mask_array = None
    else:
        mask_array = None

    # Calculate the central point
    x_array = np.arange(x_min, x_max, dg, dtype=int) + dg // 2
    y_array = np.arange(y_min, y_max, dg, dtype=int) + dg // 2
    nx, ny = len(x_array), len(y_array)
    out = np.zeros((npix, npix))
    ncount = 0.0
    for j in range(ny):
        yc = int(y_array[j])
        yim = yc - y_min
        for i in range(nx):
            xc = int(x_array[i])
            xim = xc - x_min
            if mask_array is None:
                this_psf = lsst_psf.computeImage(
                    lsst_geom.Point2D(xc, yc)
                ).getArray()
                out = out + resize_array(this_psf, (npix, npix))
                ncount += 1
            else:
                dx = 0
                dy = 0
                for _ in range(500):
                    if mask_array[yim+dy, xim+dx]==0:
                        this_psf = lsst_psf.computeImage(
                            lsst_geom.Point2D(xc+dx, yc+dy)
                        ).getArray()
                        out = out + resize_array(this_psf, (npix, npix))
                        ncount += 1
                        break
                    dx = np.random.randint(-10, 10)
                    dy = np.random.randint(-10, 10)

    out = out / ncount
    # cut out the boundary
    psf_rcut = npix // 2 - 4
    anacal.fpfs.base.truncate_square(out, psf_rcut)
    return out


def get_psf_object(
    *,
    lsst_psf,
    lsst_bbox,
    npix: int,
    dg: int = 250,
    lsst_mask: None | MaskX = None,
):
    """This function returns the average PSF model as numpy array
    Args:
    lsst_psf:  lsst PSF model
    lsst_bbox: lsst boundary box
    npix (int):  number of pixels for stamp
    dg (int): patch size
    lsst_mask (None | MaskX): mask object of LSST DM
    """
    # cut out the boundary
    psf_rcut = npix // 2 - 4

    width, height = lsst_bbox.getWidth(), lsst_bbox.getHeight()
    # Get the minimum corner
    min_corner = lsst_bbox.getMin()
    # Get the x_min and y_min
    x_min = min_corner.getX()
    y_min = min_corner.getY()

    max_corner = lsst_bbox.getMax()
    x_max = max_corner.getX()
    y_max = max_corner.getY()

    width = (width // dg) * dg - 1
    height = (height // dg) * dg - 1
    if lsst_mask is not None:
        if "INEXACT_PSF" in lsst_mask.getMaskPlaneDict().keys():
            bitv = lsst_mask.getPlaneBitMask("INEXACT_PSF")
            mask_array = bitv & lsst_mask.array
        else:
            mask_array = None
    else:
        mask_array = None

    # Calculate the central point
    x_array = np.arange(x_min, x_max, dg, dtype=int) + dg // 2
    y_array = np.arange(y_min, y_max, dg, dtype=int) + dg // 2
    nx, ny = len(x_array), len(y_array)
    out = np.zeros((ny, nx, npix, npix))
    for j in range(ny):
        yc = int(y_array[j])
        yim = yc - y_min
        for i in range(nx):
            xc = int(x_array[i])
            xim = xc - x_min
            this_psf = None
            if mask_array is None:
                this_psf = lsst_psf.computeImage(
                    lsst_geom.Point2D(xc, yc)
                ).getArray()
            else:
                dx = 0
                dy = 0
                for _ in range(500):
                    if mask_array[yim+dy, xim+dx]==0:
                        this_psf = lsst_psf.computeImage(
                            lsst_geom.Point2D(xc+dx, yc+dy)
                        ).getArray()
                        break
                    dx = np.random.randint(-10, 10)
                    dy = np.random.randint(-10, 10)
                if this_psf is None:
                    this_psf = lsst_psf.computeImage(
                        lsst_geom.Point2D(xc, yc)
                    ).getArray()
            this_psf = resize_array(this_psf, (npix, npix))
            anacal.fpfs.base.truncate_square(this_psf, psf_rcut)
            out[j, i] = this_psf

    return anacal.psf.GridPsf(x0=0, y0=0, dx=dg, dy=dg, model_array=out)

def prepare_data(
    *,
    exposure: ExposureF,
    seed: int,
    noiseId: int = 0,
    rotId: int = 0,
    npix: int = 32,
    noise_corr: NDArray | None = None,
    band: str | None = None,
    do_noise_bias_correction: bool = True,
    use_average_psf: bool = True,
    badMaskPlanes: List[str] = badMaskDefault,
    skyMap=None,
    tract: int = 0,
    patch: int = 0,
    star_cat: NDArray | None = None,
    detection: astropy.table.Table | None = None,
    **kwargs,
):
    """Prepares the data from LSST exposure
    Args:
    exposure (ExposureF):   LSST exposure
    seed (int):  random seed
    noise_corr (NDArray):  image noise correlation function (None)

    Returns:
        (dict)
    """
    from .random import get_noise_seed, image_noise_base

    pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
    noise_variance = np.nanmean(
        exposure.variance.array[
            (exposure.variance.array < 1e4) & (exposure.mask.array == 0)
        ],
    )
    if noise_variance < 1e-12:
        raise ValueError(
            "the estimated image noise variance should be positive."
        )
    noise_std = np.sqrt(noise_variance)
    mag_zero = (
        np.log10(exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()) / 0.4
    )
    wcs = exposure.getWcs()

    lsst_bbox = exposure.getBBox()
    lsst_psf = exposure.getPsf()

    if use_average_psf:
        psf = np.asarray(
            get_psf_array(
                lsst_psf=lsst_psf,
                lsst_bbox=lsst_bbox,
                npix=npix,
                dg=250,
                lsst_mask=exposure.mask,
            ),
            dtype=np.float64,
        )
    else:
        psf = get_psf_object(
            lsst_psf=lsst_psf,
            lsst_bbox=lsst_bbox,
            npix=npix,
            dg=100,
            lsst_mask=exposure.mask,
        )
    gal_array = np.asarray(
        exposure.image.array,
        dtype=np.float64,
    )

    bitValue = exposure.mask.getPlaneBitMask(badMaskPlanes)
    mask_array = (
        ((exposure.mask.array & bitValue) != 0)
        | (
            exposure.image.array
            < (
                -6.0
                * np.sqrt(
                    np.where(
                        exposure.variance.array < 0, 0, exposure.variance.array
                    )
                )
            )
        )
    ).astype(np.int16)

    if do_noise_bias_correction:
        noise_seed = (
            get_noise_seed(
                seed=seed,
                noiseId=noiseId,
                rotId=rotId,
            )
            + image_noise_base // 2
            # make sure the seed is different from
            # noise seed for simulation
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
    else:
        noise_array = None

    if band is None:
        base_column_name = None
    else:
        base_column_name = band + "_"
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
        "psf": psf,
        "mask_array": mask_array,
        "noise_array": noise_array,
        "base_column_name": base_column_name,
        "begin_x": lsst_bbox.beginX,
        "begin_y": lsst_bbox.beginY,
        "wcs": wcs,
        "skyMap": skyMap,
        "tractInfo": tractInfo,
        "patchInfo": patchInfo,
        "star_cat": star_cat,
        "detection": detection,
    }
