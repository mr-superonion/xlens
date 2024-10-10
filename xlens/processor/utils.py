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

from typing import Any

import anacal
import lsst.geom as lsst_geom
import numpy as np
from numpy.typing import NDArray


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


def resize_array(array: NDArray[Any], target_shape: tuple[int, int] = (64, 64), truth_catalog=None):
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
    if input_width > target_width:
        start_w = (input_width - target_width) // 2
        array = array[:, start_w : start_w + target_width]
        if truth_catalog is not None:
            truth_catalog["image_x"] = truth_catalog["image_x"] - start_w

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
):
    """This function returns the average PSF model as numpy array
    Args:
    lsst_psf:  lsst PSF model
    lsst_bbox: lsst boundary box
    npix (int):  number of pixels for stamp
    dg (int): patch size
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
    # Calculate the central point
    x_array = np.arange(x_min, x_max, dg, dtype=int) + dg // 2
    y_array = np.arange(y_min, y_max, dg, dtype=int) + dg // 2
    nx, ny = len(x_array), len(y_array)
    out = np.zeros((npix, npix))
    ncount = 0.0
    for j in range(ny):
        yc = int(y_array[j])
        for i in range(nx):
            xc = int(x_array[i])
            this_psf = lsst_psf.computeImage(
                lsst_geom.Point2D(xc, yc)
            ).getArray()
            out = out + resize_array(this_psf, (npix, npix))
            ncount += 1
    out = out / ncount
    # cut out the boundary
    psf_rcut = npix // 2 - 4
    anacal.fpfs.base.truncate_square(out, psf_rcut)
    return out
