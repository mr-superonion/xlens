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

"""PSF models and PSF-stamp utilities: array helpers, the
``anacal.psf.BasePsf`` wrappers, and per-patch / per-cell PSF
stamp preparation (including block construction inputs).

Split out of ``xlens.utils.image``, which re-exports every public name here
for backward compatibility.
"""

from typing import Any

import anacal
import lsst.geom as lsst_geom
import numpy as np
from numpy.typing import NDArray


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
        this_psf = self.psf.computeImage(lsst_geom.Point2D(x + self.x_min, y + self.y_min)).getArray()
        this_psf = resize_array(this_psf, self.shape)
        return this_psf


class GridPsf(anacal.psf.BasePsf):
    """Spatially varying PSF defined on a regular grid of postage stamps.

    A survey-agnostic ``anacal`` PSF adapter.  The spatial variation is
    captured by a coarse grid of pre-computed PSF stamps; :meth:`draw`
    returns the stamp of the cell containing the requested pixel
    (nearest-cell lookup, no interpolation).  It complements
    :class:`LsstPsf` for cases where the PSF is only available as
    sampled stamps rather than as a callable model -- for example a
    Euclid MER catalogue PSF, or an LSST ``CoaddPsf`` pre-sampled on a
    grid so that a single object works across surveys.

    Parameters
    ----------
    model : numpy.ndarray
        Grid of PSF stamps with shape ``(ny, nx, npix, npix)``.  Cell
        ``(i, j)`` holds the (unit-sum) PSF for the pixel region
        ``x in [j*dx, (j+1)*dx)`` and ``y in [i*dy, (i+1)*dy)`` in the image's
        own pixel frame.
    dx, dy : int
        Cell size in pixels along the x and y axes.
    """

    def __init__(self, model: NDArray, dx: int, dy: int):
        super().__init__()
        self.model = np.ascontiguousarray(model)
        if self.model.ndim != 4:
            raise ValueError("model must have shape (ny, nx, npix, npix)")
        self.dx = int(dx)
        self.dy = int(dy)

    def draw(self, x: float, y: float) -> NDArray:
        """Return the PSF stamp of the grid cell containing pixel ``(x, y)``.

        Positions outside the grid are clamped to the nearest edge cell.
        """
        ny, nx = self.model.shape[:2]
        j = int(np.clip(x // self.dx, 0, nx - 1))
        i = int(np.clip(y // self.dy, 0, ny - 1))
        return np.ascontiguousarray(self.model[i, j])

    @property
    def average(self) -> NDArray:
        """Grid-averaged, unit-sum PSF stamp (the exposure-average PSF)."""
        avg = np.ascontiguousarray(self.model.mean(axis=(0, 1)))
        return avg / avg.sum()


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
                psf_img = lsst_psf.computeImage(lsst_geom.Point2D(xc, yc)).getArray()
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


def stack_psfs_cells(*, cell_coadd, npix):
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
        resize_array(psf_img, (npix, npix)),
        dtype=np.float64,
    )
    psf_array /= np.sum(psf_array)
    psf_rcut = npix // 2 - 2
    truncate_square(psf_array, psf_rcut)
    return psf_array
