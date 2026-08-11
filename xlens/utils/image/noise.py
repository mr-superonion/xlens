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

"""Noise handling for AnaCal measurements: correlation rotation,
pure-noise generation, variance estimation, and the noise
plane used for noise-bias correction.

Split out of ``xlens.utils.image``, which re-exports every public name here
for backward compatibility.
"""


import anacal
import numpy as np
from numpy.typing import NDArray


def rotate_noise_corr(noise_corr):
    noise_max = np.amax(noise_corr)
    noise_corr = noise_corr / noise_max
    ny2, nx2 = noise_corr.shape
    if ny2 % 2 != 1 or nx2 % 2 != 1:
        raise ValueError(f"noise correlation must have odd dimensions; got {ny2}x{nx2}")
    if noise_corr[ny2 // 2, nx2 // 2] != 1:
        raise RuntimeError("noise correlation peak is not at the center pixel")
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
    survey: str | None = None,
):
    from ..random import get_noise_seed

    noise_std = np.sqrt(noise_variance)
    noise_seed = get_noise_seed(
        galaxy_seed=seed,
        noiseId=noiseId,
        rotId=rotId,
        band=band,
        survey=survey,
        is_sim=False,
    )
    # Drawn in float64 and stored in float32, to match the science plane: the
    # rounding is 7 significant digits down on a number that is itself random,
    # and AnaCal widens back to double when it reads the pixels.
    if noise_corr is None:
        noise_array = (
            np.random.RandomState(noise_seed)
            .normal(
                scale=noise_std,
                size=(ny, nx),
            )
            .astype(np.float32)
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
        ).astype(np.float32)
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
        mm &= mask_array == 0
    if np.sum(mm) < 10:
        raise ValueError("Not enough valid pixels for noise estimation")
    noise_variance = float(np.nanmedian(variance_array[mm]))
    if noise_variance < 1e-10 or np.isnan(noise_variance):
        raise ValueError("Estimated noise variance must be positive")
    return noise_variance


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
    survey: str | None = None,
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
            survey=survey,
        )
    else:
        # Rotate noise image by 90 degrees CCW to remove anisotropy
        noise_array = np.rot90(noise_array, k=1)
    noise_array = np.asarray(noise_array, dtype=np.float32)

    anacal.mask.mask_galaxy_image(
        noise_array,
        mask_array,
        star_cat,
    )
    return noise_array
