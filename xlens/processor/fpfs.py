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

"""FPFS shape measurement task wrapping :mod:`anacal.fpfs`.

Provides :class:`FpfsMeasurementTask`, a Rubin-style ``Task`` that measures
FPFS shapelet moments from coadd exposures and returns structured catalogs
ready for shear estimation.
"""

from typing import Any

import anacal
import numpy as np
from lsst.pex.config import Config, Field, FieldValidationError, ListField
from lsst.pipe.base import Task
from numpy.typing import NDArray

from .. import utils
from ..utils.constants import FPFS_C0
from ..utils.image import badMaskDefault
from ..utils.random import num_rot


class FpfsMeasurementConfig(Config):
    """Configuration for :class:`FpfsMeasurementTask`."""

    npix = Field[int](
        doc="number of pixels in stamp [pixel]",
        default=64,
    )
    bound = Field[int](
        doc="Sources to be removed if too close to boundary [pixel]",
        default=35,
    )
    sigma_shapelets = Field[float](
        doc="Shapelet's Gaussian kernel size for detection [arcsec]",
        default=0.52,
    )
    sigma_shapelets1 = Field[float](
        doc="Shapelet's Gaussian kernel size for measurement [arcsec]",
        optional=True,
        default=-1,
    )
    sigma_shapelets2 = Field[float](
        doc="Shapelet's Gaussian kernel for the second measurement [arcsec]",
        optional=True,
        default=-1,
    )
    snr_min = Field[float](
        doc="Minimum signal-to-noise ratio for the flux selection.",
        optional=True,
        default=12.0,
    )
    r2_min = Field[float](
        doc=(
            "Minimum of the size ratio (m00 + m20) / m00. Matches the same "
            "cut in the ngmix/Task path (``fpfs_m2 - 0.05 * fpfs_m0``) and "
            "``trace_min`` in ``xlens.catalog.base``."
        ),
        optional=True,
        default=0.05,
    )
    c0 = Field[float](
        doc=(
            "C0 normalisation in ``e = m22c / (m00 + c0)``, on the fixed AB "
            "nanojansky flux scale (MAG_ZERO_AB). Defaults to AnaCal's own "
            "``FpfsConfig.c0`` via ``xlens.utils.constants.FPFS_C0``."
        ),
        default=FPFS_C0,
    )
    pthres = Field[float](
        doc="peak detection threshold",
        default=0.12,
    )
    kmax_thres = Field[float](
        doc="threshold to determine the maximum k in Fourier space",
        default=1e-12,
    )
    do_noise_bias_correction = Field[bool](
        doc="whether to doulbe the noise for noise bias correction",
        default=True,
    )
    do_compute_detect_weight = Field[bool](
        doc="whether to compute detection mode",
        default=True,
    )
    return_only_linear_modes = Field[bool](
        doc="whether only return linear modes",
        default=False,
    )
    psf_model_type = Field[str](
        doc="type of psf model (choose from object, block, patch)",
        default="patch",
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes used to reject bad pixels.",
        default=badMaskDefault,
    )
    noiseId = Field[int](
        doc="Noise realization id",
        default=0,
    )
    rotId = Field[int](
        doc="rotation id",
        default=0,
    )

    def validate(self):
        super().validate()
        if self.sigma_shapelets > 2.0 or self.sigma_shapelets < 0.0:
            raise FieldValidationError(
                self.__class__.sigma_shapelets,
                self,
                "sigma_shapelets in a wrong range",
            )
        if self.sigma_shapelets1 > 2.0:
            raise FieldValidationError(
                self.__class__.sigma_shapelets1,
                self,
                "sigma_shapelets1 in a wrong range",
            )
        if self.sigma_shapelets2 > 2.0:
            raise FieldValidationError(
                self.__class__.sigma_shapelets2,
                self,
                "sigma_shapelets2 in a wrong range",
            )
        if self.noiseId < 0:
            raise FieldValidationError(
                self.__class__.noiseId,
                self,
                "We require noiseId >=0",
            )
        if self.rotId >= num_rot:
            raise FieldValidationError(
                self.__class__.rotId,
                self,
                "rotId needs to be smaller than 2",
            )

    def setDefaults(self):
        super().setDefaults()


class FpfsMeasurementTask(Task):
    """Measure FPFS shapelet observables from coadd image data.

    Wraps :func:`anacal.fpfs.process_image` behind the Rubin ``Task``
    interface.  Call :meth:`run` with the data dict prepared by
    ``AnacalTask.prepare_data``.
    """

    _DefaultName = "FpfsMeasurementTask"
    ConfigClass = FpfsMeasurementConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, FpfsMeasurementConfig)
        self.fpfs_config = anacal.fpfs.FpfsConfig(
            npix=self.config.npix,
            kmax_thres=self.config.kmax_thres,
            sigma_shapelets=self.config.sigma_shapelets,
            sigma_shapelets1=self.config.sigma_shapelets1,
            sigma_shapelets2=self.config.sigma_shapelets2,
            pthres=self.config.pthres,
            bound=self.config.bound,
            snr_min=self.config.snr_min,
            r2_min=self.config.r2_min,
            c0=self.config.c0,
        )
        return

    def run(
        self,
        *,
        pixel_scale: float,
        mag_zero: float,
        noise_variance: float,
        gal_array: NDArray,
        psf_array: NDArray,
        mask_array: NDArray,
        noise_array: NDArray | None,
        detection: NDArray | None,
        psf_object: utils.image.LsstPsf | None,
        base_column_name: str | None = None,
        begin_x: int = 0,
        begin_y: int = 0,
        **kwargs,
    ):
        """Run FPFS measurement on image arrays.

        Parameters
        ----------
        pixel_scale : float
            Pixel scale in arcsec/pixel.
        mag_zero : float
            Magnitude zeropoint.
        noise_variance : float
            Per-pixel noise variance.
        gal_array : NDArray
            Galaxy image array.
        psf_array : NDArray
            PSF image array.
        mask_array : NDArray
            Bad-pixel mask array.
        noise_array : NDArray or None
            Noise realisation for noise-bias correction.
        detection : NDArray or None
            External detection catalog with ``x1_det``, ``x2_det`` columns
            (in arcsec). If *None*, peaks are detected internally.
        psf_object : LsstPsf or None
            Position-dependent PSF model.
        base_column_name : str or None
            Prefix prepended to all output column names.
        begin_x, begin_y : int
            Pixel origin offset for sub-images.

        Returns
        -------
        np.ndarray
            Structured array of FPFS shape measurements.
        """
        assert isinstance(self.config, FpfsMeasurementConfig)
        if detection is not None:
            fpfs_peaks_dtype = np.dtype(
                [
                    ("y", np.float64),
                    ("x", np.float64),
                ]
            )
            det = np.zeros(len(detection), dtype=fpfs_peaks_dtype)
            det["x"] = detection["x1_det"] / pixel_scale - begin_x
            det["y"] = detection["x2_det"] / pixel_scale - begin_y
        else:
            det = None
        catalog = anacal.fpfs.process_image(
            fpfs_config=self.fpfs_config,
            pixel_scale=pixel_scale,
            mag_zero=mag_zero,
            noise_variance=noise_variance,
            gal_array=gal_array,
            psf_array=psf_array,
            mask_array=mask_array,
            noise_array=noise_array,
            detection=det,
            psf_object=psf_object,
            do_compute_detect_weight=self.config.do_compute_detect_weight,
            base_column_name=base_column_name,
            return_only_linear_modes=self.config.return_only_linear_modes,
            pack_linear_modes=True,
        )
        return catalog

