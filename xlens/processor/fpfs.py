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
from lsst.pex.config import Config, Field, FieldValidationError
from lsst.pipe.base import Task
from numpy.typing import NDArray

from .. import utils
from ..utils.constants import FPFS_C0
from ..utils.random import num_rot


class FpfsMeasurementConfig(Config):
    """Configuration for :class:`FpfsMeasurementTask`."""

    npix = Field[int](
        doc="number of pixels in stamp [pixel]",
        default=64,
    )
    sigma_shapelets1 = Field[float](
        doc=(
            "Shapelet's Gaussian kernel size for measurement [arcsec]. "
            "REQUIRED (> 0) whenever the fpfs subtask runs."
        ),
        optional=True,
        default=-1,
    )
    sigma_shapelets2 = Field[float](
        doc="Shapelet's Gaussian kernel for the second measurement [arcsec]",
        optional=True,
        default=-1,
    )
    c0 = Field[float](
        doc=(
            "C0 normalisation in ``e = m22c / (m00 + c0)``, on the fixed AB "
            "nanojansky flux scale (MAG_ZERO_AB). Defaults to AnaCal's own "
            "``FpfsConfig.c0`` via ``xlens.utils.constants.FPFS_C0``."
        ),
        default=FPFS_C0,
    )
    kmax_thres = Field[float](
        doc="threshold to determine the maximum k in Fourier space",
        default=1e-12,
    )
    do_noise_bias_correction = Field[bool](
        doc="whether to doulbe the noise for noise bias correction",
        default=True,
    )
    mask_value_max = Field[int](
        doc=(
            "Skip forced measurement of sources whose mask_value exceeds "
            "this (their output rows are zero-filled; applied in C++ "
            "inside ForceTask). None disables the cut. In the "
            "measure*coadds tasks this is DERIVED: validate() mirrors "
            "config.anacal.mask_value_max here, so set that instead."
        ),
        default=None,
        optional=True,
    )

    psf_model_type = Field[str](
        doc="type of psf model (choose from object, cell, patch)",
        default="patch",
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
            sigma_shapelets1=self.config.sigma_shapelets1,
            sigma_shapelets2=self.config.sigma_shapelets2,
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
        detection : NDArray
            Detection catalog from the AnaCal detector with ``x1_det``,
            ``x2_det`` columns (in arcsec). Required: FPFS no longer
            detects internally (``anacal.fpfs.process_image`` raises
            without it).
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
        # Native per-source PSF: hand the C++ ForceTask the model
        # itself -- every stamp is drawn inside its GIL-released loop
        # (no Python per-galaxy drawing), and sources outside the
        # model's coverage get mask_value = 404 written back in place.
        # The 404 sentinel is always skipped by the C++ measurement,
        # with or without a configured mask_value_max cut.
        psf_model = getattr(psf_object, "native_model", None)
        psf_offset = (
            float(getattr(psf_object, "x_min", 0.0)),
            float(getattr(psf_object, "y_min", 0.0)),
        )
        mask_value = None
        mask_value_max = self.config.mask_value_max
        has_mask_col = detection is not None and "mask_value" in (
            detection.dtype.names or ()
        )
        if has_mask_col:
            mask_value = np.ascontiguousarray(
                detection["mask_value"], dtype=np.int32
            )
        elif psf_model is not None and detection is not None:
            # writable sentinel target even without a systematics mask
            mask_value = np.zeros(len(detection), dtype=np.int32)
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
            base_column_name=base_column_name,
            mask_value=mask_value,
            mask_value_max=mask_value_max,
            psf_model=psf_model,
            psf_offset=psf_offset,
        )
        if (
            psf_model is not None
            and mask_value is not None
            and detection is not None
            and has_mask_col
        ):
            # propagate 404 sentinels the C++ wrote into the caller's
            # catalog, so later bands skip the same sources and the
            # output rows carry the flag
            detection["mask_value"] = mask_value
        return catalog
