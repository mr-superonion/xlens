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

from typing import Any

import anacal
import astropy
import numpy as np
from lsst.afw.geom import SkyWcs
from lsst.afw.image import ExposureF
from lsst.geom import Point2D
from lsst.pex.config import Config, Field, FieldValidationError, ListField
from lsst.pipe.base import Task
from numpy.typing import NDArray

from .. import utils
from ..wcs import pixel_to_sky


class AnacalConfig(Config):
    npix = Field[int](
        doc="number of pixels in stamp",
        default=64,
    )
    bound = Field[int](
        doc="Sources to be removed if too close to boundary",
        default=40,
    )
    sigma_arcsec = Field[float](
        doc="Kernel size for re-smoothing",
        default=0.40,
    )
    snr_min = Field[float](
        doc="snr min for detection",
        default=5.0,
    )
    num_epochs = Field[int](
        doc="Number of iterations",
        default=0,
    )
    force_size = Field[bool](
        doc="Whether forcing the size and shape of galaxies",
        default=False,
    )
    force_center = Field[bool](
        doc="Whether forcing the size and shape of galaxies",
        default=True,
    )
    validate_psf = Field[bool](
        doc="Whether validating PSF",
        default=False,
    )
    p_min = Field[float](
        doc="peak detection threshold",
        default=0.12,
    )
    omega_p = Field[float](
        doc="peak detection threshold",
        default=0.05,
    )
    do_noise_bias_correction = Field[bool](
        doc="whether to doulbe the noise for noise bias correction",
        default=True,
    )
    do_fpfs = Field[bool](
        doc="whether to do FPFS measurement",
        default=True,
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes used to reject bad pixels.",
        default=[],
    )
    noiseId = Field[int](
        doc="Noise realization id",
        default=0,
    )
    rotId = Field[int](
        doc="rotation id",
        default=0,
    )
    psf_model_type = Field[str](
        doc="type of psf model (choose from object, block, patch)",
        default="patch",
    )

    def validate(self):
        super().validate()
        if self.sigma_arcsec > 2.0 or self.sigma_arcsec < 0.0:
            raise FieldValidationError(
                self.__class__.sigma_arcsec,
                self,
                "sigma_arcsec in a wrong range",
            )
        if self.noiseId < 0:
            raise FieldValidationError(
                self.__class__.noiseId,
                self,
                "We require noiseId >=0",
            )
        if self.rotId >= utils.random.num_rot:
            raise FieldValidationError(
                self.__class__.rotId,
                self,
                "rotId needs to be smaller than 2",
            )

    def setDefaults(self):
        super().setDefaults()


class AnacalTask(Task):
    """Measure Fpfs FPFS observables"""

    _DefaultName = "AnacalTask"
    ConfigClass = AnacalConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, AnacalConfig)
        prior = anacal.ngmix.modelPrior()
        prior.set_sigma_a(anacal.math.qnumber(0.05))
        prior.set_sigma_x(anacal.math.qnumber(0.05))
        self.config_kwargs = {
            "p_min": self.config.p_min,
            "omega_p": self.config.omega_p,
            "sigma_arcsec": self.config.sigma_arcsec,
            "snr_peak_min": self.config.snr_min,
            "stamp_size": self.config.npix,
            "image_bound": self.config.bound,
            "num_epochs": self.config.num_epochs,
            "force_size": self.config.force_size,
            "force_center": self.config.force_center,
            "prior": prior,
        }
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
        begin_x: int = 0,
        begin_y: int = 0,
        wcs: SkyWcs | None = None,
        skyMap=None,
        tractInfo=None,
        patchInfo=None,
        detection: NDArray | None,
        lsst_psf=None,
        blocks,
        **kwargs,
    ):
        assert isinstance(self.config, AnacalConfig)

        # Base thresholds are at the AnaCal reference zeropoint (30); the Task
        # owns the mag_zero threshold scaling. The image is already normalized to
        # MAG_ZERO_AB upstream (prepare_data), so ``mag_zero`` here is MAG_ZERO_AB.
        task = anacal.task.Task(
            scale=pixel_scale,
            omega_f=0.06,
            v_min=0.013,
            omega_v=0.025,
            fpfs_c0=8.4,
            mag_zero=mag_zero,
            **self.config_kwargs,
        )

        if detection is not None:
            det = detection.copy()
            det["x1"] = det["x1"] - begin_x * pixel_scale
            det["x2"] = det["x2"] - begin_y * pixel_scale
            det["x1_det"] = det["x1_det"] - begin_x * pixel_scale
            det["x2_det"] = det["x2_det"] - begin_y * pixel_scale
        else:
            det = None
        catalog = task.process_image(
            gal_array,
            psf_array,
            variance=noise_variance,
            block_list=blocks,
            detection=det,
            noise_array=noise_array,
            mask_array=mask_array,
            do_fpfs=self.config.do_fpfs,
        )
        catalog["x1"] = catalog["x1"] + begin_x * pixel_scale
        catalog["x2"] = catalog["x2"] + begin_y * pixel_scale
        catalog["x1_det"] = catalog["x1_det"] + begin_x * pixel_scale
        catalog["x2_det"] = catalog["x2_det"] + begin_y * pixel_scale
        if self.config.validate_psf and (lsst_psf is not None):
            indexes = []
            for ic, cc in enumerate(catalog):
                try:
                    ep = np.abs(
                        1
                        - np.sum(
                            lsst_psf.computeImage(
                                Point2D(
                                    cc["x1"] / pixel_scale,
                                    cc["x2"] / pixel_scale,
                                )
                            ).getArray()
                        )
                    )
                    if ep < 1e-1:
                        indexes.append(ic)
                except Exception:
                    pass
            catalog = catalog[indexes]

        if wcs is not None:
            # catalog x1/x2 are already global (parent) pixels here, since
            # begin_x/begin_y were added back above -> no XY0 offset needed.
            ra, dec = pixel_to_sky(
                wcs,
                catalog["x1"] / pixel_scale,
                catalog["x2"] / pixel_scale,
            )
            catalog["ra"] = ra
            catalog["dec"] = dec
        return catalog

    def prepare_data(
        self,
        *,
        exposure: ExposureF,
        seed: int,
        band: str | None,
        noise_corr: NDArray | None = None,
        skyMap=None,
        tract: int = 0,
        patch: int = 0,
        star_cat: NDArray | None = None,
        psf_array: NDArray | None = None,
        mask_array: NDArray | None = None,
        noise_array: NDArray | None = None,
        detection: astropy.table.Table | None = None,
        blocks: list | None = None,
        **kwargs,
    ):
        """Prepares the data from LSST exposure
        Args:
        exposure (ExposureF):   LSST exposure
        seed (int):  random seed
        noise_corr (NDArray):  image noise correlation function (None)
        tractInfo:  tract information
        patchInfo:  patch information

        Returns:
            (dict)
        """
        assert isinstance(self.config, AnacalConfig)
        pixel_scale = float(exposure.wcs.getPixelScale().asArcseconds())
        if blocks is None:
            blocks = utils.image.get_blocks(
                lsst_psf=exposure.getPsf(),
                lsst_bbox=exposure.getBBox(),
                pixel_scale=pixel_scale,
                npix=self.config.npix,
                psf_array=psf_array,
            )
        data = utils.image.prepare_data(
            exposure=exposure,
            seed=seed,
            noiseId=self.config.noiseId,
            rotId=self.config.rotId,
            npix=self.config.npix,
            noise_corr=noise_corr,
            do_noise_bias_correction=self.config.do_noise_bias_correction,
            badMaskPlanes=self.config.badMaskPlanes,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            star_cat=star_cat,
            psf_array=psf_array,
            mask_array=mask_array,
            noise_array=noise_array,
            detection=detection,
            band=band,
            blocks=blocks,
        )
        if self.config.validate_psf:
            data["lsst_psf"] = exposure.getPsf()
        else:
            data["lsst_psf"] = None
        if band is None:
            data["base_column_name"] = None
        else:
            data["base_column_name"] = band + "_"
        if self.config.psf_model_type == "object":
            data["psf_object"] = utils.image.LsstPsf(
                psf=exposure.getPsf(),
                npix=self.config.npix,
                lsst_bbox=exposure.getBBox(),
            )
        else:
            data["psf_object"] = None
        return data
