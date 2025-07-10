from typing import Any

import anacal
import astropy
import numpy as np
from lsst.afw.detection import InvalidPsfError
from lsst.afw.geom import SkyWcs
from lsst.afw.image import ExposureF
from lsst.geom import Point2D
from lsst.pex.config import Config, Field, FieldValidationError, ListField
from lsst.pipe.base import Task
from numpy.typing import NDArray

from .. import utils


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
        default=5,
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

    def validate(self):
        super().validate()
        if self.sigma_arcsec > 2.0 or self.sigma_arcsec < 0.0:
            raise FieldValidationError(
                self.__class__.sigma_arcsec,
                self,
                "sigma_arcsec in a wrong range",
            )
        n_min = utils.random.image_noise_base // 2
        if self.noiseId < 0 or self.noiseId >= n_min:
            raise FieldValidationError(
                self.__class__.noiseId,
                self,
                "We require 0 <= noiseId < %d" % (n_min),
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

        ratio = 10.0 ** ((mag_zero - 30.0) / 2.5)
        task = anacal.task.Task(
            scale=pixel_scale,
            omega_f=0.06 * ratio,
            v_min=0.013 * ratio,
            omega_v=0.025 * ratio,
            fpfs_c0=8.4 * ratio,
            **self.config_kwargs,
        )

        if detection is not None:
            detection["x1"] = detection["x1"] - begin_x * pixel_scale
            detection["x2"] = detection["x2"] - begin_y * pixel_scale
            detection["x1_det"] = detection["x1_det"] - begin_x * pixel_scale
            detection["x2_det"] = detection["x2_det"] - begin_y * pixel_scale

        catalog = task.process_image(
            gal_array,
            psf_array,
            variance=noise_variance,
            block_list=blocks,
            detection=detection,
            noise_array=noise_array,
            mask_array=mask_array,
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
                        1 - np.sum(lsst_psf.computeImage(
                            Point2D(
                                cc["x1"] / pixel_scale,
                                cc["x2"] / pixel_scale,
                            )
                        ).getArray())
                    )
                    if ep < 1e-2:
                        indexes.append(ic)
                except InvalidPsfError:
                    pass
            catalog = catalog[indexes]

        if wcs is not None:
            ra, dec = wcs.pixelToSkyArray(
                catalog["x1"] / pixel_scale,
                catalog["x2"] / pixel_scale,
                degrees=True,
            )
            catalog["ra"] = ra
            catalog["dec"] = dec
        condition = (
            (skyMap is not None)
            and (tractInfo is not None)
            and (patchInfo is not None)
        )
        if condition:
            utils.catalog.set_isPrimary(
                catalog,
                skyMap,
                tractInfo,
                patchInfo,
                pixel_scale,
            )
        return catalog

    def prepare_data(
        self,
        *,
        exposure: ExposureF,
        seed: int,
        noise_corr: NDArray | None = None,
        band: str | None = None,
        skyMap=None,
        tract: int = 0,
        patch: int = 0,
        mask_array: NDArray | None = None,
        star_cat: NDArray | None = None,
        detection: astropy.table.Table | None = None,
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
            mask_array=mask_array,
            detection=detection,
        )
        blocks = utils.image.get_blocks(
            exposure.getPsf(),
            exposure.getBBox(),
            exposure.mask,
            data["pixel_scale"],
            self.config.npix,
        )
        data["blocks"] = blocks
        if self.config.validate_psf:
            data["lsst_psf"] = exposure.getPsf()
        else:
            data["lsst_psf"] = None
        return data
