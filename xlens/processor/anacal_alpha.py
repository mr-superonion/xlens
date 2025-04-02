from typing import Any

import anacal
import numpy as np
from lsst.afw.image import ExposureF
from lsst.pex.config import Config, Field, FieldValidationError, ListField
from numpy.typing import NDArray

from ..simulator.random import get_noise_seed, image_noise_base, num_rot
from . import utils
from .base import MeasBaseTask


class AnacalAlphaConfig(Config):
    npix = Field[int](
        doc="number of pixels in stamp",
        default=32,
    )
    bound = Field[int](
        doc="Sources to be removed if too close to boundary",
        default=40,
    )
    sigma_arcsec_det = Field[float](
        doc="kernel size for detection",
        default=0.52,
    )
    sigma_arcsec = Field[float](
        doc="Kernel size for re-smoothing",
        default=0.42,
    )
    snr_min = Field[float](
        doc="snr min for detection",
        default=8.0,
    )
    num_epochs = Field[int](
        doc="Number of iterations",
        default=0,
    )
    force_size = Field[bool](
        doc="Whether forcing the size and shape of galaxies",
        default=True,
    )
    p_min = Field[float](
        doc="peak detection threshold",
        default=0.14,
    )
    omega_p = Field[float](
        doc="peak detection threshold",
        default=0.05,
    )
    use_average_psf = Field[bool](
        doc="whether to use average PSF over the exposure",
        default=True,
    )
    do_noise_bias_correction = Field[bool](
        doc="whether to doulbe the noise for noise bias correction",
        default=True,
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes used to reject bad pixels.",
        default=["BAD", "SAT", "CR"],  # I keep CR here
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
        if self.sigma_arcsec_det > 2.0 or self.sigma_arcsec_det < 0.0:
            raise FieldValidationError(
                self.__class__.sigma_arcsec_det,
                self,
                "sigma_arcsec_det in a wrong range",
            )
        if self.noiseId < 0 or self.noiseId >= image_noise_base // 2:
            raise FieldValidationError(
                self.__class__.noiseId,
                self,
                "We require 0 <= noiseId < %d" % (image_noise_base // 2),
            )
        if self.rotId >= num_rot:
            raise FieldValidationError(
                self.__class__.rotId,
                self,
                "rotId needs to be smaller than 2",
            )

    def setDefaults(self):
        super().setDefaults()


class AnacalAlphaTask(MeasBaseTask):
    """Measure Fpfs FPFS observables"""

    _DefaultName = "AnacalAlphaTask"
    ConfigClass = AnacalAlphaConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, AnacalAlphaConfig)
        prior = anacal.ngmix.modelPrior()
        prior.set_sigma_a(anacal.math.qnumber(0.2))
        prior.set_sigma_x(anacal.math.qnumber(0.5))
        self.config_kwargs = {
            "p_min": self.config.p_min,
            "omega_p": self.config.omega_p,
            "sigma_arcsec_det": self.config.sigma_arcsec_det,
            "sigma_arcsec": self.config.sigma_arcsec,
            "snr_peak_min": self.config.snr_min,
            "stamp_size": self.config.npix,
            "image_bound": self.config.bound,
            "num_epochs": self.config.num_epochs,
            "force_size": self.config.force_size,
            "force_center": True,
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
        psf_object: utils.LsstPsf | None,
        base_column_name: str | None,
        **kwargs,
    ):
        assert isinstance(self.config, AnacalAlphaConfig)

        ratio = 10.0 ** ((mag_zero - 30.0) / 2.5)
        taskA = anacal.task.TaskAlpha(
            scale=pixel_scale,
            omega_f=0.06 * ratio,
            v_min=0.013 * ratio,
            omega_v=0.025 * ratio,
            fpfs_c0=8.4 * ratio,
            **self.config_kwargs,
        )

        blocks = anacal.geometry.get_block_list(
            gal_array.shape[0],         # image size
            gal_array.shape[1],
            256,                        # block size
            256,
            self.config.npix * 2 + 10,   # bound
            pixel_scale,
        )

        return taskA.process_image(
            gal_array,
            psf_array,
            variance=noise_variance,
            block_list=blocks,
            noise_array=noise_array,
        )

    def prepare_data(
        self,
        *,
        exposure: ExposureF,
        seed: int,
        noise_corr: NDArray | None = None,
        detection: NDArray | None = None,
        band: str | None = None,
        **kwargs,
    ):
        """Prepares the data from LSST exposure
        Args:
        exposure (ExposureF):   LSST exposure
        seed (int):  random seed
        noise_corr (NDArray):  image noise correlation function (None)
        detection (NDArray | None):  external detection catalog (None)

        Returns:
            (dict)
        """
        assert isinstance(self.config, AnacalAlphaConfig)

        pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
        noise_variance = np.average(exposure.getMaskedImage().variance.array)
        if noise_variance < 1e-12:
            raise ValueError(
                "the estimated image noise variance should be positive."
            )
        noise_std = np.sqrt(noise_variance)
        mag_zero = (
            np.log10(exposure.getPhotoCalib().getInstFluxAtZeroMagnitude())
            / 0.4
        )

        lsst_bbox = exposure.getBBox()
        lsst_psf = exposure.getPsf()
        psf_array = np.asarray(
            utils.get_psf_array(
                lsst_psf=lsst_psf,
                lsst_bbox=lsst_bbox,
                npix=self.config.npix,
                dg=250,
            ),
            dtype=np.float64,
        )
        gal_array = np.asarray(
            exposure.getMaskedImage().image.array,
            dtype=np.float64,
        )

        bitValue = exposure.mask.getPlaneBitMask(self.config.badMaskPlanes)
        mask_array = ((exposure.mask.array & bitValue) != 0).astype(np.int16)

        if self.config.do_noise_bias_correction:
            # TODO: merge the following to one code
            noise_seed = (
                get_noise_seed(
                    seed=seed,
                    noiseId=self.config.noiseId,
                    rotId=self.config.rotId,
                )
                + image_noise_base // 2
                # make sure the seed is different from
                # noise seed for simulation
            )
            ny, nx = gal_array.shape
            if noise_corr is None:
                self.log.debug(
                    "use white noise for noise bias correction"
                )
                noise_array = (
                    np.random.RandomState(noise_seed)
                    .normal(
                        scale=noise_std,
                        size=(ny, nx),
                    )
                    .astype(np.float64)
                )
            else:
                self.log.debug(
                    "use noise correlation function for noise bias correction"
                )
                noise_corr = np.rot90(m=noise_corr, k=-1)
                noise_array = (
                    anacal.noise.simulate_noise(
                        seed=noise_seed,
                        correlation=noise_corr,
                        nx=nx,
                        ny=ny,
                        scale=pixel_scale,
                    ) * noise_std
                )
        else:
            self.log.debug(
                "Do not correct for noise bias"
            )
            noise_array = None

        if not self.config.use_average_psf:
            psf_object = utils.LsstPsf(psf=lsst_psf, npix=self.config.npix)
        else:
            psf_object = None

        if band is None:
            base_column_name = None
        else:
            base_column_name = band + "_"
        return {
            "pixel_scale": pixel_scale,
            "mag_zero": mag_zero,
            "noise_variance": noise_variance,
            "gal_array": gal_array,
            "psf_array": psf_array,
            "mask_array": mask_array,
            "noise_array": noise_array,
            "psf_object": psf_object,
            "base_column_name": base_column_name,
        }
