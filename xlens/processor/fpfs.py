from typing import Any

import anacal
import numpy as np
from lsst.afw.image import ExposureF
from lsst.pex.config import Config, Field, FieldValidationError, ListField
from numpy.typing import NDArray

from ..simulator.random import get_noise_seed, image_noise_base, num_rot
from . import utils
from .base import MeasBaseTask


class FpfsMeasurementConfig(Config):
    npix = Field[int](
        doc="number of pixels in stamp",
        default=64,
    )
    bound = Field[int](
        doc="Sources to be removed if too close to boundary",
        default=32,
    )
    sigma_arcsec = Field[float](
        doc="Shapelet's Gaussian kernel size for detection",
        default=0.52,
    )
    sigma_arcsec1 = Field[float](
        doc="Shapelet's Gaussian kernel size for measurement",
        optional=True,
        default=-1,
    )
    sigma_arcsec2 = Field[float](
        doc="Shapelet's Gaussian kernel size for the second measurement",
        optional=True,
        default=-1,
    )
    snr_min = Field[float](
        doc="Shapelet's Gaussian kernel size for the second measurement",
        optional=True,
        default=12.0,
    )
    r2_min = Field[float](
        doc="Shapelet's Gaussian kernel size for the second measurement",
        optional=True,
        default=0.1,
    )
    pthres = Field[float](
        doc="peak detection threshold",
        default=0.12,
    )
    kmax_thres = Field[float](
        doc="threshold to determine the maximum k in Fourier space",
        default=1e-12,
    )
    use_average_psf = Field[bool](
        doc="whether to use average PSF over the exposure",
        default=True,
    )
    do_noise_bias_correction = Field[bool](
        doc="whether to doulbe the noise for noise bias correction",
        default=True,
    )
    do_compute_detect_weight = Field[bool](
        doc="whether to compute detection mode",
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
        if self.sigma_arcsec1 > 2.0:
            raise FieldValidationError(
                self.__class__.sigma_arcsec1,
                self,
                "sigma_arcsec1 in a wrong range",
            )
        if self.sigma_arcsec2 > 2.0:
            raise FieldValidationError(
                self.__class__.sigma_arcsec2,
                self,
                "sigma_arcsec2 in a wrong range",
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


class FpfsMeasurementTask(MeasBaseTask):
    """Measure Fpfs FPFS observables"""

    _DefaultName = "FpfsMeasurementTask"
    ConfigClass = FpfsMeasurementConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, FpfsMeasurementConfig)
        self.fpfs_config = anacal.fpfs.FpfsConfig(
            npix=self.config.npix,
            kmax_thres=self.config.kmax_thres,
            sigma_arcsec=self.config.sigma_arcsec,
            sigma_arcsec1=self.config.sigma_arcsec1,
            sigma_arcsec2=self.config.sigma_arcsec2,
            pthres=self.config.pthres,
            bound=self.config.bound,
            snr_min=self.config.snr_min,
            r2_min=self.config.r2_min,
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
        psf_object: utils.LsstPsf | None,
        base_column_name: str | None,
        **kwargs,
    ):
        assert isinstance(self.config, FpfsMeasurementConfig)
        return anacal.fpfs.process_image(
            fpfs_config=self.fpfs_config,
            pixel_scale=pixel_scale,
            mag_zero=mag_zero,
            noise_variance=noise_variance,
            gal_array=gal_array,
            psf_array=psf_array,
            mask_array=mask_array,
            noise_array=noise_array,
            detection=detection,
            psf_object=psf_object,
            do_compute_detect_weight=self.config.do_compute_detect_weight,
            base_column_name=base_column_name,
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
        assert isinstance(self.config, FpfsMeasurementConfig)

        def rotate90(image):
            rotated_image = np.zeros_like(image)
            rotated_image[1:, 1:] = np.rot90(m=image[1:, 1:], k=-1)
            return rotated_image

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
                noise_array = (
                    np.random.RandomState(noise_seed)
                    .normal(
                        scale=noise_std,
                        size=(ny, nx),
                    )
                    .astype(np.float64)
                )
            else:
                noise_corr = rotate90(noise_corr.getArray())
                noise_array = (
                    anacal.noise.simulate_noise(
                        seed=noise_seed,
                        correlation=noise_corr,
                        nx=nx,
                        ny=ny,
                        scale=pixel_scale,
                    ).astype(np.float64)
                    * noise_std
                )
        else:
            noise_array = None
        if detection is not None:
            detection = np.array(detection[["y", "x", "is_peak", "mask_value"]])

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
            "detection": detection,
            "psf_object": psf_object,
            "base_column_name": base_column_name,
        }
