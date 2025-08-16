from typing import Any

import anacal
import numpy as np
from lsst.afw.image import ExposureF
from lsst.pex.config import Config, Field, FieldValidationError, ListField
from lsst.pipe.base import Task
from numpy.typing import NDArray

from .. import utils
from ..utils.random import image_noise_base, num_rot


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
    do_noise_bias_correction = Field[bool](
        doc="whether to doulbe the noise for noise bias correction",
        default=True,
    )
    do_compute_detect_weight = Field[bool](
        doc="whether to compute detection mode",
        default=True,
    )
    use_average_psf = Field[bool](
        doc="whether to compute detection mode",
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


class FpfsMeasurementTask(Task):
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
        psf_object: utils.image.LsstPsf | None,
        base_column_name: str | None = None,
        begin_x: int = 0,
        begin_y: int = 0,
        **kwargs,
    ):
        assert isinstance(self.config, FpfsMeasurementConfig)
        if detection is not None:
            fpfs_peaks_dtype = np.dtype([
                ('y', np.float64),
                ('x', np.float64),
                ('is_peak', np.int32),
                ('mask_value', np.int32),
            ])
            det = np.zeros(len(detection), dtype=fpfs_peaks_dtype)
            det["x"] = detection["x1_det"] / pixel_scale - begin_x
            det["y"] = detection["x2_det"] / pixel_scale - begin_y
            det["is_peak"] = 1
            det["mask_value"] = 0
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
        )
        return catalog

    def prepare_data(
        self,
        *,
        exposure: ExposureF,
        seed: int,
        noise_corr: NDArray | None = None,
        band: str | None = None,
        mask_array: NDArray | None = None,
        star_cat: NDArray | None = None,
        detection: NDArray | None = None,
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
        lsst_bbox = exposure.getBBox()

        if band is None:
            base_column_name = None
        else:
            base_column_name = band + "_"
        data = utils.image.prepare_data(
            exposure=exposure,
            seed=seed,
            noiseId=self.config.noiseId,
            rotId=self.config.rotId,
            npix=self.config.npix,
            noise_corr=noise_corr,
            do_noise_bias_correction=self.config.do_noise_bias_correction,
            badMaskPlanes=self.config.badMaskPlanes,
            star_cat=star_cat,
            mask_array=mask_array,
            detection=detection,
        )

        data["base_column_name"] = base_column_name
        if not self.config.use_average_psf:
            data["psf_object"] = utils.image.LsstPsf(
                psf=exposure.getPsf(), npix=self.config.npix,
                lsst_bbox=lsst_bbox,
            )
        else:
           data["psf_object"] = None
        return data
