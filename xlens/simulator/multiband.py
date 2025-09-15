#!/usr/bin/env python
#
# simple example with ring test (rotating intrinsic galaxies)
# Copyright 20230916 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
from typing import Any

import anacal
import galsim
import lsst
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.pipe.base import Task
import lsst.meas.algorithms as meaAlg
import numpy as np
from lsst.afw.cameraGeom.testUtils import DetectorWrapper
from lsst.pex.config import Config, Field, FieldValidationError, ListField
from lsst.pipe.base import Struct
from numpy.typing import NDArray
from .galaxies import CatSim2017Catalog, OpenUniverse2024RubinRomanCatalog

from ..utils.random import (
    gal_seed_base,
    get_noise_seed,
    num_rot,
)
from .multiband_defaults import (
    mag_zero_defaults,
    noise_variance_defaults,
    psf_fwhm_defaults,
    sys_npix,
)
from .perturbation import ShearHalo, ShearRedshift


def get_noise_array(
    *,
    seed_noise: int,
    noise_std: float,
    noise_corr: NDArray | None,
    shape: tuple[int, int],
    pixel_scale: float,
) -> NDArray:
    if noise_corr is None:
        noise_array = np.random.RandomState(seed_noise).normal(
            scale=noise_std,
            size=shape,
        )
    else:
        noise_array = (
            anacal.noise.simulate_noise(
                seed=seed_noise,
                correlation=noise_corr,
                nx=shape[1],
                ny=shape[0],
                scale=pixel_scale,
            )
            * noise_std
        )
    return noise_array


class SimBaseTask(Task):
    _DefaultName = "SimBaseTask"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        pass

    def get_perturbation_object(self, **kwargs: Any) -> object:
        raise NotImplementedError(
            "'get_perturbation_object' must be implemented by subclasses."
        )


class MultibandSimBaseConfig(Config):
    galaxy_type = Field[str](
        doc="galaxy type",
        default="catsim2017",
    )
    survey_name = Field[str](
        doc="Name of the survey",
        default="LSST",
    )
    include_pixel_masks = Field[bool](
        doc="whether to include pixel masks in the simulation",
        default=False,
    )
    draw_image_noise = Field[bool](
        doc="Whether to draw image noise in the simulation",
        default=False,
    )
    galId = Field[int](
        doc="random seed index for galaxy, 0 <= galId < 10",
        default=0,
    )
    rotId = Field[int](
        doc="number of rotations",
        default=0,
    )
    noiseId = Field[int](
        doc="random seed index for noise, 0 <= noiseId < 10",
        default=0,
    )
    use_real_psf = Field[bool](
        doc="whether to use real PSF",
        default=False,
    )

    def validate(self):
        super().validate()
        if self.galId >= gal_seed_base or self.galId < 0:
            raise FieldValidationError(
                self.__class__.galId,
                self,
                "We require 0 <= galId < %d" % (gal_seed_base),
            )
        if self.rotId >= num_rot:
            raise FieldValidationError(
                self.__class__.rotId,
                self,
                "rotId needs to be smaller than 2",
            )
        if self.noiseId < 0:
            raise FieldValidationError(
                self.__class__.noiseId,
                self,
                "We require noiseId >=0 ",
            )
        if self.galaxy_type not in ["catsim2017", "RomanRubin2024"]:
            raise FieldValidationError(
                self.__class__.galaxy_type,
                self,
                "We require galaxy_type in ['catsim2017', 'RomanRubin2024']",
            )

    def setDefaults(self):
        super().setDefaults()
        self.survey_name = self.survey_name.lower()


class MultibandSimBaseTask(SimBaseTask):
    _DefaultName = "MultibandSimBaseTask"
    ConfigClass = MultibandSimBaseConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, MultibandSimBaseConfig)
        self.rotate_list = [np.pi / num_rot * i for i in range(num_rot)]
        pass

    def prepare_galaxy_catalog(
        self,
        *,
        catalog,
    ):
        assert isinstance(self.config, MultibandSimBaseConfig)
        if self.config.galaxy_type == "catsim2017":
            GalClass = CatSim2017Catalog
        elif self.config.galaxy_type == "RomanRubin2024":
            GalClass = OpenUniverse2024RubinRomanCatalog
        else:
            raise ValueError("invalid galaxy_type")
        galaxy_catalog = GalClass.from_array(table=catalog)
        return galaxy_catalog

    def simulate_images(
        self,
        *,
        wcs: lsst.afw.geom.SkyWcs,
        boundary_box: lsst.geom.Box2I,
        galaxy_catalog,
        shear_obj,
        psf_obj,
        rotId: int,
        band: str,
        mag_zero: float,
        draw_method: str = "auto",
        survey_name: str = "lsst",
        **kwargs,
    ):
        assert isinstance(self.config, MultibandSimBaseConfig)
        theta0 = self.rotate_list[rotId]
        objlist = galaxy_catalog.get_objlist(
            mag_zero=mag_zero, band=band, survey_name=survey_name,
        )

        return make_exp(
            wcs=wcs,
            boundary_box=boundary_box,
            gal_list=objlist["objlist"],
            shifts=objlist["shifts"],
            redshifts=objlist["redshifts"],
            indexes=objlist["indexes"],
            psf=psf_obj,
            shear_obj=shear_obj,
            draw_method=draw_method,
            theta0=theta0,
        )

    def run(
        self,
        *,
        band: str,
        seed: int,
        boundary_box: lsst.geom.Box2I,
        wcs: lsst.afw.geom.SkyWcs,
        catalog,
        psfImage: afwImage.ImageF | None = None,
        noiseCorrImage: afwImage.ImageF | None = None,
        exposure: afwImage.ExposureF | None = None,
        **kwargs,
    ):
        assert isinstance(self.config, MultibandSimBaseConfig)
        if self.config.use_real_psf:
            if psfImage is None:
                raise IOError("Do not have PSF input model")

        # Prepare the random number generator and basic parameters
        rotId = self.config.rotId
        survey_name = self.config.survey_name

        # Get the pixel scale in arcseconds per pixel
        pixel_scale = wcs.getPixelScale().asArcseconds()
        width = boundary_box.getWidth()
        height = boundary_box.getHeight()
        mag_zero = mag_zero_defaults[self.config.survey_name]
        zero_flux = 10.0 ** (0.4 * mag_zero)
        photo_calib = afwImage.makePhotoCalibFromCalibZeroPoint(zero_flux)

        if exposure is not None:
            self.log.debug("Using the real pixel mask")
            mask_array = exposure.getMaskedImage().mask.array
            assert mag_zero == 2.5 * np.log10(
                exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()
            )
        else:
            self.log.debug("Do not use the real pixel mask")
            mask_array = 0.0

        # Obtain PSF object for Galsim
        if psfImage is not None and self.config.use_real_psf:
            psf_galsim = galsim.InterpolatedImage(
                galsim.Image(psfImage.getArray()),
                scale=pixel_scale,
                flux=1.0,
            )
            draw_method = "no_pixel"
        else:
            psf_fwhm = psf_fwhm_defaults[band][survey_name]
            psf_galsim = galsim.Moffat(fwhm=psf_fwhm, beta=2.5)
            psf_array = psf_galsim.drawImage(
                nx=sys_npix,
                ny=sys_npix,
                scale=pixel_scale,
                wcs=None,
            ).array
            psfImage = afwImage.ImageF(sys_npix, sys_npix)
            assert psfImage is not None
            psfImage.array[:, :] = psf_array
            draw_method = "auto"

        # and psf kernel for the LSST exposure
        kernel = afwMath.FixedKernel(psfImage.convertD())
        kernel_psf = meaAlg.KernelPsf(kernel)

        # Obtain Noise correlation array
        if noiseCorrImage is None:
            noise_corr = None
            variance = noise_variance_defaults[band][survey_name]
            self.log.debug("No correlation, variance:", variance)
        else:
            noise_corr = noiseCorrImage.getArray()
            variance = np.amax(noise_corr)
            noise_corr = noise_corr / variance
            ny, nx = noise_corr.shape
            assert noise_corr[ny // 2, nx // 2] == 1
            self.log.debug("With correlation, variance:", variance)
        noise_std = np.sqrt(variance)

        galaxy_catalog = self.prepare_galaxy_catalog(catalog=catalog)
        shear_obj = self.get_perturbation_object()
        data, truth_catalog = self.simulate_images(
            boundary_box=boundary_box,
            wcs=wcs,
            galaxy_catalog=galaxy_catalog,
            shear_obj=shear_obj,
            psf_obj=psf_galsim,
            rotId=rotId,
            band=band,
            mag_zero=mag_zero,
            draw_method=draw_method,
            survey_name=survey_name,
        )

        exp_out = afwImage.ExposureF(boundary_box)
        exp_out.getMaskedImage().image.array[:, :] = data
        exp_out.setPhotoCalib(photo_calib)
        exp_out.setPsf(kernel_psf)
        exp_out.setWcs(wcs)
        exp_out.getMaskedImage().variance.array[:, :] = variance
        filter_label = afwImage.FilterLabel(band=band, physical=band)
        exp_out.setFilter(filter_label)
        detector = DetectorWrapper().detector
        exp_out.setDetector(detector)
        del data, photo_calib, kernel_psf, filter_label, detector

        if self.config.draw_image_noise:
            galaxy_seed = seed * gal_seed_base + self.config.galId
            seed_noise = get_noise_seed(
                galaxy_seed=galaxy_seed,
                noiseId=self.config.noiseId,
                rotId=self.config.rotId,
                band=band,
                is_sim=True,
            )
            noise_array = get_noise_array(
                seed_noise=seed_noise,
                noise_std=noise_std,
                noise_corr=noise_corr,
                shape=(height, width),
                pixel_scale=pixel_scale,
            )
            exp_out.getMaskedImage().image.array[:, :] = (
                exp_out.getMaskedImage().image.array[:, :] + noise_array
            )
        exp_out.getMaskedImage().mask.array[:, :] = mask_array
        del mask_array

        outputs = Struct(
            outputExposure=exp_out, outputTruthCatalog=truth_catalog
        )
        return outputs


class MultibandSimShearTaskConfig(MultibandSimBaseConfig):
    z_bounds = ListField[float](
        doc="boundary list of the redshift",
        default=[-0.01, 20.0],
    )
    mode = Field[int](
        doc=(
            "Note that there are three options in each redshift bin\n"
            "+ 0: g=-0.02;\n"
            "+ 1: g=0.02;\n"
            "+ 2: g=0.00\n\n"
            "For example, if the number of redshift bins is 4 (with z_bounds: "
            "[0., 0.5, 1.0, 1.5, 2.0]), and mode = 7 which in ternary "
            "is '0021' - this means the shear is (-0.02, -0.02, 0., 0.02) in "
            "each bin, respectively."
        ),
        default=0,
    )
    test_target = Field[str](
        doc="the shear component to test",
        default="g1",
    )
    test_value = Field[float](
        doc="absolute value of the shear",
        default=0.02,
    )
    kappa_value = Field[float](
        doc="kappa value to use, 0. means no kappa",
        default=0.,
    )

    def validate(self):
        super().validate()
        n_zbins = len(self.z_bounds) - 1
        mode_max = 3 ** (n_zbins)
        if self.mode >= mode_max:
            raise FieldValidationError(
                self.__class__.mode,
                self,
                "mode needs to be smaller than %d" % mode_max,
            )

        if self.test_target not in ["g1", "g2"]:
            raise FieldValidationError(
                self.__class__.test_target,
                self,
                "test target can only be 'g1' or 'g2'",
            )

        if self.test_value < 0.0 or self.test_value > 0.30:
            raise FieldValidationError(
                self.__class__.test_value,
                self,
                "test_value should be in [0.00, 0.30]",
            )


    def setDefaults(self):
        super().setDefaults()


class MultibandSimShearTask(MultibandSimBaseTask):
    _DefaultName = "MultibandSimShearTask"
    ConfigClass = MultibandSimShearTaskConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def get_perturbation_object(self, **kwargs: Any):
        assert isinstance(self.config, MultibandSimShearTaskConfig)
        return ShearRedshift(
            mode=self.config.mode,
            g_dist=self.config.test_target,
            shear_value=self.config.test_value,
            z_bounds=self.config.z_bounds,
            kappa_value=self.config.kappa_value,
        )


class MultibandSimHaloTaskConfig(MultibandSimBaseConfig):
    mass = Field[float](
        doc="halo mass",
        default=5e-14,
    )
    conc = Field[float](
        doc="halo concertration",
        default=1.0,
    )
    z_lens = Field[float](
        doc="halo redshift",
        default=1.0,
    )
    ra_lens = Field[float](
        doc="halo ra [degree]",
        default=200.0,
    )
    dec_lens = Field[float](
        doc="halo dec [degree]",
        default=0.0,
    )
    z_source = Field[float](
        doc="source galaxy redshift",
        default=None,
    )

    no_kappa = Field[bool](
        doc="whether to exclude kappa field",
        default=False,
    )

    def validate(self):
        super().validate()
        if self.mass < 1e8:
            raise FieldValidationError(
                self.__class__.mass,
                self,
                "halo mass too small",
            )
        if self.z_lens < 0 or self.z_lens > 10.0:
            raise FieldValidationError(
                self.__class__.z_lens,
                self,
                "halo redshift is wrong",
            )
        if self.z_lens > self.z_source:
            raise FieldValidationError(
                self.__class__.z_lens,
                self,
                "halo redshift is larger than source redshift",
            )

    def setDefaults(self):
        super().setDefaults()


class MultibandSimHaloTask(MultibandSimBaseTask):
    _DefaultName = "MultibandSimHaloTask"
    ConfigClass = MultibandSimHaloTaskConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, MultibandSimHaloTaskConfig)

    def prepare_galaxy_catalog(
        self,
        *,
        catalog,
    ):
        assert isinstance(self.config, MultibandSimHaloTaskConfig)
        galaxy_catalog = super().prepare_galaxy_catalog(
            catalog=catalog,
        )
        # for fix source redshift
        galaxy_catalog.input_catalog["redshift"] = self.config.z_source
        return galaxy_catalog

    def get_perturbation_object(self, **kwargs: Any):
        assert isinstance(self.config, MultibandSimHaloTaskConfig)
        return ShearHalo(
            mass=self.config.mass,
            conc=self.config.conc,
            z_lens=self.config.z_lens,
            ra_lens=self.config.ra_lens,
            dec_lens=self.config.dec_lens,
            no_kappa=self.config.no_kappa,
        )
