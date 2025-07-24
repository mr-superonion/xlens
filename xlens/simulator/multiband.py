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
import lsst.meas.algorithms as meaAlg
import numpy as np
from .perturbation import ShearHalo, ShearRedshift
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.wcs import make_dm_wcs
from lsst.pex.config import Config, Field, FieldValidationError, ListField
from lsst.pipe.base import Struct
from numpy.typing import NDArray

from ..utils.image import resize_array
from ..utils.random import (
    gal_seed_base,
    get_noise_seed,
    image_noise_base,
    num_rot,
)
from .base import SimBaseTask
from .galaxies.catsim import CatSim2017Catalog
from .galaxies.skyCatalog import OpenUniverse2024RubinRomanCatalog
from .multiband_defaults import (
    mag_zero_defaults,
    noise_variance_defaults,
    psf_fwhm_defaults,
    sys_npix,
)


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


class MultibandSimBaseConfig(Config):
    survey_name = Field[str](
        doc="Name of the survey",
        default="LSST",
    )
    layout = Field[str](
        doc="Layout of the galaxy distribution (random, grid, hex)",
        default="random",
    )
    galaxy_type = Field[str](
        doc="galaxy type",
        default="catsim2017",
    )
    extend_ratio = Field[float](
        doc="The ratio to extend for the size of simulated image",
        default=1.06,
    )
    sep = Field[float](
        doc="separation distance (arcsec) between galaxies (for grid and hex)",
        default=11.0,
    )
    order_truth_catalog = Field[bool](
        doc="Whether to keep the order in truth catalog",
        default=False,
    )
    include_pixel_masks = Field[bool](
        doc="whether to include pixel masks in the simulation",
        default=False,
    )
    include_stars = Field[bool](
        doc="whether to include stars in the simulation",
        default=False,
    )
    draw_image_noise = Field[bool](
        doc="Whether to draw image noise in the simulation",
        default=False,
    )
    galId = Field[int](
        doc="random seed index, 0 <= galId < 10",
        default=0,
    )
    rotId = Field[int](
        doc="number of rotations",
        default=0,
    )
    noiseId = Field[int](
        doc="random seed for noise, 0 <= noiseId < 10",
        default=0,
    )
    use_real_psf = Field[bool](
        doc="whether to use real PSF",
        default=False,
    )
    force_pixel_center = Field[bool](
        doc="whether to force galaxies at pixel center",
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
        if self.noiseId < 0 or self.noiseId >= image_noise_base // 2:
            raise FieldValidationError(
                self.__class__.noiseId,
                self,
                "We require 0 <= noiseId < %d" % (image_noise_base // 2),
            )
        if self.layout not in ["grid", "hex", "random"]:
            raise FieldValidationError(
                self.__class__.layout,
                self,
                "We require layout in ['grid', 'hex', 'random']",
            )
        if self.sep <= 0:
            raise FieldValidationError(
                self.__class__.sep,
                self,
                "We require sep > 0.0 arcsec",
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
        rng: np.random.RandomState,
        dim: int,
        pixel_scale: float,
        sep: float = 11.0,
        indice_id=None,
        **kwargs,
    ):
        assert isinstance(self.config, MultibandSimBaseConfig)
        # prepare galaxy catalog
        coadd_dim = dim - 10
        # galaxy catalog;
        if self.config.galaxy_type == "catsim2017":
            GalClass = CatSim2017Catalog
        elif self.config.galaxy_type == "RomanRubin2024":
            GalClass = OpenUniverse2024RubinRomanCatalog
        else:
            raise ValueError("invalid galaxy_type")
        galaxy_catalog = GalClass(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=0.0,
            pixel_scale=pixel_scale,
            layout=self.config.layout,
            simple_coadd_bbox=True,
            sep=sep,
            indice_id=indice_id,
        )
        if self.config.force_pixel_center:
            galaxy_catalog.shifts_array["dx"] = (
                np.floor(galaxy_catalog.shifts_array["dx"] / pixel_scale)
                * pixel_scale
                + pixel_scale / 2.0
            )
            galaxy_catalog.shifts_array["dy"] = (
                np.floor(galaxy_catalog.shifts_array["dy"] / pixel_scale)
                * pixel_scale
                + pixel_scale / 2.0
            )
        return galaxy_catalog

    def simulate_images(
        self,
        *,
        rng: np.random.RandomState,
        galaxy_catalog,
        shear_obj,
        psf_obj,
        rotId: int,
        band: str,
        coadd_dim: int,
        mag_zero: float,
        draw_method: str = "auto",
        **kwargs,
    ):
        assert isinstance(self.config, MultibandSimBaseConfig)
        galaxy_kwargs = {
            "survey_name": self.config.survey_name.upper(),
            "star_catalog": None,
            "dither": False,
            "rotate": False,
            "cosmic_rays": False,
            "bad_columns": False,
            "star_bleeds": False,
            "noise_factor": 0.0,
            "draw_gals": True,
            "draw_stars": False,
            "draw_bright": False,
            "draw_noise": False,
        }

        res = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            shear_obj=shear_obj,
            psf=psf_obj,
            theta0=self.rotate_list[rotId],
            bands=[band],
            coadd_dim=coadd_dim,
            calib_mag_zero=mag_zero,
            simple_coadd_bbox=True,
            draw_method=draw_method,
            **galaxy_kwargs,
        )

        # write galaxy images
        image = res["band_data"][band][0].getMaskedImage().image.array
        truth = res["truth_info"]
        se_wcs = res["se_wcs"][band][0]
        del res
        dm_wcs = make_dm_wcs(se_wcs)
        return image, truth, dm_wcs

    def run(
        self,
        *,
        band: str,
        seed: int,
        boundaryBox,
        wcs: lsst.afw.geom.SkyWcs,
        psfImage: afwImage.ImageF | None = None,
        noiseCorrImage: afwImage.ImageF | None = None,
        exposure: afwImage.ExposureF | None = None,
        patch: int = 0,
        **kwargs,
    ):
        assert isinstance(self.config, MultibandSimBaseConfig)
        if self.config.use_real_psf:
            if psfImage is None:
                raise IOError("Do not have PSF input model")
        if self.config.order_truth_catalog and self.config.layout in [
            "grid",
            "hex",
        ]:
            indice_id = patch
        else:
            indice_id = None

        # Prepare the random number generator and basic parameters
        rotId = self.config.rotId
        survey_name = self.config.survey_name
        seed_gal = seed * gal_seed_base + self.config.galId
        rng = np.random.RandomState(seed_gal)

        # Get the pixel scale in arcseconds per pixel
        pixel_scale = wcs.getPixelScale().asArcseconds()
        width = boundaryBox.getWidth()
        height = boundaryBox.getHeight()
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

        dim = int(max(width, height) * self.config.extend_ratio)
        galaxy_catalog = self.prepare_galaxy_catalog(
            rng=rng,
            dim=dim,
            pixel_scale=pixel_scale,
            sep=self.config.sep,
            indice_id=indice_id,
        )
        if dim % 2 == 1:
            dim = dim + 1
        coadd_dim = dim - 10
        shear_obj = self.get_perturbation_object()
        data, truth_catalog, dm_wcs = self.simulate_images(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            shear_obj=shear_obj,
            psf_obj=psf_galsim,
            rotId=rotId,
            band=band,
            coadd_dim=coadd_dim,
            mag_zero=mag_zero,
            draw_method=draw_method,
        )
        self.log.debug(f"current shape of data is {data.shape}")
        self.log.debug(f"resizing data to {height} x {width}")
        data, truth_catalog = resize_array(
            data,
            (height, width),
            truth_catalog,
        )

        exp_out = afwImage.ExposureF(boundaryBox)
        exp_out.getMaskedImage().image.array[:, :] = data
        exp_out.setPhotoCalib(photo_calib)
        exp_out.setPsf(kernel_psf)
        exp_out.setWcs(dm_wcs)
        exp_out.getMaskedImage().variance.array[:, :] = variance
        del data, photo_calib, kernel_psf, dm_wcs

        if self.config.draw_image_noise:
            seed_noise = get_noise_seed(
                seed=seed,
                noiseId=self.config.noiseId,
                rotId=self.config.rotId,
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
        doc="number of rotations",
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

        if self.force_pixel_center:
            if self.mode != 2 or self.rotId != 0:
                raise FieldValidationError(
                    self.__class__.force_pixel_center,
                    self,
                    "Cannot force galaxies at pixel center",
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
        doc="source redshift",
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
        if self.force_pixel_center:
            raise FieldValidationError(
                self.__class__.force_pixel_center,
                self,
                "Cannot force galaxies at pixel center",
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
        rng: np.random.RandomState,
        dim: int,
        pixel_scale: float,
        sep: float = 11.0,
        indice_id=None,
        **kwargs,
    ):
        assert isinstance(self.config, MultibandSimHaloTaskConfig)
        galaxy_catalog = super().prepare_galaxy_catalog(
            rng=rng,
            dim=dim,
            pixel_scale=pixel_scale,
            sep=sep,
            indice_id=indice_id,
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
