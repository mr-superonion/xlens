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
import os
from copy import deepcopy
from typing import Any

import anacal
import fitsio
import lsst.afw.image as afw_image
import numpy as np
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.psfs import make_dm_psf, make_fixed_psf, make_ps_psf
from descwl_shear_sims.shear import ShearRedshift
from descwl_shear_sims.sim import get_se_dim, make_sim
from descwl_shear_sims.stars import StarCatalog
from descwl_shear_sims.surveys import DEFAULT_SURVEY_BANDS, get_survey
from descwl_shear_sims.wcs import make_dm_wcs
from lsst.pex.config import (
    Config,
    DictField,
    Field,
    FieldValidationError,
    ListField,
)
from numpy.typing import NDArray

from .base import SimBaseTask

"""
Reference noise level for magnitude zero poitn 30
"LSST": {
    "g": 0.315,
    "r": 0.371,
    "i": 0.595,
    "z": 1.155,
}
"HSC": {
    "i": 0.937,
}
"""


class SimpleSimBaseConfig(Config):
    root_dir = Field[str](
        doc="root directory",
        default="try1",
    )
    bands = ListField[str](
        doc="Image bands to simulate",
        default=["r"],
    )
    mag_zero = Field[float](
        doc="Magnitude zero point",
        default=30.0,
    )
    layout = Field[str](
        doc="Layout of the galaxy distribution",
        default="random_disk",
    )
    stellar_density = Field[float](
        doc="Number density of stellar contamination [arcmin^-2]",
        optional=True,
        default=-1.0,
    )
    coadd_dim = Field[int](
        doc="Dims of the simulated coadd images",
        default=500,
    )
    buff = Field[int](
        doc="Buffer length to avoid galaxies hitting the exposure boundary",
        default=32,
    )
    survey_name = Field[str](
        doc="Survey name (LSST, HSC etc)",
        default="LSST",
    )
    psf_fwhm = Field[float](
        doc="PSF seeing size in arcsec",
        default=0.8,
    )
    psf_e1 = Field[float](
        doc="PSF ellipticity (first component), e1~2g1",
        default=0.0,
    )
    psf_e2 = Field[float](
        doc="PSF ellipticity (second component), e2~2g2",
        default=0.0,
    )
    psf_variation = Field[float](
        doc="variation of PSF field across the image",
        default=0.0,
    )
    rotate = Field[bool](
        doc="Whether do rotation",
        default=False,
    )
    dither = Field[bool](
        doc="Whether do dithering",
        default=False,
    )
    cosmic_rays = Field[bool](
        doc="whether to include cosmic ray",
        default=False,
    )
    bad_columns = Field[bool](
        doc="whether to include bad columns",
        default=False,
    )
    star_bleeds = Field[bool](
        doc="whether to include star bleeds",
        default=False,
    )
    draw_bright = Field[bool](
        doc="whether to draw bright stars",
        default=False,
    )
    draw_image_noise = Field[bool](
        doc="Whether to draw image noise",
        default=True,
    )
    noise_stds = DictField(
        doc="A dictionary of standard deviation of image noise",
        keytype=str,
        itemtype=float,
        default={
            "g": 0.315,
            "r": 0.371,
            "i": 0.595,
            "z": 1.155,
        },
    )
    nrot = Field[int](
        doc="number of rotations",
        default=2,
    )
    test_target = Field[str](
        doc="the shear component to test",
        default="g1",
    )
    test_value = Field[float](
        doc="absolute value of the shear",
        default=0.02,
    )
    mode_list = ListField[int](
        doc="the modes of the shear",
        default=[0, 1],
    )

    def validate(self):
        super().validate()
        if not set(self.bands).issubset(self.noise_stds.keys()):
            raise FieldValidationError(
                self.__class__.bands, self, "band list is not supported"
            )

        if self.psf_variation < 0.0:
            raise FieldValidationError(
                self.__class__.psf_variation,
                self,
                "psf_variation should be >=0",
            )

    def setDefaults(self):
        super().setDefaults()
        if isinstance(self.stellar_density, float):
            if self.stellar_density < -1e-3:
                self.stellar_density = None
        if not os.path.isdir(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

        # input truth directory
        src_dir = os.path.join(self.root_dir, "input_src")
        if not os.path.isdir(src_dir):
            os.makedirs(src_dir, exist_ok=True)

        # image directory
        img_dir = os.path.join(self.root_dir, "image")
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir, exist_ok=True)


class SimpleSimBaseTask(SimBaseTask):
    _DefaultName = "SimpleSimBaseTask"
    ConfigClass = SimpleSimBaseConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, SimpleSimBaseConfig)
        nrot = self.config.nrot
        # number of redshiftbins
        self.rotate_list = [np.pi / nrot * i for i in range(nrot)]
        pass

    def get_psf_object(self, rng, scale):
        assert isinstance(self.config, SimpleSimBaseConfig)
        if self.config.psf_variation < 1e-5:
            psf_obj = make_fixed_psf(
                psf_type="moffat",
                psf_fwhm=self.config.psf_fwhm,
            ).shear(e1=self.config.psf_e1, e2=self.config.psf_e2)
        else:
            se_dim = get_se_dim(
                coadd_scale=scale,
                coadd_dim=self.config.coadd_dim,
                se_scale=scale,
                rotate=self.config.rotate,
                dither=self.config.dither,
            )
            psf_obj = make_ps_psf(
                rng=rng, dim=se_dim, variation_factor=self.config.psf_variation
            )
        return psf_obj

    def get_truth_src_name(self, ifield: int, mode: int, irot: int) -> str:
        assert isinstance(self.config, SimpleSimBaseConfig)
        outdir = os.path.join(self.config.root_dir, "input_src")
        return "%s/src-%05d_%s-%d_rot%d.fits" % (
            outdir,
            ifield,
            self.config.test_target,
            mode,
            irot,
        )

    def get_image_name(
        self, ifield: int, mode: int, irot: int, band: str
    ) -> str:
        assert isinstance(self.config, SimpleSimBaseConfig)
        outdir = os.path.join(self.config.root_dir, "image")
        return "%s/image-%05d_%s-%d_rot%d_%s.fits" % (
            outdir,
            ifield,
            self.config.test_target,
            mode,
            irot,
            band,
        )

    def write_truth(
        self, *, ifield: int, mode: int, irot: int, data: NDArray
    ) -> None:
        assert isinstance(self.config, SimpleSimBaseConfig)
        name = self.get_truth_src_name(ifield, mode, irot)
        fitsio.write(name, data)

    def write_image(
        self,
        *,
        ifield: int,
        mode: int,
        irot: int,
        band: str,
        data: NDArray,
    ) -> None:
        assert isinstance(self.config, SimpleSimBaseConfig)
        name = self.get_image_name(ifield, mode, irot, band)
        fitsio.write(name, data)

    def get_random_seed(
        self,
        *,
        ifield: int,
        irot: int,
        band: str | None = None,
    ) -> int:
        """This function returns the random seed for image noise simulation.
        It makes sure that different modes have the same seed. But different
        rotated version, different bands have different seeds.
        """
        assert isinstance(self.config, SimpleSimBaseConfig)
        nbands = len(self.config.noise_stds.keys())
        nbands2 = nbands + 1  # this incldes the one with all bands
        if band is None:
            # The case includes all bands
            band_id = 0
        else:
            band_id = list(self.config.noise_stds.keys()).index(band) + 1
        return (ifield * self.config.nrot + irot) * nbands2 + band_id

    def get_noise_corr(self) -> NDArray | None:
        """This function get the image noise correlation function"""
        assert isinstance(self.config, SimpleSimBaseConfig)
        noise_corr_file_name = os.path.join(
            self.config.root_dir,
            "noise_correlation.fits",
        )
        if os.path.isfile(noise_corr_file_name):
            noise_corr = fitsio.read(noise_corr_file_name)
        else:
            noise_corr = None
        return noise_corr

    def get_noise_array(
        self,
        *,
        seed,
        noise_std,
        noise_corr,
        shape,
        pixel_scale,
    ) -> NDArray | float:
        assert isinstance(self.config, SimpleSimBaseConfig)
        if self.config.draw_image_noise:
            if noise_corr is None:
                noise_array = np.random.RandomState(seed).normal(
                    scale=noise_std,
                    size=shape,
                )
            else:
                noise_array = (
                    anacal.noise.simulate_noise(
                        seed=seed,
                        correlation=noise_corr,
                        nx=shape[1],
                        ny=shape[0],
                        scale=pixel_scale,
                    )
                    * noise_std
                )
            self.log.debug("Simulated noise STD is: %.2f" % np.std(noise_array))
        else:
            noise_array = 0.0
        return noise_array

    def get_star_sim_info(self, rng, pixel_scale, psf_obj, band_list):
        assert isinstance(self.config, SimpleSimBaseConfig)
        star_kwargs = {
            "coadd_dim": self.config.coadd_dim,
            "dither": self.config.dither,
            "rotate": self.config.rotate,
            "calib_mag_zero": self.config.mag_zero,
            "survey_name": self.config.survey_name,
            "cosmic_rays": self.config.cosmic_rays,
            "bad_columns": self.config.bad_columns,
            "star_bleeds": self.config.star_bleeds,
            "draw_bright": self.config.draw_bright,
            "draw_method": "auto",
            "noise_factor": 1.0,
            "g1": 0.0,
            "g2": 0.0,
            "draw_gals": False,
            "draw_noise": False,
        }
        if "random" in self.config.layout:
            star_catalog = StarCatalog(
                rng=rng,
                coadd_dim=self.config.coadd_dim,
                buff=self.config.buff,
                density=self.config.stellar_density,
                min_density=2.0,
                max_density=100.0,
                layout=self.config.layout,
            )
            draw_stars = True
        else:
            star_catalog = None
            draw_stars = False

        empty_galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=self.config.coadd_dim,
            pixel_scale=pixel_scale,
            buff=self.config.buff,
            layout=self.config.layout,
            select_observable="r_ab",
            select_upper_limit=0,  # make an empty galaxy catalog
        )
        return make_sim(
            rng=rng,
            galaxy_catalog=empty_galaxy_catalog,
            star_catalog=star_catalog,
            psf=psf_obj,
            draw_stars=draw_stars,
            bands=band_list,
            **star_kwargs,
        )

    def get_truth_src(self, ifield, mode, irot):
        name = self.get_truth_src_name(ifield, mode, irot)
        if os.path.isfile(name):
            return fitsio.read(name)
        else:
            raise FileNotFoundError("Cannot find input truth source catalog.")

    def get_dm_exposure(
        self,
        *,
        ifield: int,
        mode: int,
        irot: int,
        band_list: list[str],
    ):
        assert isinstance(self.config, SimpleSimBaseConfig)
        self.log.debug(
            "Preparing the exposure for (ifield: %d, mode: %d irot: %d )"
            % (ifield, mode, irot)
        )
        rng = np.random.RandomState(ifield)  # for PSF and stars
        pixel_scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.config.survey_name],
            survey_name=self.config.survey_name,
        ).pixel_scale
        psf_obj = self.get_psf_object(rng, pixel_scale)
        star_outcome = self.get_star_sim_info(
            rng,
            pixel_scale,
            psf_obj,
            band_list,
        )

        noise_corr = self.get_noise_corr()
        variance = 0.0
        weight_sum = 0.0
        ny = self.config.coadd_dim + 10
        nx = self.config.coadd_dim + 10
        shape = (ny, nx)
        gal_array = np.zeros(shape)
        msk_array = np.zeros(shape, dtype=int)
        band = None
        for i, band in enumerate(band_list):
            self.log.debug("reading %s band" % band)
            seed = self.get_random_seed(
                ifield=ifield,
                irot=irot,
                band=band,
            )
            self.log.debug("The random seed is %d" % seed)
            image_name = self.get_image_name(ifield, mode, irot, band)
            # Add noise
            noise_std = self.config.noise_stds[band]
            weight = 1.0 / (self.config.noise_stds[band]) ** 2.0
            variance += (noise_std * weight) ** 2.0
            self.log.debug("Using noisy setup with std: %.2f" % noise_std)
            noise_array = self.get_noise_array(
                seed=seed,
                noise_std=noise_std,
                noise_corr=noise_corr,
                shape=shape,
                pixel_scale=pixel_scale,
            )
            star_array = (
                star_outcome["band_data"][band][0].getMaskedImage().image.array
            )
            msk_array = msk_array | (
                star_outcome["band_data"][band][0].getMaskedImage().mask.array
            )
            gal_array = (
                gal_array
                + (fitsio.read(image_name) + star_array + noise_array) * weight
            )
            weight_sum += weight

        masked_image = afw_image.MaskedImageF(ny, nx)
        masked_image.image.array[:, :] = gal_array / weight_sum
        masked_image.variance.array[:, :] = variance / (weight_sum) ** 2.0
        std_final = np.sqrt(variance / (weight_sum) ** 2.0)
        self.log.debug("The final noise variance is %.2f" % std_final)
        masked_image.mask.array[:, :] = msk_array
        exp = afw_image.ExposureF(masked_image)

        zero_flux = 10.0 ** (0.4 * self.config.mag_zero)
        photo_calib = afw_image.makePhotoCalibFromCalibZeroPoint(zero_flux)
        exp.setPhotoCalib(photo_calib)
        psf_dim = star_outcome["psf_dims"][0]
        se_wcs = star_outcome["se_wcs"][band][0]
        dm_psf = make_dm_psf(
            psf=psf_obj,
            psf_dim=psf_dim,
            wcs=deepcopy(se_wcs),
        )
        exp.setPsf(dm_psf)
        dm_wcs = make_dm_wcs(se_wcs)
        exp.setWcs(dm_wcs)
        return exp

    def simulate_images(
        self,
        *,
        rng,
        galaxy_catalog,
        shear_obj,
        psf_obj,
        ifield,
        irot,
        mode,
    ):
        assert isinstance(self.config, SimpleSimBaseConfig)
        galaxy_kwargs = {
            "cosmic_rays": False,
            "bad_columns": False,
            "star_bleeds": False,
            "draw_method": "auto",
            "noise_factor": 0.0,
            "draw_gals": True,
            "draw_stars": False,
            "draw_bright": False,
            "star_catalog": None,
        }
        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=self.config.coadd_dim,
            shear_obj=shear_obj,
            psf=psf_obj,
            dither=self.config.dither,
            rotate=self.config.rotate,
            bands=self.config.bands,
            theta0=self.rotate_list[irot],
            calib_mag_zero=self.config.mag_zero,
            survey_name=self.config.survey_name,
            **galaxy_kwargs,
        )
        self.write_truth(
            ifield=ifield,
            mode=mode,
            irot=irot,
            data=sim_data["truth_info"],
        )
        # write galaxy images
        for _bn in self.config.bands:
            mi = sim_data["band_data"][_bn][0].getMaskedImage()
            gdata = mi.getImage().getArray()
            self.write_image(
                ifield=ifield,
                mode=mode,
                irot=irot,
                band=_bn,
                data=gdata,
            )
        return

    def run(self, ifield: int):
        assert isinstance(self.config, SimpleSimBaseConfig)
        rng = np.random.RandomState(ifield)
        scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.config.survey_name],
            survey_name=self.config.survey_name,
        ).pixel_scale
        psf_obj = self.get_psf_object(rng, scale)
        # galaxy catalog; you can make your own
        galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=self.config.coadd_dim,
            buff=self.config.buff,
            pixel_scale=scale,
            layout=self.config.layout,
        )

        for mode in self.config.mode_list:
            shear_obj = self.get_perturbation_object(
                mode=mode,
            )
            for irot in range(self.config.nrot):
                self.simulate_images(
                    rng=rng,
                    galaxy_catalog=galaxy_catalog,
                    shear_obj=shear_obj,
                    psf_obj=psf_obj,
                    ifield=ifield,
                    irot=irot,
                    mode=mode,
                )
        return

    def get_sim_id_list(self, min_id, max_id):
        """Generate indices for simulations
        Args:
        min_id (int):   minimum id
        max_id (int):   maximum id

        Returns:
        out (list):     a list of simulation identifications
        """
        assert isinstance(self.config, SimpleSimBaseConfig)
        out = [
            [fid, mode, irot]
            for fid in range(min_id, max_id)
            for mode in self.config.mode_list
            for irot in range(self.config.nrot)
        ]
        return out

    def clear(self):
        import shutil

        assert isinstance(self.config, SimpleSimBaseConfig)
        if os.path.isdir(self.config.root_dir):
            shutil.rmtree(self.config.root_dir)
        return


class SimpleSimShearConfig(SimpleSimBaseConfig):
    z_bounds = ListField[float](
        doc="boundary list of the redshift",
        default=[-0.01, 20.0],
    )
    mode_list = ListField[int](
        doc="the modes of the shear",
        default=[0, 1],
    )

    def validate(self):
        super().validate()
        n_zbins = len(self.z_bounds) - 1
        mode_max = 3 ** (n_zbins)
        if np.max(self.mode_list) >= mode_max:
            raise FieldValidationError(
                self.__class__.mode_list,
                self,
                "mode needs to be smaller than %d" % mode_max,
            )

        if self.test_target not in ["g1", "g2"]:
            raise FieldValidationError(
                self.__class__.test_target,
                self,
                "test target can only be 'g1' or 'g2'",
            )

        if self.test_value < 0.0 or self.test_value > 0.08:
            raise FieldValidationError(
                self.__class__.test_value,
                self,
                "test_value should be in [0.00, 0.08]",
            )

    def setDefaults(self):
        super().setDefaults()


class SimpleSimShearTask(SimpleSimBaseTask):
    _DefaultName = "SimpleSimShearTask"
    ConfigClass = SimpleSimShearConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, SimpleSimShearConfig)
        self.perturbation_kwargs = {
            "g_dist": self.config.test_target,
            "shear_value": self.config.test_value,
            "z_bounds": self.config.z_bounds,
        }

    def get_perturbation_object(self, mode: int, **kwargs: Any):
        return ShearRedshift(
            mode=mode,
            g_dist=self.perturbation_kwargs["g_dist"],
            shear_value=self.perturbation_kwargs["shear_value"],
            z_bounds=self.perturbation_kwargs["z_bounds"],
        )
