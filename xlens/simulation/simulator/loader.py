#!/usr/bin/env python
#
# Copyright 20221013 Xiangchong Li.
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
import pickle
from configparser import ConfigParser
from copy import deepcopy

import anacal
import fitsio
import lsst.afw.image as afw_image
import lsst.geom as lsst_geom
import numpy as np
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.psfs import make_dm_psf, make_fixed_psf
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.stars import StarCatalog
from descwl_shear_sims.surveys import DEFAULT_SURVEY_BANDS, get_survey
from descwl_shear_sims.wcs import make_dm_wcs

from .base import SimulateBase


class MakeDMExposure(SimulateBase):
    def __init__(
        self,
        config_name,
    ):
        """A Class to load DM exposures

        Args:
        config_name (str):              configuration file name
        """
        cparser = ConfigParser()
        cparser.read(config_name)
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find image directory")
        self.kargs = {
            "cosmic_rays": self.cosmic_rays,
            "bad_columns": self.bad_columns,
            "star_bleeds": self.star_bleeds,
            "draw_bright": self.draw_bright,
            "draw_method": "auto",
            "noise_factor": self.noise_ratio,
            "g1": 0.0,
            "g2": 0.0,
            "draw_gals": False,
            "draw_noise": False,
        }
        if self.bands != "a":
            self.blist = [b for b in self.bands]
        else:
            self.blist = ["g", "r", "i", "z"]
        self.deep_noise_frac = cparser.get(
            "FPFS",
            "deep_noise_frac",
            fallback=0.1
        )
        return

    def get_sim_info(self, rng, pixel_scale, psf_obj):
        if "random" in self.layout:
            star_catalog = StarCatalog(
                rng=rng,
                coadd_dim=self.coadd_dim,
                buff=self.buff,
                density=self.stellar_density,
                min_density=2.0,
                max_density=100.0,
                layout=self.layout,
            )
            draw_stars = True
        else:
            # isolated sims are not interesting, tested many times..
            star_catalog = None
            draw_stars = False
        galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=self.coadd_dim,
            pixel_scale=pixel_scale,
            buff=self.buff,
            layout=self.layout,
            select_observable="r_ab",
            select_upper_limit=0,  # make an empty galaxy catalog
        )
        if self.bands != "a":
            self.blist = [b for b in self.bands]
        else:
            self.blist = ["g", "r", "i", "z"]
        return make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            star_catalog=star_catalog,
            coadd_dim=self.coadd_dim,
            psf=psf_obj,
            draw_stars=draw_stars,
            dither=self.dither,
            rotate=self.rotate,
            bands=self.blist,
            calib_mag_zero=self.calib_mag_zero,
            survey_name=self.survey_name,
            **self.kargs,
        )

    def generate_exposure(self, file_name):
        field_id = int(file_name.split("image-")[-1].split("_")[0])
        rng = np.random.RandomState(field_id)
        pixel_scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.survey_name],
            survey_name=self.survey_name,
        ).pixel_scale
        if "wide" in file_name:
            print('hit wide')
            psf_obj = make_fixed_psf(
                    psf_type="moffat",
                    psf_fwhm=0.7,
                ).shear(e1=self.psf_e1, e2=self.psf_e2)
        elif "deep" in file_name:
            print('hit deep')
            psf_obj = make_fixed_psf(
                    psf_type="moffat",
                    psf_fwhm=0.5,
                ).shear(e1=self.psf_e1, e2=self.psf_e2)
        else:
            psf_obj = make_fixed_psf(
                    psf_type="moffat",
                    psf_fwhm=0.7,
                ).shear(e1=self.psf_e1, e2=self.psf_e2)
        if self.input_cat_dir is None:
            star_outcome = self.get_sim_info(rng, pixel_scale, psf_obj)
        else:
            assert os.path.isdir(self.input_cat_dir)
            tmp_fname = "info-%05d.p" % field_id
            tmp_fname = os.path.join(self.input_cat_dir, tmp_fname)
            star_outcome = pickle.load(open(tmp_fname, "rb"))

        if self.bands != "a":
            bands = self.bands
        else:
            bands = "aaaa"

        if self.corr_fname is None:
            noise_corr = None
        else:
            noise_corr = fitsio.read(self.corr_fname)
        variance = 0.0
        weight_sum = 0.0
        ny = self.coadd_dim + 10
        nx = self.coadd_dim + 10
        gal_array = np.zeros((ny, nx))
        msk_array = np.zeros((ny, nx), dtype=int)
        for i, band in enumerate(self.blist):
            print("reading %s band" % band)
            # Add noise
            nstd_f = self.base_std[band] * self.noise_ratio
            if "deep" in file_name:
                nstd_f *= np.sqrt(self.deep_noise_frac)
                weight = 1.0 / (self.base_std[band] * np.sqrt(self.deep_noise_frac)) ** 2.0
            weight = 1.0 / (self.base_std[band]) ** 2.0
            variance += (nstd_f * weight) ** 2.0
            print("Using noisy setup with std: %.2f" % nstd_f)
            if nstd_f > 1e-4:
                seed = self.get_seed_from_fname(file_name, band)
                print("The random seed is %d" % seed)
                if noise_corr is None:
                    noise_array = np.random.RandomState(seed).normal(
                        scale=nstd_f,
                        size=(ny, nx),
                    )
                else:
                    noise_array = (
                        anacal.noise.simulate_noise(
                            seed=seed,
                            correlation=noise_corr,
                            nx=nx,
                            ny=ny,
                            scale=pixel_scale,
                        )
                        * nstd_f
                    )
                print("Simulated noise STD is: %.2f" % np.std(noise_array))
            else:
                noise_array = 0.0
            star_array = (
                star_outcome["band_data"][band][0].getMaskedImage().image.array
            )
            msk_array = msk_array | (
                star_outcome["band_data"][band][0].getMaskedImage().mask.array
            )
            gal_array = (
                gal_array
                + (
                    fitsio.read(file_name.replace("_xxx", "_%s" % bands[i]))
                    + star_array
                    + noise_array
                )
                * weight
            )
            weight_sum += weight
            del noise_array
        masked_image = afw_image.MaskedImageF(ny, nx)
        masked_image.image.array[:, :] = gal_array / weight_sum
        masked_image.variance.array[:, :] = variance / (weight_sum) ** 2.0
        masked_image.mask.array[:, :] = msk_array
        exp = afw_image.ExposureF(masked_image)

        zero_flux = 10.0 ** (0.4 * self.calib_mag_zero)
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

    def run(self, file_name):
        return self.generate_exposure(file_name)


class MakeBrightInfo(MakeDMExposure):
    def __init__(
        self,
        config_name,
    ):
        """A Class to load DM exposures

        Args:
        config_name (str):              configuration file name
        """
        super().__init__(config_name)
        assert "random" in self.layout
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find image directory")
        assert self.input_cat_dir is not None
        if not os.path.isdir(self.input_cat_dir):
            os.makedirs(self.input_cat_dir, exist_ok=True)
        assert self.draw_bright
        if self.stellar_density is not None:
            assert self.stellar_density > 1e-5
        return

    def make_bright_catalog(self, file_name):
        field_id = int(file_name.split("image-")[-1].split("_")[0])

        rng = np.random.RandomState(field_id)
        pixel_scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.survey_name],
            survey_name=self.survey_name,
        ).pixel_scale
        psf_obj = self.get_psf_obj(rng, pixel_scale)
        star_outcome = self.get_sim_info(rng, pixel_scale, psf_obj)

        wcs = star_outcome["band_data"][self.blist[0]][0].wcs
        binfo = star_outcome["bright_info"]
        nn = len(binfo)
        plist = wcs.skyToPixel(
            [
                lsst_geom.SpherePoint(
                    ss["ra"],
                    ss["dec"],
                    lsst_geom.degrees,
                )
                for ss in binfo
            ]
        )
        out = np.array(
            [
                (
                    plist[i].getX(),
                    plist[i].getY(),
                    binfo[i]["radius_pixels"],
                    binfo[i]["has_bleed"],
                )
                for i in range(nn)
            ],
            dtype=[("x", "f4"), ("y", "f4"), ("r", "f4"), ("has_bleed", "?")],
        )
        return out, star_outcome

    def run(self, file_name):
        assert self.input_cat_dir is not None
        src_name = os.path.join(
            self.input_cat_dir,
            file_name.split("/")[-1],
        )
        src_name = src_name.replace(
            "image-",
            "brightstar-",
        ).replace(
            "_xxx",
            "",
        )
        if os.path.isfile(src_name):
            print("Already has output for simulation: %s." % src_name)
            return
        star_cat, star_outcome = self.make_bright_catalog(file_name)
        fitsio.write(src_name, star_cat)

        pname = os.path.join(
            self.input_cat_dir,
            file_name.split("/")[-1],
        )
        pname = (
            pname.replace(
                "image-",
                "info-",
            )
            .replace(
                "_xxx",
                "",
            )
            .replace(".fits", ".p")
        )
        pickle.dump(star_outcome, open(pname, "wb"))
        return

class MakeDMExposureSV(SimulateBase):
    def __init__(
        self,
        config_name,
        cat_i,
    ):
        """A Class to load DM exposures

        Args:
        config_name (str):              configuration file name
        """
        cparser = ConfigParser()
        cparser.read(config_name)
        super().__init__(cparser)
        self.img_dir = self.img_dir.replace("xxx", str(cat_i))
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find image directory")
        self.kargs = {
            "cosmic_rays": self.cosmic_rays,
            "bad_columns": self.bad_columns,
            "star_bleeds": self.star_bleeds,
            "draw_bright": self.draw_bright,
            "draw_method": "auto",
            "noise_factor": self.noise_ratio,
            "g1": 0.0,
            "g2": 0.0,
            "draw_gals": False,
            "draw_noise": False,
        }
        if self.bands != "a":
            self.blist = [b for b in self.bands]
        else:
            self.blist = ["g", "r", "i", "z"]
        self.deep_noise_frac = cparser.get(
            "FPFS",
            "deep_noise_frac",
            fallback=0.1
        )
        return

    def get_sim_info(self, rng, pixel_scale, psf_obj):
        if "random" in self.layout:
            star_catalog = StarCatalog(
                rng=rng,
                coadd_dim=self.coadd_dim,
                buff=self.buff,
                density=self.stellar_density,
                min_density=2.0,
                max_density=100.0,
                layout=self.layout,
            )
            draw_stars = True
        else:
            # isolated sims are not interesting, tested many times..
            star_catalog = None
            draw_stars = False
        galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=self.coadd_dim,
            pixel_scale=pixel_scale,
            buff=self.buff,
            layout=self.layout,
            select_observable="r_ab",
            select_upper_limit=0,  # make an empty galaxy catalog
        )
        if self.bands != "a":
            self.blist = [b for b in self.bands]
        else:
            self.blist = ["g", "r", "i", "z"]
        return make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            star_catalog=star_catalog,
            coadd_dim=self.coadd_dim,
            psf=psf_obj,
            draw_stars=draw_stars,
            dither=self.dither,
            rotate=self.rotate,
            bands=self.blist,
            calib_mag_zero=self.calib_mag_zero,
            survey_name=self.survey_name,
            **self.kargs,
        )

    def generate_exposure(self, file_name):
        field_id = int(file_name.split("image-")[-1].split("_")[0])
        rng = np.random.RandomState(field_id)
        pixel_scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.survey_name],
            survey_name=self.survey_name,
        ).pixel_scale

        psf_obj = make_fixed_psf(
                psf_type="moffat",
                psf_fwhm=0.7,
            ).shear(e1=self.psf_e1, e2=self.psf_e2)
        
        if self.input_cat_dir is None:
            star_outcome = self.get_sim_info(rng, pixel_scale, psf_obj)
        else:
            assert os.path.isdir(self.input_cat_dir)
            tmp_fname = "info-%05d.p" % field_id
            tmp_fname = os.path.join(self.input_cat_dir, tmp_fname)
            star_outcome = pickle.load(open(tmp_fname, "rb"))

        if self.bands != "a":
            bands = self.bands
        else:
            bands = self.blist

        if self.corr_fname is None:
            noise_corr = None
        else:
            noise_corr = fitsio.read(self.corr_fname)
        variance = 0.0
        weight_sum = 0.0
        ny = self.coadd_dim + 10
        nx = self.coadd_dim + 10
        gal_array = np.zeros((ny, nx))
        msk_array = np.zeros((ny, nx), dtype=int)
        for i, band in enumerate(self.blist):
            print("reading %s band" % band)
            # Add noise
            nstd_f = self.base_std[band] * self.noise_ratio
            if "deep" in file_name or "deg" in file_name:
                nstd_f *= np.sqrt(self.deep_noise_frac)
                weight = 1.0 / (self.base_std[band] * np.sqrt(self.deep_noise_frac)) ** 2.0
            weight = 1.0 / (self.base_std[band]) ** 2.0
            variance += (nstd_f * weight) ** 2.0
            print("Using noisy setup with std: %.2f" % nstd_f)
            if nstd_f > 1e-4:
                seed = self.get_seed_from_fname(file_name, band)
                print("The random seed is %d" % seed)
                if noise_corr is None:
                    noise_array = np.random.RandomState(seed).normal(
                        scale=nstd_f,
                        size=(ny, nx),
                    )
                else:
                    noise_array = (
                        anacal.noise.simulate_noise(
                            seed=seed,
                            correlation=noise_corr,
                            nx=nx,
                            ny=ny,
                            scale=pixel_scale,
                        )
                        * nstd_f
                    )
                print("Simulated noise STD is: %.2f" % np.std(noise_array))
            else:
                noise_array = 0.0
            star_array = (
                star_outcome["band_data"][band][0].getMaskedImage().image.array
            )
            msk_array = msk_array | (
                star_outcome["band_data"][band][0].getMaskedImage().mask.array
            )
            gal_array = (
                gal_array
                + (
                    fitsio.read(file_name.replace("_xxx", "_%s" % bands[i]))
                    + star_array
                    + noise_array
                )
                * weight
            )
            weight_sum += weight
            del noise_array
        masked_image = afw_image.MaskedImageF(ny, nx)
        masked_image.image.array[:, :] = gal_array / weight_sum
        masked_image.variance.array[:, :] = variance / (weight_sum) ** 2.0
        masked_image.mask.array[:, :] = msk_array
        exp = afw_image.ExposureF(masked_image)

        zero_flux = 10.0 ** (0.4 * self.calib_mag_zero)
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

    def run(self, file_name):
        return self.generate_exposure(file_name)
