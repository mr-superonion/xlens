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
from configparser import ConfigParser
from copy import deepcopy

import fitsio
import lsst.afw.image as afw_image
import numpy as np
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.psfs import make_dm_psf, make_fixed_psf
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.stars import StarCatalog
from descwl_shear_sims.surveys import DEFAULT_SURVEY_BANDS, get_survey
from descwl_shear_sims.wcs import make_dm_wcs

from .base import SimulateBase

_band_map = {
    "g": 0,
    "r": 1,
    "i": 2,
    "z": 3,
    "a": 4,
}

# all the standard deviations are normalized to magnitude zero point 30
_nstd_map = {
    "LSST": {
        "g": 0.315,
        "r": 0.371,
        "i": 0.595,
        "z": 1.155,
    },
    "HSC": {
        "g": 0.964,
        "r": 0.964,
        "i": 0.964,
        "z": 0.964,
    },
}


class MakeDMExposure(SimulateBase):
    def __init__(
        self,
        config_name,
        noise_std_overwrite=None,
        bands_overwrite=None,
    ):
        """A Class to load DM exposures

        Args:
        config_name (str):              configuration file name
        noise_std_overwrite (float):    overwrite noise standard deviation
        bands_overwrite (float):        overwrite bands with this value
        """
        cparser = ConfigParser()
        cparser.read(config_name)
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find image directory")
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir, exist_ok=True)

        self.load_configure(cparser)
        if bands_overwrite is not None:
            self.bands = bands_overwrite

        # Systematics
        self.draw_bright = cparser.getboolean(
            "simulation",
            "draw_bright",
            fallback=False,
        )
        self.star_bleeds = cparser.getboolean(
            "simulation",
            "star_bleeds",
            fallback=False,
        )
        self.cosmic_rays = cparser.getboolean(
            "simulation",
            "cosmic_rays",
            fallback=False,
        )
        self.bad_columns = cparser.getboolean(
            "simulation",
            "bad_columns",
            fallback=False,
        )

        noise_std = cparser.getfloat(
            "simulation",
            "noise_std",
            fallback=None,
        )
        if noise_std_overwrite is not None:
            noise_std = noise_std_overwrite
        if noise_std is None:
            self.base_std = deepcopy(_nstd_map)[self.survey_name]
        else:
            dd = deepcopy(_nstd_map)[self.survey_name]
            self.base_std = {key: noise_std for key in dd.keys()}
        self.noise_ratio = cparser.getfloat(
            "simulation",
            "noise_ratio",
            fallback=0.0,
        )
        return

    def load_configure(self, cparser):
        # number of rotation of galaxies (positions and shapes)
        self.nrot = cparser.getint("simulation", "nrot", fallback=2)
        # whehter rotate single exposure or not
        self.rotate = cparser.getboolean("simulation", "rotate", fallback=False)
        # whehter do the dithering
        self.dither = cparser.getboolean("simulation", "dither", fallback=False)
        self.coadd_dim = cparser.getint("simulation", "coadd_dim")
        # buffer length to avoid galaxies hitting the boundary of the exposure
        self.buff = cparser.getint("simulation", "buff")

        self.stellar_density = cparser.getfloat(
            "simulation",
            "stellar_density",
            fallback=0.0,
        )
        self.layout = cparser.get("simulation", "layout")

        self.psf_fwhm = cparser.getfloat(
            "simulation",
            "psf_fwhm",
            fallback=None,
        )
        self.survey_name = cparser.get(
            "simulation",
            "survey_name",
            fallback="LSST",
        )
        print("Simulating survey: %s" % self.survey_name)
        self.psf_e1 = cparser.getfloat(
            "simulation",
            "psf_e1",
            fallback=0.0,
        )
        self.psf_e2 = cparser.getfloat(
            "simulation",
            "psf_e2",
            fallback=0.0,
        )
        return

    def get_seed_from_fname(self, fname, band):
        """This function returns the random seed for simulation.
        It makes sure that different sheared versions have the same seed
        """
        # field id
        fid = int(fname.split("image-")[-1].split("_")[0]) + 212
        # rotation id
        rid = int(fname.split("rot")[1][0])
        # band id
        bid = deepcopy(_band_map)[band]
        _nbands = len(_band_map.values())
        return ((fid * self.nrot + rid) * _nbands + bid) * 3

    def generate_exposure(self, fname):
        field_id = int(fname.split("image-")[-1].split("_")[0]) + 212
        rng = np.random.RandomState(field_id)
        if "random" in self.layout:
            star_catalog = StarCatalog(
                rng=rng,
                coadd_dim=self.coadd_dim,
                buff=self.buff,
                density=self.stellar_density,
                layout=self.layout,
            )
            draw_stars = True
        else:
            star_catalog = None
            draw_stars = False

        scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.survey_name],
            survey_name=self.survey_name,
        ).pixel_scale
        galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=self.coadd_dim,
            pixel_scale=scale,
            buff=self.buff,
            layout=self.layout,
            density=0,
        )
        psf_obj = make_fixed_psf(
            psf_type="moffat",
            psf_fwhm=self.psf_fwhm,
        ).shear(e1=self.psf_e1, e2=self.psf_e2)
        kargs = {
            "cosmic_rays": self.cosmic_rays,
            "bad_columns": self.bad_columns,
            "star_bleeds": self.star_bleeds,
            "draw_bright": self.draw_bright,
            "draw_method": "auto",
        }
        if self.bands != "a":
            blist = [b for b in self.bands]
            bands = self.bands
        else:
            blist = ["g", "r", "i", "z"]
            bands = "aaaa"
        star_outcome = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            star_catalog=star_catalog,
            coadd_dim=self.coadd_dim,
            psf=psf_obj,
            draw_gals=False,
            draw_stars=draw_stars,
            dither=self.dither,
            rotate=self.rotate,
            bands=blist,
            noise_factor=0.0,
            calib_mag_zero=self.calib_mag_zero,
            survey_name=self.survey_name,
            g1=0.0,
            g2=0.0,
            **kargs,
        )
        variance = 0.0
        weight_sum = 0.0
        ny = self.coadd_dim + 10
        nx = self.coadd_dim + 10
        gal_array = np.zeros((ny, nx))
        msk_array = np.zeros((ny, nx), dtype=int)
        for i, band in enumerate(blist):
            print("reading %s band" % band)
            # Add noise
            nstd_f = self.base_std[band] * self.noise_ratio
            weight = 1.0 / (self.base_std[band]) ** 2.0
            variance += (nstd_f * weight) ** 2.0
            seed = self.get_seed_from_fname(fname, band)
            rng2 = np.random.RandomState(seed)
            print("Using noisy setup with std: %.2f" % nstd_f)
            print("The random seed is %d" % seed)
            star_array = star_outcome["band_data"][band][0].getMaskedImage().image.array
            msk_array = msk_array & (star_outcome["band_data"][band][0].getMaskedImage().mask.array)
            gal_array = (
                gal_array
                + (
                    fitsio.read(fname.replace("_xxx", "_%s" % bands[i]))
                    + star_array
                    + rng2.normal(
                        scale=nstd_f,
                        size=(ny, nx),
                    )
                )
                * weight
            )
            weight_sum += weight
        masked_image = afw_image.MaskedImageF(ny, nx)
        masked_image.image.array[:, :] = gal_array / weight_sum
        masked_image.variance.array[:, :] = variance / (weight_sum) ** 2.0
        masked_image.mask.array[:, :] = msk_array
        exp = afw_image.ExposureF(masked_image)

        zero_flux = 10.0 ** (0.4 * self.calib_mag_zero)
        photo_calib = afw_image.makePhotoCalibFromCalibZeroPoint(zero_flux)
        exp.setPhotoCalib(photo_calib)
        psf_dim = star_outcome["psf_dims"][0]
        se_wcs = star_outcome["se_wcs"][0]
        dm_psf = make_dm_psf(psf=psf_obj, psf_dim=psf_dim, wcs=deepcopy(se_wcs))
        exp.setPsf(dm_psf)
        dm_wcs = make_dm_wcs(se_wcs)
        exp.setWcs(dm_wcs)
        return exp

    def run(self, fname):
        return self.generate_exposure(fname)
