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

import astropy.io.fits as pyfits
import numpy as np
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.psfs import make_fixed_psf
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.stars import StarCatalog

from .simulator import SimulateBase

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
        "a": 0.2186,
    },
    "HSC": {
        "g": 0.964,
        "r": 0.964,
        "i": 0.964,
        "z": 0.964,
        "a": 0.964,
    },
}


class MakeDMExposure(SimulateBase):
    def __init__(
        self,
        config_name,
        noise_ratio=None,
        bands=None,
    ):
        cparser = ConfigParser()
        cparser.read(config_name)
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find image directory")
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir)

        self.load_configure(cparser)
        if noise_ratio is not None:
            self.noise_ratio = noise_ratio
        if bands is not None:
            self.bands = bands
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

        psf_fwhm = cparser.getfloat(
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
        psf_e1 = cparser.getfloat(
            "simulation",
            "psf_e1",
            fallback=0.0,
        )
        psf_e2 = cparser.getfloat(
            "simulation",
            "psf_e2",
            fallback=0.0,
        )
        self.psf = make_fixed_psf(
            psf_type="moffat",
            psf_fwhm=psf_fwhm,
        ).shear(e1=psf_e1, e2=psf_e2)
        self.noise_std = deepcopy(_nstd_map[self.survey_name])
        return

    def get_sim_fname(self, min_id, max_id, nshear=2):
        """Generate filename for simulations
        Args:
            ftype (str):    file type ('src' for source, and 'image' for exposure
            min_id (int):   minimum id
            max_id (int):   maximum id
            nshear (int):   number of shear
            nrot (int):     number of rotations
        Returns:
            out (list):     a list of file name
        """
        out = [
            os.path.join(self.img_dir, "image-%05d_g1-%d_rot%d_xxx.fits" % (fid, gid, rid))
            for fid in range(min_id, max_id)
            for gid in self.shear_mode_list
            for rid in range(self.nrot)
        ]
        return out

    def get_seed_from_fname(self, fname, band):
        """This function returns the random seed for simulation.
        It makes sure that different sheared versions have the same seed
        """
        # field id
        fid = int(fname.split("image-")[-1].split("_")[0]) + 212
        # rotation id
        rid = int(fname.split("rot")[1][0])
        b_map = deepcopy(_band_map)
        # band id
        bid = b_map[band]
        _nbands = len(b_map.values())
        return (fid * self.nrot + rid) * _nbands + bid

    def generate_exposure(self, fname):
        field_id = int(fname.split("image-")[-1].split("_")[0]) + 212
        rng = np.random.RandomState(field_id)
        star_catalog = StarCatalog(
            rng=rng,
            coadd_dim=self.coadd_dim,
            buff=self.buff,
            density=self.stellar_density,
            layout=self.layout,
        )
        galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=self.coadd_dim,
            buff=self.buff,
            layout=self.layout,
            density=1,
        )
        star_outcome = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            star_catalog=star_catalog,
            coadd_dim=self.coadd_dim,
            psf=self.psf,
            draw_gals=False,
            draw_stars=True,
            draw_bright=False,
            dither=self.dither,
            rotate=self.rotate,
            bands=[b for b in self.bands],
            noise_factor=0.0,
            cosmic_rays=False,
            bad_columns=False,
            star_bleeds=False,
            draw_method="auto",
            g1=0.0,
            g2=0.0,
        )
        gal_array = None
        msk_array = None
        variance = 0.0
        weight_sum = 0.0
        for band in self.bands:
            print("reading %s band" % band)
            this_gal_array = pyfits.getdata(fname.replace("_xxx", "_%s" % band))
            if gal_array is None:
                gal_array = np.zeros_like(this_gal_array)
                msk_array = np.zeros(gal_array.shape, dtype=int)
            # Add noise
            nstd_f = self.noise_std[band] * self.noise_ratio
            weight = 1.0 / (self.noise_std[band]) ** 2.0
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
                    this_gal_array
                    + star_array
                    + rng2.normal(
                        scale=nstd_f,
                        size=this_gal_array.shape,
                    )
                )
                * weight
            )
            weight_sum += weight
        assert gal_array is not None
        exposure = star_outcome["band_data"][self.bands[0]][0]
        masked_image = exposure.getMaskedImage()
        masked_image.image.array[:, :] = gal_array / weight_sum
        masked_image.variance.array[:, :] = variance / (weight_sum) ** 2.0
        masked_image.mask.array[:, :] = msk_array
        return exposure

    def run(self, fname):
        return self.generate_exposure(fname)
