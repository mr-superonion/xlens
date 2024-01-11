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
import gc
import glob
import json
import os
from configparser import ConfigParser, ExtendedInterpolation
from copy import deepcopy

import fitsio
import numpy as np
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.shear import ShearRedshift
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.surveys import DEFAULT_SURVEY_BANDS, get_survey

band_list = ["g", "r", "i", "z"]
nband = len(band_list)


class SimulateBase(object):
    def __init__(self, cparser):
        # image directory
        self.root_dir = cparser.get("simulation", "root_dir")
        self.sim_name = cparser.get("simulation", "sim_name")
        self.img_dir = os.path.join(self.root_dir, self.sim_name)

        # catalog directory
        cat_dir = cparser.get("simulation", "cat_dir")
        self.cat_dir = os.path.join(self.root_dir, cat_dir)

        # summary directory
        sum_dir = cparser.get("simulation", "sum_dir")
        self.sum_dir = os.path.join(self.root_dir, sum_dir)
        return


class SimulateImage(SimulateBase):
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir, exist_ok=True)

        # layout of the simulation (random, random_disk, or hex)
        self.layout = cparser.get("simulation", "layout")
        # Whether do rotation or dithering
        self.rotate = cparser.getboolean("simulation", "rotate")
        self.dither = cparser.getboolean("simulation", "dither")
        # version of the PSF simulation: 0 -- fixed PSF
        self.psf_variation = cparser.getfloat(
            "simulation",
            "psf_variation",
            fallback=0.0,
        )
        # length of the exposure
        self.coadd_dim = cparser.getint("simulation", "coadd_dim")
        # buffer length to avoid galaxies hitting the boundary of the exposure
        self.buff = cparser.getint("simulation", "buff")
        # number of rotations for ring test
        self.nrot = cparser.getint("simulation", "nrot")
        # number of redshiftbins
        self.rot_list = [np.pi / self.nrot * i for i in range(self.nrot)]
        self.shear_mode_list = json.loads(cparser.get("simulation", "shear_mode_list"))
        self.shear_component = cparser.get(
            "simulation",
            "shear_component",
            fallback="g1",
        )
        self.z_bounds = json.loads(cparser.get("simulation", "z_bounds"))
        self.nshear = len(self.shear_mode_list)
        self.shear_value = cparser.getfloat("simulation", "shear_value")
        self.survey_name = cparser.get(
            "simulation",
            "survey_name",
            fallback="LSST",
        )
        self.psf_fwhm = cparser.getfloat(
            "simulation",
            "psf_fwhm",
            fallback=None,
        )
        if self.psf_fwhm is None:
            if self.survey_name == "HSC":
                self.psf_fwhm = 0.6  # arcsec
            elif self.survey_name == "LSST":
                self.psf_fwhm = 0.8  # arcsec
            elif self.survey_name == "Euclid":
                # larger than Euclid, make sure over sample
                self.psf_fwhm = 0.19  # arcsec
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
        self.calib_mag_zero = 30.0

        # Stars
        self.draw_stars = cparser.getboolean(
            "simulation",
            "draw_stars",
            fallback=False,
        )
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

        # Systematics
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
        return

    def simulate_image(self, ifield):
        print("Simulating for field: %d" % ifield)
        rng = np.random.RandomState(ifield)

        scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.survey_name],
            survey_name=self.survey_name,
        ).pixel_scale

        kargs = {
            "cosmic_rays": self.cosmic_rays,
            "bad_columns": self.bad_columns,
            "star_bleeds": self.star_bleeds,
            "draw_method": "auto",
        }
        psf = make_fixed_psf(
            psf_type="moffat",
            psf_fwhm=self.psf_fwhm,
        ).shear(e1=self.psf_e1, e2=self.psf_e2)
        psf_fname = "%s/PSF_%s_32.fits" % (self.root_dir, self.sim_name)
        if not os.path.isfile(psf_fname):
            psf_data = psf.shift(0.5 * scale, 0.5 * scale).drawImage(nx=64, ny=64, scale=scale).array
            fitsio.write(psf_fname, psf_data)

        nfiles = len(glob.glob("%s/image-%05d_g1-*" % (self.img_dir, ifield)))
        if nfiles == self.nrot * self.nshear * nband:
            print("We aleady have all the images for this subfield.")
            return

        # galaxy catalog; you can make your own
        galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=self.coadd_dim,
            buff=self.buff,
            pixel_scale=scale,
            layout=self.layout,
        )
        print("Simulation has galaxies: %d" % len(galaxy_catalog))
        for shear_mode in self.shear_mode_list:
            for irot in range(self.nrot):
                shear_obj = ShearRedshift(
                    z_bounds=self.z_bounds,
                    mode=shear_mode,
                    g_dist=self.shear_component,
                    shear_value=self.shear_value,
                )
                sim_data = make_sim(
                    rng=rng,
                    galaxy_catalog=galaxy_catalog,
                    star_catalog=None,
                    coadd_dim=self.coadd_dim,
                    shear_obj=shear_obj,
                    psf=psf,
                    draw_gals=True,
                    draw_stars=self.draw_stars,
                    draw_bright=self.draw_bright,
                    dither=self.dither,
                    rotate=self.rotate,
                    bands=band_list,
                    noise_factor=0.0,
                    theta0=self.rot_list[irot],
                    calib_mag_zero=self.calib_mag_zero,
                    survey_name=self.survey_name,
                    **kargs,
                )
                # write galaxy images
                for band_name in band_list:
                    gal_fname = "%s/image-%05d_g1-%d_rot%d_%s.fits" % (
                        self.img_dir,
                        ifield,
                        shear_mode,
                        irot,
                        band_name,
                    )
                    mi = sim_data["band_data"][band_name][0].getMaskedImage()
                    gdata = mi.getImage().getArray()
                    fitsio.write(gal_fname, gdata)
                    del mi, gdata, gal_fname
                del sim_data
                gc.collect()
        del galaxy_catalog, psf
        return

    def run(self, ifield):
        self.simulate_image(ifield)
