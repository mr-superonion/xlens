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
import shutil
from configparser import ConfigParser, ExtendedInterpolation
from copy import deepcopy

import fitsio
import numpy as np
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf
from descwl_shear_sims.shear import ShearRedshift
from descwl_shear_sims.sim import get_se_dim, make_sim
from descwl_shear_sims.surveys import DEFAULT_SURVEY_BANDS, get_survey

from .perturbation import ShearHalo, ShearKappa

band_list = ["g", "r", "i", "z"]
# band_list = ["i"]
nband = len(band_list)

_band_map = {
    "g": 0,
    "r": 1,
    "i": 2,
    "z": 3,
    "a": 4,
}

default_config = {
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

# all the standard deviations are normalized to magnitude zero point 30
DEFAULT_ZERO_POINT = 30.0
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


class SimulateBase(object):
    def __init__(self, cparser):
        # image directory
        self.root_dir = cparser.get("simulation", "root_dir")
        self.sim_name = cparser.get("simulation", "sim_name")
        self.img_dir = os.path.join(self.root_dir, self.sim_name)

        # catalog directory
        cat_dir = cparser.get("simulation", "cat_dir", fallback=None)
        if cat_dir is not None:
            self.cat_dir = os.path.join(self.root_dir, cat_dir)
            cat_dm_dir = cparser.get("simulation", "cat_dm_dir", fallback=None)
            if cat_dm_dir is None and cat_dir is not None:
                cat_dm_dir = cat_dir.replace("cat", "cat_dm")
            self.cat_dm_dir = os.path.join(self.root_dir, cat_dm_dir)
        else:
            self.cat_dir = None
            self.cat_dm_dir = None

        input_cat_dir = cparser.get(
            "simulation",
            "input_cat_dir",
            fallback=None,
        )
        if input_cat_dir is not None:
            self.input_cat_dir = os.path.join(self.root_dir, input_cat_dir)
        else:
            self.input_cat_dir = None

        # summary directory
        sum_dir = cparser.get("simulation", "sum_dir", fallback=None)
        if sum_dir is not None:
            self.sum_dir = os.path.join(self.root_dir, sum_dir)

        # number of rotations for ring test
        self.nrot = cparser.getint("simulation", "nrot")
        # number of redshiftbins
        self.rot_list = [np.pi / self.nrot * i for i in range(self.nrot)]

        # bands used for measurement
        self.bands = cparser.get("simulation", "band")

        # magnitude zero point
        self.calib_mag_zero = DEFAULT_ZERO_POINT

        # layout of the simulation (random, random_disk, or hex)
        self.layout = cparser.get("simulation", "layout")
        # Whether do rotation or dithering
        self.rotate = cparser.getboolean("simulation", "rotate")
        self.dither = cparser.getboolean("simulation", "dither")
        # size of the exposure image
        self.coadd_dim = cparser.getint("simulation", "coadd_dim")
        # buffer length to avoid galaxies hitting the boundary of the exposure
        self.buff = cparser.getint("simulation", "buff")
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
        # version of the PSF simulation: 0 -- fixed PSF
        self.psf_variation = cparser.getfloat(
            "simulation",
            "psf_variation",
            fallback=0.0,
        )
        assert self.psf_variation >= 0.0

        self.shear_comp_sim = cparser.get(
            "simulation",
            "shear_component",
            fallback="g1",
        )
        self.shear_mode_list = json.loads(
            cparser.get("simulation", "shear_mode_list"),
        )
        self.nshear = len(self.shear_mode_list)

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

        # Stars
        self.stellar_density = cparser.getfloat(
            "simulation",
            "stellar_density",
            fallback=-1.0,
        )
        if self.stellar_density < -0.01:
            self.stellar_density = None
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

        # Noise
        noise_std = cparser.getfloat(
            "simulation",
            "noise_std",
            fallback=None,
        )
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
        self.corr_fname = cparser.get(
            "simulation",
            "noise_corr_fname",
            fallback=None,
        )
        return

    def get_sim_fnames(self, min_id, max_id, field_only=False):
        """Generate filename for simulations
        Args:
            ftype (str):    file type ('src' for gal, and 'image' for exposure
            min_id (int):   minimum id
            max_id (int):   maximum id
            field_only (bool): only include filed number
        Returns:
            out (list):     a list of file name
        """
        if field_only:
            out = [
                os.path.join(
                    self.img_dir,
                    "image-%05d_xxx.fits" % (fid),
                )
                for fid in range(min_id, max_id)
            ]
        else:
            out = [
                os.path.join(
                    self.img_dir,
                    "image-%05d_g1-%d_rot%d_xxx.fits" % (fid, gid, rid),
                )
                for fid in range(min_id, max_id)
                for gid in self.shear_mode_list
                for rid in range(self.nrot)
            ]
        return out

    def get_psf_obj(self, rng, scale):
        if self.psf_variation < 1e-5:
            psf_obj = make_fixed_psf(
                psf_type="moffat",
                psf_fwhm=self.psf_fwhm,
            ).shear(e1=self.psf_e1, e2=self.psf_e2)
        else:
            se_dim = get_se_dim(
                coadd_scale=scale,
                coadd_dim=self.coadd_dim,
                se_scale=scale,
                rotate=self.rotate,
                dither=self.dither,
            )
            psf_obj = make_ps_psf(
                rng=rng, dim=se_dim, variation_factor=self.psf_variation
            )
        return psf_obj

    def get_seed_from_fname(self, fname, band):
        """This function returns the random seed for image noise simulation.
        It makes sure that different sheared versions have the same seed.
        But different rotated version, different bands have different seeds.
        """
        # field id
        fid = int(fname.split("image-")[-1].split("_")[0]) + 212
        # rotation id
        rid = int(fname.split("rot")[1][0])
        # band id
        bm = deepcopy(_band_map)
        bid = bm[band]
        _nbands = len(bm.values())
        return ((fid * self.nrot + rid) * _nbands + bid) * 3

    def clear_image(self):
        if os.path.isdir(self.img_dir):
            shutil.rmtree(self.img_dir)
        return

    def clear_catalog(self):
        if os.path.isdir(self.cat_dir):
            shutil.rmtree(self.cat_dir)

        if os.path.isdir(self.cat_dm_dir):
            shutil.rmtree(self.cat_dm_dir)
        return

    def clear_summary(self):
        if os.path.isdir(self.sum_dir):
            shutil.rmtree(self.sum_dir)
        return

    def clear_all(self):
        if hasattr(self, "img_dir"):
            self.clear_image()
        if hasattr(self, "cat_dir"):
            self.clear_catalog()
        if hasattr(self, "sum_dir"):
            self.clear_summary()
        return

    def write_image(self, exposure, filename):
        fitsio.write(filename, exposure.getMaskedImage().image.array)
        return

    def write_ds9_region(self, xy, filename):
        """
        Write a list of (x, y) positions to a DS9 region file.

        Args:
        xy (list of tuples): List of (x, y) positions.
        filename (str): Name of the file to save the regions.
        """
        header = "# Region file format: DS9 version 4.1\n"
        header += "global color=green dashlist=8 3 width=1"
        header += "font='helvetica 10 normal roman' "
        header += "select=1 highlite=1 dash=0 fixed=0 edit=1"
        header += "move=1 delete=1 include=1 source=1\n"
        header += "image\n"  # Coordinate system, using image pixels
        with open(filename, "w") as file:
            file.write(header)
            for x, y in xy:
                file.write("point(%d,%d) # point=circle\n" % (x + 1, y + 1))
        return


class SimulateBatchBase(SimulateBase):
    def __init__(
        self,
        cparser,
        min_id,
        max_id,
        ncores,
    ):
        super().__init__(cparser)
        # simulation parameter
        nids = max_id - min_id
        self.n_per_c = nids // ncores
        self.mid = nids % ncores
        self.min_id = min_id
        self.max_id = max_id
        self.rest_list = list(np.arange(ncores * self.n_per_c, nids) + min_id)
        print("number of files per core is: %d" % self.n_per_c)
        return

    def get_range(self, icore):
        ibeg = self.min_id + icore * self.n_per_c
        iend = min(ibeg + self.n_per_c, self.max_id)
        id_range = list(range(ibeg, iend))
        if icore < len(self.rest_list):
            id_range.append(self.rest_list[icore])
        return id_range


class SimulateImage(SimulateBase):
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir, exist_ok=True)

        self.z_bounds = json.loads(cparser.get("simulation", "z_bounds"))
        self.shear_value = cparser.getfloat("simulation", "shear_value")
        return

    def run(self, ifield):
        print("Simulating for field: %d" % ifield)
        rng = np.random.RandomState(ifield)
        scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.survey_name],
            survey_name=self.survey_name,
        ).pixel_scale
        psf_obj = self.get_psf_obj(rng, scale)

        kargs = deepcopy(default_config)

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
        bl = deepcopy(band_list)
        print("Simulation has galaxies: %d" % len(galaxy_catalog))
        for shear_mode in self.shear_mode_list:
            shear_obj = ShearRedshift(
                z_bounds=self.z_bounds,
                mode=shear_mode,
                g_dist=self.shear_comp_sim,
                shear_value=self.shear_value,
            )
            for irot in range(self.nrot):
                sim_data = make_sim(
                    rng=rng,
                    galaxy_catalog=galaxy_catalog,
                    coadd_dim=self.coadd_dim,
                    shear_obj=shear_obj,
                    psf=psf_obj,
                    dither=self.dither,
                    rotate=self.rotate,
                    bands=bl,
                    theta0=self.rot_list[irot],
                    calib_mag_zero=self.calib_mag_zero,
                    survey_name=self.survey_name,
                    **kargs,
                )
                # write galaxy images
                for band_name in bl:
                    gal_fname = "%s/image-%05d_%s-%d_rot%d_%s.fits" % (
                        self.img_dir,
                        ifield,
                        self.shear_comp_sim,
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
        del galaxy_catalog
        return


class SimulateImageKappa(SimulateBase):
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir, exist_ok=True)

        self.shear_value = cparser.getfloat("simulation", "shear_value")
        self.kappa = cparser.getfloat("simulation", "kappa")
        return

    def run(self, ifield):
        print("Simulating for field: %d" % ifield)
        rng = np.random.RandomState(ifield)

        scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.survey_name],
            survey_name=self.survey_name,
        ).pixel_scale
        psf_obj = self.get_psf_obj(rng, scale)

        kargs = deepcopy(default_config)

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
        bl = deepcopy(band_list)
        print("Simulation has galaxies: %d" % len(galaxy_catalog))
        for shear_mode in self.shear_mode_list:
            shear_obj = ShearKappa(
                mode=shear_mode,
                g_dist=self.shear_comp_sim,
                shear_value=self.shear_value,
                kappa=self.kappa,
            )
            for irot in range(self.nrot):
                sim_data = make_sim(
                    rng=rng,
                    galaxy_catalog=galaxy_catalog,
                    coadd_dim=self.coadd_dim,
                    shear_obj=shear_obj,
                    psf=psf_obj,
                    dither=self.dither,
                    rotate=self.rotate,
                    bands=bl,
                    theta0=self.rot_list[irot],
                    calib_mag_zero=self.calib_mag_zero,
                    survey_name=self.survey_name,
                    **kargs,
                )
                # write galaxy images
                for band_name in bl:
                    gal_fname = "%s/image-%05d_%s-%d_rot%d_%s.fits" % (
                        self.img_dir,
                        ifield,
                        self.shear_comp_sim,
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
        del galaxy_catalog
        return


class SimulateImageHalo(SimulateBase):
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir, exist_ok=True)
        return

    def run(self, ifield):
        print("Simulating for field: %d" % ifield)
        rng = np.random.RandomState(ifield)

        scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.survey_name],
            survey_name=self.survey_name,
        ).pixel_scale
        psf_obj = self.get_psf_obj(rng, scale)

        kargs = deepcopy(default_config)

        # galaxy catalog; you can make your own
        galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=self.coadd_dim,
            buff=self.buff,
            pixel_scale=scale,
            layout=self.layout,
        )
        bl = deepcopy(band_list)
        print("Simulation has galaxies: %d" % len(galaxy_catalog))
        for shear_mode in self.shear_mode_list:
            par = [4e14, 6.0, 0.2]
            shear_obj = ShearHalo(
                mass=par[0],
                conc=par[1],
                z_lens=par[2],
            )
            for irot in range(self.nrot):
                sim_data = make_sim(
                    rng=rng,
                    galaxy_catalog=galaxy_catalog,
                    coadd_dim=self.coadd_dim,
                    shear_obj=shear_obj,
                    psf=psf_obj,
                    dither=self.dither,
                    rotate=self.rotate,
                    bands=bl,
                    theta0=self.rot_list[irot],
                    calib_mag_zero=self.calib_mag_zero,
                    survey_name=self.survey_name,
                    **kargs,
                )
                # write galaxy images
                for band_name in bl:
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
        del galaxy_catalog
        return
