#!/usr/bin/env python
#
# FPFS shear estimator
# Copyright 20220312 Xiangchong Li.
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
import glob
import os
import time
from configparser import ConfigParser, ExtendedInterpolation

import fitsio
import jax
import numpy as np
from fpfs.catalog import fpfs_catalog, read_catalog

from ..simulator import SimulateBatchBase


class SummarySimFpfs(SimulateBatchBase):
    def __init__(
        self,
        config_name,
        min_id=0,
        max_id=1000,
        ncores=1,
    ):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser, min_id, max_id, ncores)
        if not os.path.isdir(self.cat_dir):
            raise FileNotFoundError("Cannot find image directory")
        if not os.path.isdir(self.sum_dir):
            os.makedirs(self.sum_dir, exist_ok=True)

        # FPFS parameters
        self.radial_n = cparser.getint("FPFS", "radial_n", fallback=2)
        self.nord = cparser.getint("FPFS", "nord", fallback=4)
        assert self.radial_n in [2, 4]
        if self.radial_n == 2:
            assert self.nord >= 4
        if self.radial_n == 4:
            assert self.nord >= 6
        self.ncov_fname = cparser.get(
            "FPFS",
            "ncov_fname",
            fallback="",
        )
        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")
        self.cov_mat = fitsio.read(self.ncov_fname)

        self.pthres = cparser.getfloat("FPFS", "pthres", fallback=0.0)
        self.pratio = cparser.getfloat("FPFS", "pratio", fallback=0.02)

        self.ratio = cparser.getfloat("FPFS", "ratio")
        self.c0 = cparser.getfloat("FPFS", "c0")
        self.c2 = cparser.getfloat("FPFS", "c2")
        self.alpha = cparser.getfloat("FPFS", "alpha")
        self.beta = cparser.getfloat("FPFS", "beta")
        upper_mag = cparser.getfloat("FPFS", "magcut", fallback=27.5)
        self.lower_m00 = 10 ** ((self.calib_mag_zero - upper_mag) / 2.5)
        self.noise_rev = cparser.getboolean("FPFS", "noise_rev", fallback=True)

        # shear setup
        self.shear_value = cparser.getfloat("simulation", "shear_value")
        self.g_comp_sim = cparser.get(
            "simulation",
            "shear_component",
            fallback="g1",
        )
        self.g_comp_measure = cparser.getint(
            "FPFS",
            "g_component_measure",
            fallback=1,
        )
        assert self.g_comp_measure in [1, 2], "The g_comp_measure in configure file is not supported"

        self.ofname = os.path.join(
            self.sum_dir,
            "bin_%s.fits" % (upper_mag),
        )
        return

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.zeros((len(id_range), 4))
        cat_obj = fpfs_catalog(
            cov_mat=self.cov_mat,
            snr_min=self.lower_m00 / np.sqrt(self.cov_mat[0, 0]),
            ratio=self.ratio,
            c0=self.c0,
            c2=self.c2,
            alpha=self.alpha,
            beta=self.beta,
            pthres=self.pthres,
            pratio=self.pratio,
        )
        if self.noise_rev:
            if self.g_comp_measure == 1:
                func = jax.jit(cat_obj.measure_g1_noise_correct)
            elif self.g_comp_measure == 2:
                func = jax.jit(cat_obj.measure_g2_noise_correct)
            else:
                raise ValueError("g_comp_measure should be 1 or 2")
        else:
            if self.g_comp_measure == 1:
                func = jax.jit(cat_obj.measure_g1)
            elif self.g_comp_measure == 2:
                func = jax.jit(cat_obj.measure_g2)
            else:
                raise ValueError("g_comp_meausre should be 1 or 2")
        print("start core: %d, with id: %s" % (icore, id_range))
        start_time = time.time()
        for icount, ifield in enumerate(id_range):
            for irot in range(self.nrot):
                in_nm1 = os.path.join(
                    self.cat_dir,
                    "src-%05d_%s-0_rot%d_%s.fits" % (ifield, self.g_comp_sim, irot, self.bands),
                )
                e1_1, r1_1 = self.get_sum_e_r(in_nm1, func, read_catalog)
                in_nm2 = os.path.join(
                    self.cat_dir,
                    "src-%05d_%s-1_rot%d_%s.fits" % (ifield, self.g_comp_sim, irot, self.bands),
                )
                e1_2, r1_2 = self.get_sum_e_r(in_nm2, func, read_catalog)
                out[icount, 0] = ifield
                out[icount, 1] = out[icount, 1] + (e1_2 - e1_1)
                out[icount, 2] = out[icount, 2] + (e1_1 + e1_2) / 2.0
                out[icount, 3] = out[icount, 3] + (r1_1 + r1_2) / 2.0
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 4.0
        print("elapsed time: %.2f seconds" % elapsed_time)
        return out

    def get_sum_e_r(self, in_nm, func, read_func):
        assert os.path.isfile(in_nm), "Cannot find input galaxy shear catalogs : %s " % (in_nm)
        mm = read_func(in_nm)
        e1_sum, r1_sum = jax.numpy.sum(jax.lax.map(func, mm), axis=0)
        return e1_sum, r1_sum

    def display_result(self):
        flist = glob.glob("%s/bin_*.*.fits" % (self.sum_dir))
        for fname in flist:
            mag = fname.split("/")[-1].split("bin_")[-1].split(".fits")[0]
            print("magnitude is: %s" % mag)
            a = fitsio.read(fname)
            a = a[np.argsort(a[:, 0])]
            nsim = a.shape[0]
            b = np.average(a, axis=0)
            print(
                "multiplicative bias:",
                b[1] / b[3] / self.shear_value / 2.0 - 1,
            )
            print(
                "1-sigma error:",
                np.std(a[:, 1] / a[:, 3]) / self.shear_value / 2.0 / np.sqrt(nsim),
            )
            print("additive bias:", b[2] / b[3])
            print(
                "1-sigma error:",
                np.std(a[:, 2] / a[:, 3]) / np.sqrt(nsim),
            )
        return
