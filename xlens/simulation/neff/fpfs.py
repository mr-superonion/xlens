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
import os
import time
from configparser import ConfigParser

import fitsio
import jax
import numpy as np
from fpfs.catalog import fpfs_catalog, read_catalog

from ..simulator import SimulateBatchBase


class NeffSimFPFS(SimulateBatchBase):
    def __init__(
        self,
        config_name,
        min_id=0,
        max_id=1000,
        ncores=1,
    ):
        cparser = ConfigParser()
        cparser.read(config_name)
        super().__init__(cparser, min_id, max_id, ncores)
        if not os.path.isdir(self.cat_dir):
            raise FileNotFoundError("Cannot find image directory")

        # FPFS setup
        self.ratio = cparser.getfloat("FPFS", "ratio")
        self.c0 = cparser.getfloat("FPFS", "c0")
        self.c2 = cparser.getfloat("FPFS", "c2")
        self.alpha = cparser.getfloat("FPFS", "alpha")
        self.beta = cparser.getfloat("FPFS", "beta")
        self.thres2 = cparser.getfloat("FPFS", "thres2", fallback=0.0)
        self.snr_min = cparser.getfloat("FPFS", "snr_min", fallback=12.0)
        self.noise_rev = cparser.getboolean("FPFS", "noise_rev", fallback=True)
        self.ncov_fname = cparser.get(
            "FPFS",
            "ncov_fname",
            fallback="",
        )
        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")
        self.cov_mat = fitsio.read(self.ncov_fname)

        # shear setup
        self.g_comp_measure = cparser.getint(
            "FPFS",
            "g_comp_measureonent_measure",
            fallback=1,
        )
        assert self.g_comp_measure in [1, 2], \
            "The g_comp_measure in configure file is not supported"
        self.imode = self.shear_mode_list[-1]

        coadd_dim = cparser.getint("simulation", "coadd_dim")
        buff = cparser.getint("simulation", "buff")
        coadd_scale = cparser.getfloat("simulation", "coadd_scale", fallback=0.2)
        radius = ((coadd_dim + 10) / 2.0 - buff) * coadd_scale / 60.0
        self.area = np.pi * radius**2  # [arcmin^2]
        return

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.zeros((len(id_range), 2))
        cat_obj = fpfs_catalog(
            cov_mat=self.cov_mat,
            snr_min=self.snr_min,
            ratio=self.ratio,
            c0=self.c0,
            c2=self.c2,
            alpha=self.alpha,
            beta=self.beta,
            thres2=self.thres2,
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
            in_nm1 = os.path.join(
                self.cat_dir,
                "src-%05d_g1-%d_rot0_%s.fits" % (ifield, self.imode, self.bands),
            )
            ell, e_r = self.get_sum_e_r(in_nm1, func, read_catalog)
            out[icount, 0] = ell
            out[icount, 1] = e_r
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("elapsed time: %.2f seconds" % elapsed_time)
        return out
