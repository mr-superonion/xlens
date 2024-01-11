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
import gc
import os
from argparse import ArgumentParser
from configparser import ConfigParser

import fitsio
import impt
import jax
import numpy as np
from impt.fpfs.future import prepare_func_e

from ..simulator import SimulateBase


class NeffSimFPFS(SimulateBase):
    def __init__(
        self,
        config_name,
        min_id=0,
        max_id=1000,
        ncores=1,
    ):
        cparser = ConfigParser()
        cparser.read(config_name)
        super().__init__(cparser)
        if not os.path.isdir(self.cat_dir):
            raise FileNotFoundError("Cannot find image directory")

        # simulation parameter
        nids = max_id - min_id
        self.n_per_c = nids // ncores
        self.mid = nids % ncores
        self.min_id = min_id
        self.max_id = max_id
        self.rest_list = list(np.arange(ncores * self.n_per_c, nids) + min_id)
        print("number of files per core is: %d" % self.n_per_c)

        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")
        self.cov_mat = fitsio.read(self.ncov_fname)

        # configuration of FPFS
        self.ratio = cparser.getfloat("FPFS", "ratio")
        self.c0 = cparser.getfloat("FPFS", "c0")
        self.c2 = cparser.getfloat("FPFS", "c2")
        self.alpha = cparser.getfloat("FPFS", "alpha")
        self.beta = cparser.getfloat("FPFS", "beta")
        return

    def get_range(self, icore):
        ibeg = self.min_id + icore * self.n_per_c
        iend = min(ibeg + self.n_per_c, self.max_id)
        id_range = list(range(ibeg, iend))
        if icore < len(self.rest_list):
            id_range.append(self.rest_list[icore])
        return id_range

    def get_sum_e_r(self, in_nm, e1, enoise, res1, rnoise):
        if not os.path.isfile(in_nm):
            print("Cannot find input galaxy shear catalogs : %s " % (in_nm))
        mm = impt.fpfs.read_catalog(in_nm)

        # noise bias
        def fune(carry, ss):
            y = e1._obs_func(ss) - enoise._obs_func(ss)
            return carry + y, y

        def funr(carry, ss):
            y = res1._obs_func(ss) - rnoise._obs_func(ss)
            return carry + y, y

        e1_sum, _ = jax.lax.scan(fune, 0.0, mm)
        r1_sum, _ = jax.lax.scan(funr, 0.0, mm)
        del mm
        gc.collect()
        return e1_sum, r1_sum

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.empty((len(id_range), 2))
        # print("start core: %d, with id: %s" % (icore, id_range))
        for icount, ifield in enumerate(id_range):
            e1, enoise, res1, rnoise = prepare_func_e(
                cov_mat=self.cov_mat,
                ratio=self.ratio,
                c0=self.c0,
                c2=self.c2,
                alpha=self.alpha,
                beta=self.beta,
                g_comp=1,
            )
            in_nm1 = os.path.join(
                self.cat_dir,
                "src-%05d_g1-1_rot0.fits" % (ifield),
            )
            ell, e_r = self.get_sum_e_r(in_nm1, e1, enoise, res1, rnoise)
            out[icount, 0] = ell
            out[icount, 1] = e_r
            del e1, enoise, res1, rnoise, ell, e_r
            gc.collect()
        return out
