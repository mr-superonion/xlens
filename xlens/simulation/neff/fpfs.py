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
import time
from configparser import ConfigParser, ExtendedInterpolation

import numpy as np
from anacal.fpfs import CatalogTask
from anacal.fpfs.table import Catalog, Covariance

from ..simulator.base import SimulateBatchBase


class NeffSimFpfs(SimulateBatchBase):
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
        assert self.cat_dir is not None
        if not os.path.isdir(self.cat_dir):
            raise FileNotFoundError(
                "Cannot find catalog directory: %s" % self.cat_dir
            )

        # FPFS parameters
        self.norder = cparser.getint("FPFS", "norder", fallback=4)
        self.det_nrot = cparser.getint("FPFS", "det_nrot", fallback=4)
        self.pthres = cparser.getfloat("FPFS", "pthres", fallback=0.12)
        self.c0 = cparser.getfloat("FPFS", "c0")
        self.snr_min = cparser.getfloat("FPFS", "snr_min", fallback=10.0)
        self.r2_min = cparser.getfloat("FPFS", "r2_min", fallback=0.05)

        self.ncov_fname = cparser.get(
            "FPFS",
            "ncov_fname",
            fallback="",
        )
        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")
        self.cov_matrix = Covariance.from_fits(self.ncov_fname)

        # shear setup
        self.shear_value = cparser.getfloat("simulation", "shear_value")

        self.ename = cparser.get("FPFS", "ename", fallback="e1")
        assert int(self.ename[-1]) > 0
        self.egname = self.ename + "_g" + self.ename[-1]

        # neff
        coadd_dim = cparser.getint("simulation", "coadd_dim")
        buff = cparser.getint("simulation", "buff")
        coadd_scale = cparser.getfloat(
            "simulation", "coadd_scale", fallback=0.2
        )
        radius = ((coadd_dim + 10) / 2.0 - buff) * coadd_scale / 60.0
        self.area = np.pi * radius**2  # [arcmin^2]
        return

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.zeros((len(id_range), 2))
        ctask = CatalogTask(
            norder=self.norder,
            det_nrot=self.det_nrot,
            cov_matrix=self.cov_matrix,
        )
        ctask.update_parameters(
            snr_min=self.snr_min,
            r2_min=self.r2_min,
            c0=self.c0,
        )
        id_range = self.get_range(icore)
        out = np.zeros((len(id_range), 2))

        print("start core: %d, with id: %s" % (icore, id_range))
        start_time = time.time()
        assert self.cat_dir is not None
        en = self.ename
        egn = self.egname
        for icount, ifield in enumerate(id_range):
            nm = os.path.join(
                self.cat_dir,
                "src_1-%05d_g1-0_rot0_%s.fits" % (ifield, self.bands),
            )

            src = Catalog.from_fits(nm)
            mom = ctask.run(catalog=src)
            out[icount, 0] = np.sum(mom[en] * mom["w"])
            out[icount, 1] = np.sum(mom[egn] * mom["w"] + mom[en] * mom["w_g1"])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("elapsed time: %.2f seconds" % elapsed_time)
        del ctask
        gc.collect()
        return out
