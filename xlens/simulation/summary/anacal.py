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
from anacal.fpfs import FpfsCatalog

from ..simulator.base import SimulateBatchBase

pf = {
    "snr_min": 1.0,
    "r2_min": 100.0,
    "r2_max": 100.0,
}


class SummarySimAnacal(SimulateBatchBase):
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
            raise FileNotFoundError("Cannot find catalog directory")

        # FPFS parameters
        self.radial_n = cparser.getint("FPFS", "radial_n", fallback=2)
        self.nord = cparser.getint("FPFS", "nord", fallback=4)
        assert self.radial_n in [2, 4]
        if self.radial_n == 2:
            assert self.nord >= 4
        if self.radial_n == 4:
            assert self.nord >= 6

        self.pthres = cparser.getfloat("FPFS", "pthres", fallback=0.8)
        self.pratio = cparser.getfloat("FPFS", "pratio", fallback=0.0)
        self.pthres2 = cparser.getfloat("FPFS", "pthres2", fallback=0.12)
        self.det_nrot = cparser.getint("FPFS", "det_nrot", fallback=4)

        self.c0 = cparser.getfloat("FPFS", "c0")
        self.c2 = cparser.getfloat("FPFS", "c2")
        self.alpha = cparser.getfloat("FPFS", "alpha")
        self.beta = cparser.getfloat("FPFS", "beta")
        self.snr_min = cparser.getfloat("FPFS", "snr_min", fallback=10.0)
        self.r2_min = cparser.getfloat("FPFS", "r2_min", fallback=0.05)
        rmax = cparser.getfloat("FPFS", "r_max", fallback=10000)
        self.rmax2 = rmax * rmax

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
        self.shear_value = cparser.getfloat("simulation", "shear_value")
        self.g_comp_measure = cparser.getint(
            "FPFS",
            "g_component_measure",
            fallback=1,
        )
        assert self.g_comp_measure in [
            1,
            2,
        ], "The g_comp_measure in configure file is not supported"

        # summary
        if not os.path.isdir(self.sum_dir):
            os.makedirs(self.sum_dir, exist_ok=True)
        self.test_obs = cparser.get("FPFS", "test_obs", fallback="snr_min")
        self.cut = getattr(self, self.test_obs)
        self.ofname = os.path.join(
            self.sum_dir,
            "bin_%s_%02d.fits"
            % (
                self.test_obs,
                int(self.cut * pf[self.test_obs]),
            ),
        )
        self.image_center = (self.coadd_dim + 10) / 2.0
        return

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.zeros((len(id_range), 4))
        cat_obj = FpfsCatalog(
            cov_mat=self.cov_mat,
            snr_min=self.snr_min,
            r2_min=self.r2_min,
            c0=self.c0,
            c2=self.c2,
            alpha=self.alpha,
            beta=self.beta,
            pthres=self.pthres,
            pratio=self.pratio,
            pthres2=self.pthres2,
            det_nrot=self.det_nrot,
        )
        print("start core: %d, with id: %s" % (icore, id_range))
        start_time = time.time()
        for icount, ifield in enumerate(id_range):
            for irot in range(self.nrot):
                in_nm1 = os.path.join(
                    self.cat_dir,
                    "src-%05d_%s-0_rot%d_%s.fits"
                    % (
                        ifield,
                        self.shear_comp_sim,
                        irot,
                        self.bands,
                    ),
                )
                e1_1, r1_1 = self.get_obs_sum(in_nm1, cat_obj)
                in_nm2 = os.path.join(
                    self.cat_dir,
                    "src-%05d_%s-1_rot%d_%s.fits"
                    % (
                        ifield,
                        self.shear_comp_sim,
                        irot,
                        self.bands,
                    ),
                )
                e1_2, r1_2 = self.get_obs_sum(in_nm2, cat_obj)
                out[icount, 0] = ifield
                out[icount, 1] = out[icount, 1] + (e1_2 - e1_1)
                out[icount, 2] = out[icount, 2] + (e1_1 + e1_2) / 2.0
                out[icount, 3] = out[icount, 3] + (r1_1 + r1_2) / 2.0
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 4.0
        print("elapsed time: %.2f seconds" % elapsed_time)
        return out

    def get_obs_sum(self, mname, cat_obj):
        if self.g_comp_measure == 1:
            func = cat_obj.measure_g1_renoise
        elif self.g_comp_measure == 2:
            func = cat_obj.measure_g2_renoise
        else:
            raise ValueError("g_comp_measure should be 1 or 2")
        assert os.path.isfile(mname), (
            "Cannot find galaxy shear catalogs : %s " % mname
        )
        mm = fitsio.read(mname)
        nname = mname.replace("src-", "noise-")
        if os.path.isfile(nname):
            nn = fitsio.read(nname)
        else:
            nn = jax.numpy.zeros_like(mm)
        dname = mname.replace("src-", "det-")
        det = fitsio.read(dname)
        r2 = (det[:, 0] - self.image_center) ** 2.0 + (
            det[:, 1] - self.image_center
        ) ** 2.0
        sel = r2 < self.rmax2
        mm = mm[sel]
        nn = nn[sel]
        del sel
        e1_sum, r1_sum = jax.numpy.sum(func(mm, nn), axis=0)
        return e1_sum, r1_sum

    def display_result(self, test_obs=None):
        if test_obs is None:
            cname = self.test_obs
        else:
            cname = test_obs

        spt = "bin_%s_" % cname
        flist = glob.glob("%s/%s*.fits" % (self.sum_dir, spt))
        res = []
        for fname in flist:
            obs = float(fname.split("/")[-1].split(spt)[-1].split(".fits")[0])
            obs = obs / float(pf[cname])
            print("%s is: %s" % (cname, obs))
            a = fitsio.read(fname)
            rave = np.average(a[:, 3])
            msk = (a[:, 3] >= 100) & (a[:, 3] < rave * 2.0)
            a = a[msk]
            a = a[np.argsort(a[:, 0])]
            nsim = a.shape[0]
            b = np.average(a, axis=0)
            mbias = b[1] / b[3] / self.shear_value / 2.0 - 1
            print(
                "multiplicative bias:",
                mbias,
            )
            merr = (
                np.std(a[:, 1])
                / np.average(a[:, 3])
                / self.shear_value
                / 2.0
                / np.sqrt(nsim)
            )
            print(
                "1-sigma error:",
                merr,
            )
            cbias = b[2] / b[3]
            print("additive bias:", cbias)
            cerr = np.std(a[:, 2]) / np.average(a[:, 3]) / np.sqrt(nsim)
            print(
                "1-sigma error:",
                cerr,
            )
            res.append((obs, mbias, merr, cbias, cerr))
        dtype = [
            (cname, "float"),
            ("mbias", "float"),
            ("merr", "float"),
            ("cbias", "float"),
            ("cerr", "float"),
        ]
        res = np.sort(np.array(res, dtype=dtype), order=cname)
        return res
