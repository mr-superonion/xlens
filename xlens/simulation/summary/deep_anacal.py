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
import numpy as np

from ..simulator.base import SimulateBatchBase

pf = {
    "snr_min": 1.0,
    "r2_min": 100.0,
    "r2_max": 100.0,
}


class SummarySimDeepAnacal(SimulateBatchBase):
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
        self.nord = cparser.getint("FPFS", "nord", fallback=4)
        self.det_nrot = cparser.getint("FPFS", "det_nrot", fallback=4)
        self.pthres = cparser.getfloat("FPFS", "pthres", fallback=0.12)
        self.c0 = cparser.getfloat("FPFS", "c0", fallback=10)
        self.snr_min = cparser.getfloat("FPFS", "snr_min", fallback=10.0)
        self.r2_min = cparser.getfloat("FPFS", "r2_min", fallback=0.05)

        # shear setup
        self.shear_value = cparser.getfloat("simulation", "shear_value")

        # summary
        if not os.path.isdir(self.sum_dir):
            os.makedirs(self.sum_dir, exist_ok=True)
        self.test_obs = cparser.get("FPFS", "test_obs", fallback="snr_min")
        self.cut = getattr(self, self.test_obs)
        self.ofname = os.path.join(
            self.sum_dir,
            "bin.fits"
        )
        self.image_center = (self.coadd_dim + 10) / 2.0
        return

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.zeros((len(id_range), 9))
        print("start core: %d, with id: %s" % (icore, id_range))
        start_time = time.time()
        assert self.cat_dir is not None
        for icount, ifield in enumerate(id_range):
            for irot in range(self.nrot):
                swm = os.path.join(
                    self.cat_dir,
                    "src-wide-%05d_%s-0_rot%d_%s.fits"
                    % (
                        ifield,
                        self.shear_comp_sim,
                        irot,
                        self.bands,
                    ),
                )
                swp = os.path.join(
                    self.cat_dir,
                    "src-wide-%05d_%s-1_rot%d_%s.fits"
                    % (
                        ifield,
                        self.shear_comp_sim,
                        irot,
                        self.bands,
                    ),
                )
                sdp = swp.replace("wide", "deep")
                sdm = swm.replace("wide", "deep")
                wp = fitsio.read(swp)
                wm = fitsio.read(swm)
                dp = fitsio.read(sdp)
                dm = fitsio.read(sdm)
                ep = np.sum(wp["fpfs_e1"] * wp["fpfs_w"])
                qp = np.sum(wp["fpfs_q1"] * wp["fpfs_w"])
                rp = np.sum(
                    dp["fpfs_de1_dg1"] * dp["fpfs_w"] + dp["fpfs_e1"] * dp["fpfs_dw_dg1"]
                )
                rqp = np.sum(
                    dp["fpfs_dq1_dg1"] * dp["fpfs_w"] + dp["fpfs_q1"] * dp["fpfs_dw_dg1"]
                )
                em = np.sum(wm["fpfs_e1"] * wm["fpfs_w"])
                qm = np.sum(wm["fpfs_q1"] * wm["fpfs_w"])
                rm = np.sum(
                    dm["fpfs_de1_dg1"] * dm["fpfs_w"] + dm["fpfs_e1"] * dm["fpfs_dw_dg1"]
                )
                rqm = np.sum(
                    dm["fpfs_dq1_dg1"] * dm["fpfs_w"] + dm["fpfs_q1"] * dm["fpfs_dw_dg1"]
                )
                out[icount, 0] = ifield
                out[icount, 1] = out[icount, 1] + ep
                out[icount, 2] = out[icount, 2] + em
                out[icount, 3] = out[icount, 3] + rp
                out[icount, 4] = out[icount, 4] + rm
                out[icount, 5] = out[icount, 5] + qp
                out[icount, 6] = out[icount, 6] + qm
                out[icount, 7] = out[icount, 7] + rqp
                out[icount, 8] = out[icount, 8] + rqm
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 4.0
        print("elapsed time: %.2f seconds" % elapsed_time)
        return out

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
