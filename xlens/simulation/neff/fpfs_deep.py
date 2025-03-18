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
import numpy as np

from ..simulator.base import SimulateBatchBase


class NeffSimFpfsDeep(SimulateBatchBase):
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


        # shear setup
        self.g_comp_measure = cparser.getint(
            "FPFS",
            "g_comp_measureonent_measure",
            fallback=1,
        )
        assert self.g_comp_measure in [
            1,
            2,
        ], "The g_comp_measure in configure file is not supported"
        self.imode = self.shear_mode_list[-1]

        # neff
        coadd_dim = cparser.getint("simulation", "coadd_dim")
        buff = cparser.getint("simulation", "buff")
        coadd_scale = cparser.getfloat(
            "simulation", "coadd_scale", fallback=0.2
        )
        radius = ((coadd_dim + 10) / 2.0 - buff) * coadd_scale / 60.0
        self.area = np.pi * radius**2  # [arcmin^2]
        self.ofname = os.path.join(
            self.sum_dir,
            "neff.fits"
        )
        return

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.zeros((len(id_range), 2))

        print("start core: %d, with id: %s" % (icore, id_range))
        start_time = time.time()
        for icount, ifield in enumerate(id_range):

            wf = os.path.join(
                self.cat_dir,
                "src-wide-%05d_g1-%d_rot0_%s.fits"
                % (ifield, self.imode, self.bands),
            )
            df = os.path.join(
                self.cat_dir,
                "src-deep-%05d_g1-%d_rot0_%s.fits"
                % (ifield, self.imode, self.bands),
            )
            dw = fitsio.read(wf)
            dd = fitsio.read(df)
            out[icount, 0] = np.sum(dw["fpfs_w"] * dw["fpfs_e1"])
            out[icount, 1] = np.sum(dd["fpfs_w"] * dd["fpfs_de1_dg1"] + dd["fpfs_dw_dg1"] * dd["fpfs_e1"])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("elapsed time: %.2f seconds" % elapsed_time)
        return out
    
class NeffSimFpfsAnacal(SimulateBatchBase):
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


        # shear setup
        self.g_comp_measure = cparser.getint(
            "FPFS",
            "g_comp_measureonent_measure",
            fallback=1,
        )
        assert self.g_comp_measure in [
            1,
            2,
        ], "The g_comp_measure in configure file is not supported"
        self.imode = self.shear_mode_list[-1]

        # neff
        coadd_dim = cparser.getint("simulation", "coadd_dim")
        buff = cparser.getint("simulation", "buff")
        coadd_scale = cparser.getfloat(
            "simulation", "coadd_scale", fallback=0.2
        )
        radius = ((coadd_dim + 10) / 2.0 - buff) * coadd_scale / 60.0
        self.area = np.pi * radius**2  # [arcmin^2]
        self.ofname = os.path.join(
            self.sum_dir,
            "neff.fits"
        )
        return

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.zeros((len(id_range), 2))

        print("start core: %d, with id: %s" % (icore, id_range))
        start_time = time.time()
        for icount, ifield in enumerate(id_range):

            wf = os.path.join(
                self.cat_dir,
                "src-%05d_g1-%d_rot0_%s.fits"
                % (ifield, self.imode, self.bands),
            )
            dw = fitsio.read(wf)
            out[icount, 0] = np.sum(dw["fpfs_w"] * dw["fpfs_e1"])
            out[icount, 1] = np.sum(dw["fpfs_w"] * dw["fpfs_de1_dg1"] + dw["fpfs_dw_dg1"] * dw["fpfs_e1"])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("elapsed time: %.2f seconds" % elapsed_time)
        return out
