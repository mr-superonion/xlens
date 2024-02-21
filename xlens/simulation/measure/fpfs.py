#
# LSST Data Management System
# Copyright 20082014 LSST Corpoalphan.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import os
import time
from configparser import ConfigParser, ExtendedInterpolation

import fitsio
import fpfs
import numpy as np
import numpy.lib.recfunctions as rfn

from ..loader import MakeDMExposure
from ..simulator import SimulateBase
from .utils import get_psf_array


class FPFSMeasurementTask(SimulateBase):
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find image directory")
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir, exist_ok=True)

        # setup FPFS task
        self.sigma_as = cparser.getfloat("FPFS", "sigma_as")
        self.sigma_det = cparser.getfloat("FPFS", "sigma_det")
        self.rcut = cparser.getint("FPFS", "rcut", fallback=32)
        self.thres2 = cparser.getfloat("FPFS", "thres2", fallback=0.0)
        self.ngrid = 2 * self.rcut
        psf_rcut = cparser.getint("FPFS", "psf_rcut", fallback=22)
        self.psf_rcut = min(psf_rcut, self.rcut)

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
        self.center_name = cparser.get(
            "FPFS",
            "center_name",
            fallback="deblend_peak_center",
        )
        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")
        return

    def process_image(self, gal_array, psf_array, cov_elem, pixel_scale, dm_fname):
        # measurement task
        task = fpfs.image.measure_source(
            psf_array,
            sigma_arcsec=self.sigma_as,
            sigma_detect=self.sigma_det,
            nord=self.nord,
            pix_scale=pixel_scale,
        )
        if dm_fname is None:
            coords = task.detect_sources(
                img_data=gal_array,
                psf_data=psf_array,
                cov_elem=cov_elem,
                thres=8.0,
                thres2=self.thres2,
                bound=self.rcut,
            )
        else:
            print("Using detected centers: %s" % dm_fname)
            cname_list = [self.center_name + "_y", self.center_name + "_x"]
            tmp = fitsio.read(dm_fname)
            msk = (tmp["deblend_nPeaks"] == 1) & (tmp["deblend_nChild"] == 0)
            coords = rfn.structured_to_unstructured(tmp[msk][cname_list])
            del tmp, msk
            coords = np.int_(np.round(coords))

        print("pre-selected number of sources: %d" % len(coords))
        out = task.get_results(task.measure(gal_array, coords))
        out2 = task.get_results_detection(coords)
        return out, out2

    def run(self, exposure, dm_fname=None):
        pixel_scale = exposure.getWcs().getPixelScale().asArcseconds()
        masked_image = exposure.getMaskedImage()
        gal_array = masked_image.image.array
        variance = np.average(masked_image.variance.array)
        self.image_nx = gal_array.shape[1]

        psf_array = get_psf_array(exposure, ngrid=self.ngrid)
        fpfs.image.util.truncate_square(psf_array, self.psf_rcut)
        if not os.path.isfile(self.ncov_fname):
            # FPFS noise cov task
            noise_task = fpfs.image.measure_noise_cov(
                psf_array,
                sigma_arcsec=self.sigma_as,
                sigma_detect=self.sigma_det,
                nord=self.nord,
                pix_scale=pixel_scale,
            )
            noise_pow = np.ones((self.ngrid, self.ngrid)) * variance * self.ngrid**2.0
            cov_elem = np.array(noise_task.measure(noise_pow))
            fitsio.write(self.ncov_fname, cov_elem, overwrite=True)
        else:
            cov_elem = fitsio.read(self.ncov_fname)

        start_time = time.time()
        cat, det = self.process_image(
            gal_array,
            psf_array,
            cov_elem,
            pixel_scale,
            dm_fname,
        )
        elapsed_time = time.time() - start_time
        print(f"elapsed time: {elapsed_time} seconds")
        return cat, det


class ProcessSimFPFS(MakeDMExposure):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.config_name = config_name
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir, exist_ok=True)
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        self.detection_dir = cparser.get(
            "FPFS",
            "detection_dir",
            fallback="",
        )
        print(self.detection_dir)

    def run(self, file_name):
        print("processing file: %s" % file_name)
        cat_name = os.path.join(self.cat_dir, file_name.split("/")[-1])
        cat_name = cat_name.replace(
            "image-",
            "src-",
        ).replace(
            "_xxx",
            "_%s" % self.bands,
        )
        if len(self.detection_dir) == 0 or not os.path.isdir(self.detection_dir):
            dm_fname = None
        else:
            dm_fname = os.path.join(self.detection_dir, file_name.split("/")[-1])
            dm_fname = dm_fname.replace(
                "image-",
                "src-",
            ).replace(
                "_xxx",
                "_%s" % self.bands,
            )
        det_name = cat_name.replace("src-", "det-")
        if os.path.isfile(cat_name) and os.path.isfile(det_name):
            print("Already has measurement for simulation: %s." % cat_name)
            return
        exposure = self.generate_exposure(file_name)
        meas_task = FPFSMeasurementTask(self.config_name)
        cat, det = meas_task.run(exposure, dm_fname=dm_fname)
        if self.do_debug_exposure:
            img_name = cat_name.replace("src-", "img-")
            self.write_image(exposure, img_name)
        fpfs.io.save_catalog(
            det_name,
            det,
            dtype="position",
            nord="4",
        )
        fpfs.io.save_catalog(
            cat_name,
            cat,
            dtype="shape",
            nord="4",
        )
        if self.do_ds9_region:
            ds9_name = cat_name.replace("src-", "reg-")
            ds9_name = ds9_name.replace(".fits", ".reg")
            pos = det[["fpfs_x", "fpfs_y"]]
            self.write_ds9_region(pos, ds9_name)
            del pos
        return
