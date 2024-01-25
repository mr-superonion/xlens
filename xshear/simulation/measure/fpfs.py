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
import json
import os
import time
from configparser import ConfigParser, ExtendedInterpolation

import fitsio
import fpfs
import numpy as np
from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.psfs import make_fixed_psf
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.stars import StarCatalog

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
        self.ngrid = 2 * self.rcut
        psf_rcut = cparser.getint("FPFS", "psf_rcut", fallback=22)
        self.psf_rcut = min(psf_rcut, self.rcut)
        self.nord = cparser.getint("FPFS", "nord", fallback=4)
        self.ncov_fname = cparser.get(
            "FPFS",
            "ncov_fname",
            fallback="",
        )
        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")
        return

    def process_image(self, gal_array, psf_array, cov_elem, pixel_scale):
        # measurement task
        meas_task = fpfs.image.measure_source(
            psf_array,
            sigma_arcsec=self.sigma_as,
            sigma_detect=self.sigma_det,
            nord=self.nord,
            pix_scale=pixel_scale,
        )

        npad = (self.image_nx - psf_array.shape[0]) // 2
        coords = meas_task.detect_sources(
            img_data=gal_array,
            psf_data=np.pad(psf_array, (npad, npad), mode="constant"),
            cov_elem=cov_elem,
            thres=9.5,
            thres2=-1.0,
            bound=self.rcut,
        )
        print("pre-selected number of sources: %d" % len(coords))
        out = meas_task.measure(gal_array, coords)
        out = meas_task.get_results(out)
        coords = meas_task.get_results_detection(coords)
        return out, coords

    def measure_exposure(self, exposure):
        pixel_scale = exposure.getWcs().getPixelScale().asArcseconds()
        masked_image = exposure.getMaskedImage()
        gal_array = masked_image.image.array[:, :]
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
            # By default, we use uncorrelated noise
            # TODO: enable correlated noise here
            noise_pow = np.ones((self.ngrid, self.ngrid)) * variance * self.ngrid**2.0
            cov_elem = np.array(noise_task.measure(noise_pow))
            fitsio.write(self.ncov_fname, cov_elem, overwrite=True)
        else:
            cov_elem = fitsio.read(self.ncov_fname)
        assert np.all(np.diagonal(cov_elem) > 1e-10), "The covariance matrix is incorrect"

        start_time = time.time()
        cat, det = self.process_image(gal_array, psf_array, cov_elem, pixel_scale)
        del gal_array, psf_array, cov_elem
        # Stop the timer
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        return cat, det

    def run(self, exposure):
        return self.measure_exposure(exposure)


class ProcessSimFPFS(MakeDMExposure):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.meas_task = FPFSMeasurementTask(config_name)
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir, exist_ok=True)

    def run(self, file_name):
        print("processing file: %s" % file_name)
        out_name = os.path.join(self.cat_dir, file_name.split("/")[-1])
        out_name = out_name.replace(
            "image-",
            "src-",
        ).replace(
            "_xxx",
            "_%s" % self.bands,
        )
        det_name = out_name.replace("src-", "det-")
        if os.path.isfile(out_name) and os.path.isfile(det_name):
            print("Already has measurement for simulation: %s." % out_name)
            return
        exposure = self.generate_exposure(file_name)
        cat, det = self.meas_task.measure_exposure(exposure)
        fpfs.io.save_catalog(
            det_name,
            det,
            dtype="position",
            nord="4",
        )
        fpfs.io.save_catalog(
            out_name,
            cat,
            dtype="shape",
            nord="4",
        )
        if self.do_ds9_region:
            ds9_name = out_name.replace("src-", "reg-")
            ds9_name = ds9_name.replace(".fits", ".reg")
            pos = det[["fpfs_x", "fpfs_y"]]
            self.write_ds9_region(pos, ds9_name)
        if self.do_debug_exposure:
            img_name = out_name.replace("src-", "img-")
            self.write_image(exposure, img_name)
        return
