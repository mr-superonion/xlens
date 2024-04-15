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
import gc
import os
import time
from configparser import ConfigParser, ExtendedInterpolation

import anacal
import fitsio
import numpy as np

from ..simulator.base import SimulateBase
from ..simulator.loader import MakeDMExposure
from .utils import get_psf_array

# from memory_profiler import profile


def rotate90(image):
    rotated_image = np.zeros_like(image)
    rotated_image[1:, 1:] = np.rot90(m=image[1:, 1:], k=-1)
    return rotated_image


class ProcessSimAnacal(SimulateBase):
    # @profile
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser)
        self.config_name = config_name
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find image directory")
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir, exist_ok=True)

        # setup FPFS task
        self.sigma_as = cparser.getfloat("FPFS", "sigma_as")
        self.rcut = cparser.getint("FPFS", "rcut", fallback=32)
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

        self.pthres = cparser.getfloat("FPFS", "pthres", fallback=0.0)
        self.pratio = cparser.getfloat("FPFS", "pratio", fallback=0.02)
        self.wdet_cut = cparser.getfloat("FPFS", "wdet_cut", fallback=0.00)
        self.det_nrot = cparser.getint("FPFS", "det_nrot", fallback=4)

        self.ncov_fname = cparser.get(
            "FPFS",
            "ncov_fname",
            fallback="",
        )
        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")

        self.corr_fname = cparser.get(
            "simulation",
            "noise_corr_fname",
            fallback=None,
        )

        self.center_name = cparser.get(
            "FPFS",
            "center_name",
            fallback="deblend_peak_center",
        )
        return

    def process_image(
        self,
        gal_array,
        psf_array,
        cov_elem,
        pixel_scale,
        noise_array,
    ):
        # Detection
        nn = self.coadd_dim + 10
        dtask = anacal.fpfs.FpfsDetect(
            nx=nn,
            ny=nn,
            psf_array=psf_array,
            pix_scale=pixel_scale,
            sigma_arcsec=self.sigma_as,
            det_nrot=self.det_nrot,
        )
        std_m00, std_v = dtask.get_stds(cov_elem)
        coords = dtask.run(
            gal_array=gal_array,
            fthres=8.5,
            pthres=self.pthres,
            pratio=self.pratio,
            bound=self.rcut + 5,
            std_m00=std_m00,
            std_v=std_v,
            noise_array=noise_array,
            wdet_cut=self.wdet_cut,
        )
        del dtask
        print("pre-selected number of sources: %d" % len(coords))

        mtask = anacal.fpfs.FpfsMeasure(
            psf_array=psf_array,
            pix_scale=pixel_scale,
            sigma_arcsec=self.sigma_as,
            det_nrot=self.det_nrot,
        )
        src = mtask.run(gal_array=gal_array, det=coords)
        if noise_array is not None:
            noise = mtask.run(noise_array, det=coords, do_rotate=True)
            src = src + noise
        else:
            noise = None
        sel = (src[:, mtask.di["m00"]] + src[:, mtask.di["m20"]]) > 1e-5
        coords = np.array(coords)[sel]
        src = src[sel]
        if noise is not None:
            noise = noise[sel]
        del mtask, sel
        gc.collect()
        return coords, src, noise

    def prepare_data(self, file_name):
        dm_task = MakeDMExposure(self.config_name)
        # using seeds that are not used in simulation
        seed = dm_task.get_seed_from_fname(file_name, "i") + 1
        exposure = dm_task.generate_exposure(file_name)
        pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
        variance = np.average(exposure.getMaskedImage().variance.array)
        psf_array = np.asarray(get_psf_array(exposure, ngrid=self.ngrid))
        anacal.fpfs.util.truncate_square(psf_array, self.psf_rcut)
        gal_array = np.asarray(exposure.getMaskedImage().image.array)
        del exposure, dm_task
        ny = self.coadd_dim + 10
        nx = self.coadd_dim + 10
        noise_std = np.sqrt(variance)
        if variance > 1e-8:
            if self.corr_fname is None:
                noise_array = np.random.RandomState(seed).normal(
                    scale=noise_std,
                    size=(ny, nx),
                )
            else:
                noise_corr = fitsio.read(self.corr_fname)
                noise_corr = rotate90(noise_corr)
                noise_array = (
                    anacal.noise.simulate_noise(
                        seed=seed,
                        correlation=noise_corr,
                        nx=nx,
                        ny=ny,
                        scale=pixel_scale,
                    )
                    * noise_std
                )
        else:
            noise_array = None
        cov_elem = fitsio.read(self.ncov_fname)
        gc.collect()
        return {
            "gal_array": gal_array,
            "psf_array": psf_array,
            "cov_elem": cov_elem,
            "pixel_scale": pixel_scale,
            "noise_array": noise_array,
        }

    # @profile
    def run(self, file_name):
        src_name = os.path.join(self.cat_dir, file_name.split("/")[-1])
        src_name = src_name.replace(
            "image-",
            "src-",
        ).replace(
            "_xxx",
            "_%s" % self.bands,
        )
        det_name = src_name.replace("src-", "det-")
        noi_name = src_name.replace("src-", "noise-")

        # dm_task = MakeDMExposure(self.config_name)
        # seed = dm_task.get_seed_from_fname(file_name, "i") + 1

        if os.path.isfile(src_name) and os.path.isfile(det_name):
            print("Already has measurement for simulation: %s." % src_name)
            return

        data = self.prepare_data(file_name)
        start_time = time.time()
        det, src, noise = self.process_image(
            gal_array=data["gal_array"],
            psf_array=data["psf_array"],
            cov_elem=data["cov_elem"],
            pixel_scale=data["pixel_scale"],
            noise_array=data["noise_array"],
        )
        del data
        elapsed_time = time.time() - start_time
        print("Elapsed time: %.2f seconds, number of gals: %d" % (elapsed_time, len(src)))
        fitsio.write(det_name, np.asarray(det))
        fitsio.write(src_name, np.asarray(src))
        if noise is not None:
            fitsio.write(noi_name, np.asarray(noise))
        del src, det, noise
        gc.collect()
        return
