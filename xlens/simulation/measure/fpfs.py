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

import fitsio
import fpfs
import jax
import jax.numpy as jnp
import numpy as np
import numpy.lib.recfunctions as rfn

from ..simulator.base import SimulateBase
from ..simulator.loader import MakeDMExposure
from .utils import get_psf_array

# from memory_profiler import profile


class ProcessSimFpfs(SimulateBase):
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
        self.det_nrot = cparser.getint("FPFS", "det_nrot", fallback=8)

        self.ncov_fname = cparser.get(
            "FPFS",
            "ncov_fname",
            fallback="",
        )
        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")

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
        # measurement task
        task = fpfs.image.measure_source(
            psf_array,
            sigma_arcsec=self.sigma_as,
            nord=self.nord,
            pix_scale=pixel_scale,
            det_nrot=self.det_nrot,
        )
        coords = task.detect_source(
            img_array=gal_array,
            psf_array=psf_array,
            cov_elem=cov_elem,
            fthres=8.5,
            pthres=self.pthres,
            pratio=self.pratio,
            bound=self.rcut + 5,
            noise_array=noise_array,
        )

        print("pre-selected number of sources: %d" % len(coords))
        src = task.measure(gal_array, coords)
        noise = task.measure(noise_array, coords, psf_f=task.psf_rot_f)
        src = src + noise
        sel = (src[:, task.di["m00"]] + src[:, task.di["m20"]]) > 1e-5
        coords = coords[sel]
        src = src[sel]
        noise = noise[sel]
        del task
        return coords, src, noise

    def prepare_data(self, file_name):
        dm_task = MakeDMExposure(self.config_name)
        exposure = dm_task.generate_exposure(file_name)
        pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
        variance = np.average(exposure.getMaskedImage().variance.array)
        seed = dm_task.get_seed_from_fname(file_name, "i") + 1
        ny = self.coadd_dim + 10
        nx = self.coadd_dim + 10
        rng = np.random.RandomState(seed)
        noise_array = rng.normal(
            scale=np.sqrt(variance),
            size=(ny, nx),
        )
        gal_array = jnp.asarray(exposure.getMaskedImage().image.array)

        psf_array = np.asarray(get_psf_array(exposure, ngrid=self.ngrid))
        fpfs.image.util.truncate_square(psf_array, self.psf_rcut)
        del exposure
        if not os.path.isfile(self.ncov_fname):
            # FPFS noise cov task
            noise_task = fpfs.image.measure_noise_cov(
                psf_array,
                sigma_arcsec=self.sigma_as,
                nord=self.nord,
                pix_scale=pixel_scale,
                det_nrot=self.det_nrot,
            )
            noise_pow = (
                np.ones((self.ngrid, self.ngrid)) * variance * 2.0 * self.ngrid**2.0
            )  # variance times 2
            cov_elem = noise_task.measure(noise_pow)
            fitsio.write(self.ncov_fname, np.asarray(cov_elem), overwrite=True)
            del noise_task
        else:
            cov_elem = jnp.asarray(fitsio.read(self.ncov_fname))
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
        if os.path.isfile(src_name) and os.path.isfile(det_name):
            print("Already has measurement for simulation: %s." % src_name)
            return

        data = self.prepare_data(file_name)
        start_time = time.time()
        det, src, noise = self.process_image(**data)
        elapsed_time = time.time() - start_time
        print("Elapsed time: %.2f seconds, number of gals: %d" % (elapsed_time, len(src)))
        del data
        fitsio.write(det_name, np.asarray(det))
        fitsio.write(src_name, np.asarray(src))
        fitsio.write(noi_name, np.asarray(noise))
        del src, det
        gc.collect()
        return
