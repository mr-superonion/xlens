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
from deep_anacal import deep_anacal
import fitsio
import numpy as np
from numpy.typing import NDArray

from ..simulator.base import SimulateBase
from ..simulator.loader_band import MakeDMExposureSV_SingleBand
from .utils import get_gridpsf_obj, get_psf_array

npix_patch = 256
npix_overlap = 64
npix_default = 64


def rotate90(image):
    rotated_image = np.zeros_like(image)
    rotated_image[1:, 1:] = np.rot90(m=image[1:, 1:], k=-1)
    return rotated_image


class DeepAnacalForcePhot(SimulateBase):
    # @profile
    def __init__(self, config_name, cat_i):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser)
        self.config_name = config_name
        self.img_dir = self.img_dir.replace("xxx", str(cat_i))
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find image directory")
        assert self.cat_dir is not None
        self.cat_dir = self.cat_dir.replace("xxx", str(cat_i))
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir, exist_ok=True)
        self.cat_i = cat_i
        # setup FPFS task
        self.sigma_arcsec = cparser.getfloat("FPFS", "sigma_arcsec")
        self.rcut = cparser.getint("FPFS", "rcut", fallback=32)
        self.ngrid = 2 * self.rcut
        psf_rcut = cparser.getint("FPFS", "psf_rcut", fallback=22)
        self.psf_rcut = min(psf_rcut, self.rcut)
        self.nord = cparser.getint("FPFS", "nord", fallback=4)
        self.det_nrot = cparser.getint("FPFS", "det_nrot", fallback=4)
        assert self.nord >= 4
        assert self.det_nrot >= 4

        self.pthres = cparser.getfloat("FPFS", "pthres", fallback=0.12)
        self.klim_thres = cparser.getint("FPFS", "klim_thres", fallback=1e-12)

        # pixel level noise correlation function
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
        
        self.deep_noise_frac = cparser.get(
            "FPFS",
            "deep_noise_frac",
            fallback=0.1
        )
        return

    def get_file_names(self, file_name, band):
        assert self.cat_dir is not None
        srcs_name = os.path.join(self.cat_dir, file_name.split("/")[-1])
        self.src_det_name = srcs_name.replace(
            "image-",
            "det-",
        ).replace(
            "_xxx",
            "_%s" % "a",
        )
        self.srcd_name = srcs_name.replace(
            "image-",
            "src-",
        ).replace(
            "_xxx",
            "_%s" % band,
        )
        return

    def process_image(
        self,
        data,
    ):
        scale = data["pixel_scale"]
        seed = data["seed"]
        # Detection (use wide field images)
        fpfs_config = anacal.fpfs.FpfsConfig(sigma_arcsec=self.sigma_arcsec)
        dtask = anacal.fpfs.FpfsDeepWideImage(
            nx=npix_patch, ny=npix_patch, scale=scale,
            sigma_arcsec=self.sigma_arcsec, klim=2.650718801466388/0.2,
            use_estimate=True, npix_overlap=npix_overlap, bound=fpfs_config.bound
        )
        ftask_w = deep_anacal.create_fpfs_task(
            fpfs_config, scale, 0.5 * data["noise_std"]**2*10, data["psf_array"]
            )
        ftask_d = deep_anacal.create_fpfs_task(
            fpfs_config, scale, data["noise_std"]**2, data["psf_array"]
            )
        std_m00 = np.sqrt(ftask_w.std_m00**2 + ftask_d.std_m00**2)
        coords_deep = fitsio.read(self.src_det_name)
        
        print(f"pre-selected number: {len(coords_deep)}")

        if not os.path.isfile(self.srcd_name):
            print(
                "Mesuring Shapelet modes with sigma_arcsec=%.2f arcsec"
                % self.sigma_arcsec
            )
            res_wide, res_deep = deep_anacal.run_deep_anacal(
                scale=scale,
                fpfs_config=fpfs_config,
                gal_array_w=np.zeros_like(data["gal_array"]),
                gal_array_d=data["gal_array"],
                psf_array_w=data["psf_array"],
                psf_array_d=data["psf_array"],
                noise_var_w=data["noise_std"]**2*10,
                noise_var_d=data["noise_std"]**2,
                noise_array_w=data["noise_wide"],
                noise_array_d=data["noise_array"],
                noise_array_d90=data["noise_array90"],
                detection_w=None,
                detection_d=coords_deep,
            )
            fitsio.write(self.srcd_name, res_deep)
            del coords_deep, res_wide, res_deep
        gc.collect()
        return

    def prepare_data(self, file_name, band):
        dm_task = MakeDMExposureSV_SingleBand(self.config_name, self.cat_i, band)
        # using seeds that are not used in simulation
        seed = dm_task.get_seed_from_fname(file_name, "i") + 1
        exposure = dm_task.generate_exposure(file_name)
        pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
        variance = np.average(exposure.getMaskedImage().variance.array)
        psf_obj = get_gridpsf_obj(
            exposure,
            ngrid=self.ngrid,
            psf_rcut=self.psf_rcut,
            dg=250,
        )
        psf_array = get_psf_array(
            exposure,
            ngrid=self.ngrid,
            psf_rcut=self.psf_rcut,
            dg=250,
        ).astype(np.float64)
        gal_array = np.asarray(
            exposure.getMaskedImage().image.array,
            dtype=np.float64,
        )
        mask_array = np.asanyarray(
            exposure.getMaskedImage().mask.array,
            dtype=np.int16,
        )
        del exposure, dm_task
        ny = self.coadd_dim + 10
        nx = self.coadd_dim + 10
        noise_std = np.sqrt(variance)
        if variance > 1e-8:
            if self.corr_fname is None:
                rng = np.random.RandomState(seed)
                noise_array = (
                    rng
                    .normal(
                        scale=noise_std,
                        size=(ny, nx),
                    )
                    .astype(np.float64)
                )
                noise_array90 = (
                    rng
                    .normal(
                        scale=noise_std,
                        size=(ny, nx),
                    )
                    .astype(np.float64)
                )
                noise_wide = (
                    rng
                    .normal(
                        scale=noise_std * np.sqrt(10),
                        size=(ny, nx),
                    )
                    .astype(np.float64)
                )
                noise_array90 = rotate90(noise_array90)
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
                    ).astype(np.float64)
                    * noise_std
                )
                noise_array = None
        else:
            noise_array = None
            noise_array2 = None
        star_cat = None
        gc.collect()
        return {
            "gal_array": gal_array,
            "psf_array": psf_array,
            "noise_array": noise_array,
            "noise_array90": noise_array90,
            "noise_wide": noise_wide,
            "pixel_scale": pixel_scale,
            "noise_std": noise_std,
            "psf_obj": psf_obj,
            "mask_array": mask_array,
            "star_cat": star_cat,
            "seed": seed,
        }

    # @profile
    def run(self, file_name):
        start_time = time.time()
        for band in "griz":
            data = self.prepare_data(file_name, band)
            self.get_file_names(file_name, band)
            self.process_image(data)
            del data
        elapsed_time = time.time() - start_time
        print("Elapsed time: %.2f seconds" % elapsed_time)
        return
