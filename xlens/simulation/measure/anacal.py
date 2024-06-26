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
from numpy.typing import NDArray

from ..simulator.base import SimulateBase
from ..simulator.loader import MakeDMExposure
from .utils import get_gridpsf_obj, get_psf_array


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
        assert self.cat_dir is not None
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir, exist_ok=True)

        # setup FPFS task
        self.sigma_arcsec = cparser.getfloat("FPFS", "sigma_arcsec")
        self.sigma_arcsec2 = cparser.getfloat(
            "FPFS",
            "sigma_arcsec2",
            fallback=None,
        )
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

        # noise covariance matrix on basis modes
        self.ncov_fname = cparser.get(
            "FPFS",
            "ncov_fname",
            fallback=None,
        )
        if self.ncov_fname is None:
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")

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
        self.estimate_cov_matrix = True
        return

    def get_file_names(self, file_name):
        assert self.cat_dir is not None
        srcs_name = os.path.join(self.cat_dir, file_name.split("/")[-1])
        self.srcs_name = srcs_name.replace(
            "image-",
            "src_s-",
        ).replace(
            "_xxx",
            "_%s" % self.bands,
        )
        self.srcs2_name = self.srcs_name.replace("src_s-", "src_s2-")
        self.srcd_name = self.srcs_name.replace("src_s-", "src_d-")
        self.det_name = self.srcs_name.replace("src_s-", "det-")
        return

    def process_image(
        self,
        gal_array: NDArray,
        psf_array: NDArray,
        cov_matrix: anacal.fpfs.table.Covariance,
        pixel_scale: float,
        noise_array: NDArray | None,
        psf_obj: anacal.fpfs.BasePsf | None,
        mask_array: NDArray,
        star_cat: NDArray,
    ):
        # Detection
        nn = self.coadd_dim + 10
        mag_zero = cov_matrix.mag_zero
        if os.path.isfile(self.det_name):
            coords = fitsio.read(self.det_name)
        else:
            dtask = anacal.fpfs.FpfsDetect(
                nx=nn,
                ny=nn,
                psf_array=psf_array,
                mag_zero=mag_zero,
                pixel_scale=pixel_scale,
                sigma_arcsec=self.sigma_arcsec,
                cov_matrix=cov_matrix,
                det_nrot=self.det_nrot,
                klim_thres=self.klim_thres,
            )
            coords = dtask.run(
                gal_array=gal_array,
                fthres=8.5,
                pthres=self.pthres,
                bound=self.rcut + 5,
                noise_array=noise_array,
                mask_array=mask_array,
                star_cat=star_cat,
            )
            fitsio.write(self.det_name, coords)
            del dtask
        print("pre-selected number of sources: %d" % len(coords))

        # Detection Modes
        if not os.path.isfile(self.srcd_name):
            print("Mesuring Detection modes")
            mtask_d = anacal.fpfs.FpfsMeasure(
                psf_array=psf_array,
                mag_zero=mag_zero,
                pixel_scale=pixel_scale,
                sigma_arcsec=self.sigma_arcsec,
                klim_thres=self.klim_thres,
                nord=-1,
                det_nrot=self.det_nrot,
            )
            src_d = mtask_d.run(
                gal_array=gal_array,
                det=coords,
                noise_array=noise_array,
                psf=psf_obj,
            )
            src_d.write(self.srcd_name)
            del mtask_d, src_d

        # Shapelet Modes
        if not os.path.isfile(self.srcs_name):
            print(
                "Mesuring Shapelet modes with sigma_arcsec=%.2f arcsec"
                % self.sigma_arcsec
            )
            mtask_s = anacal.fpfs.FpfsMeasure(
                psf_array=psf_array,
                mag_zero=mag_zero,
                pixel_scale=pixel_scale,
                sigma_arcsec=self.sigma_arcsec,
                klim_thres=self.klim_thres,
                nord=self.nord,
                det_nrot=-1,
            )
            src_s = mtask_s.run(
                gal_array=gal_array,
                det=coords,
                noise_array=noise_array,
                psf=psf_obj,
            )
            src_s.write(self.srcs_name)
            del mtask_s, src_s

        # Shapelet Modes
        if (not os.path.isfile(self.srcs2_name)) and (
            self.sigma_arcsec2 is not None
        ):
            print(
                "Mesuring Shapelet modes with sigma_arcsec=%.2f arcsec"
                % self.sigma_arcsec2
            )
            mtask_s2 = anacal.fpfs.FpfsMeasure(
                psf_array=psf_array,
                mag_zero=mag_zero,
                pixel_scale=pixel_scale,
                sigma_arcsec=self.sigma_arcsec2,
                klim_thres=self.klim_thres,
                nord=self.nord,
                det_nrot=-1,
            )
            src_s2 = mtask_s2.run(
                gal_array=gal_array,
                det=coords,
                noise_array=noise_array,
                psf=psf_obj,
            )
            src_s2.write(self.srcs2_name)
            del mtask_s2, src_s2

        # Shapelet Modes (second scale)
        del coords
        gc.collect()
        return

    def prepare_data(self, file_name):
        dm_task = MakeDMExposure(self.config_name)
        # using seeds that are not used in simulation
        seed = dm_task.get_seed_from_fname(file_name, "i") + 1
        exposure = dm_task.generate_exposure(file_name)
        pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
        variance = np.average(exposure.getMaskedImage().variance.array)
        mag_zero = (
            np.log10(exposure.getPhotoCalib().getInstFluxAtZeroMagnitude())
            / 0.4
        )
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
                noise_array = (
                    np.random.RandomState(seed)
                    .normal(
                        scale=noise_std,
                        size=(ny, nx),
                    )
                    .astype(np.float64)
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
                    ).astype(np.float64)
                    * noise_std
                )
        else:
            noise_array = None

        assert self.ncov_fname is not None
        if os.path.isfile(self.ncov_fname):
            cov_matrix = anacal.fpfs.table.Covariance.from_fits(self.ncov_fname)
        else:
            assert self.estimate_cov_matrix
            cov_task = anacal.fpfs.FpfsNoiseCov(
                psf_array=psf_array,
                mag_zero=mag_zero,
                pixel_scale=pixel_scale,
                sigma_arcsec=self.sigma_arcsec,
                nord=self.nord,
                det_nrot=self.det_nrot,
                klim_thres=self.klim_thres,
            )
            cov_matrix = cov_task.measure(variance=variance)
            cov_matrix.write(self.ncov_fname)

        if self.input_cat_dir is not None:
            field_id = int(file_name.split("image-")[-1].split("_")[0])
            tmp_fname = "brightstar-%05d.fits" % field_id
            tmp_fname = os.path.join(self.input_cat_dir, tmp_fname)
            star_cat = fitsio.read(tmp_fname)[["x", "y", "r"]]
            star_cat["r"] = star_cat["r"] * 1.0
        else:
            star_cat = None
        gc.collect()
        return {
            "gal_array": gal_array,
            "psf_array": psf_array,
            "cov_matrix": cov_matrix,
            "pixel_scale": pixel_scale,
            "noise_array": noise_array,
            "psf_obj": psf_obj,
            "mask_array": mask_array,
            "star_cat": star_cat,
        }

    # @profile
    def run(self, file_name):
        data = self.prepare_data(file_name)
        self.get_file_names(file_name)
        start_time = time.time()
        self.process_image(
            gal_array=data["gal_array"],
            psf_array=data["psf_array"],
            cov_matrix=data["cov_matrix"],
            pixel_scale=data["pixel_scale"],
            noise_array=data["noise_array"],
            psf_obj=data["psf_obj"],
            mask_array=data["mask_array"],
            star_cat=data["star_cat"],
        )
        del data
        elapsed_time = time.time() - start_time
        print("Elapsed time: %.2f seconds" % elapsed_time)
        return
