# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "HaloMcBiasMultibandPipeConfig",
    "HaloMcBiasMultibandPipe",
    "HaloMcBiasMultibandPipeConnections",
]

import logging
from typing import Any

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.pex.config import Field
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
)
from lsst.utils.logging import LsstLogAdapter
from lsst.skymap import BaseSkyMap

import galsim
from astropy.cosmology import Planck18
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel


class HaloMcBiasMultibandPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "band"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "dataType": "",
    },
):
    skymap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    src00List = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{inputCoaddName}Coadd_anacal_meas{dataType}_0_rot0",
        dimensions=("skymap", "band", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    src01List = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{inputCoaddName}Coadd_anacal_meas{dataType}_0_rot1",
        dimensions=("skymap", "band", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class HaloMcBiasMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=HaloMcBiasMultibandPipeConnections,
):
    ename = Field[str](
        doc="ellipticity column name",
        default="e",
    )

    xname = Field[str](
        doc="detection coordinate row name",
        default="x",
    )

    yname = Field[str](
        doc="detection coordinate column name",
        default="y",
    )
    mass = Field[float](
        doc="halo mass",
        default=5e-14,
    )

    conc = Field[float](
        doc="halo concertration",
        default=1.0,
    )

    z_lens = Field[float](
        doc="halo redshift",
        default=1.0,
    )

    z_source = Field[float](
        doc="source redshift",
        default=None,
    )

    def validate(self):
        super().validate()
        if len(self.connections.dataType) == 0:
            raise ValueError("connections.dataTape missing")


class HaloMcBiasMultibandPipe(PipelineTask):
    _DefaultName = "HaloMcTask"
    ConfigClass = HaloMcBiasMultibandPipeConfig

    def __init__(
        self,
        *,
        config: HaloMcBiasMultibandPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, HaloMcBiasMultibandPipeConfig)

        self.ename = self.config.ename
        self.egname = lambda x, y: self.ename + str(x) + "_" + "g" + str(y)
        self.xname = self.config.xname
        self.yname = self.config.yname
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, HaloMcBiasMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        self.run(**inputs)
        return

    @staticmethod
    def _rotate_spin_2(e1, e2, angle):
        """
        Rotate a spin-2 field by an array of angles (one per e1, e2 pair)
        """
        # Ensure e1, e2, and angle are numpy arrays of the same length
        e1 = np.asarray(e1)
        e2 = np.asarray(e2)
        angle = np.asarray(angle)

        assert (
            e1.shape == e2.shape == angle.shape
        ), "e1, e2, and angle must have the same shape"

        # Create an empty output array for the rotated values
        output = np.zeros((2, len(e1)))

        # Compute cos(2*angle) and sin(2*angle) for each angle
        cos_2angle = np.cos(2 * angle)
        sin_2angle = np.sin(2 * angle)

        # Apply the rotation for each e1, e2 pair
        output[0] = cos_2angle * e1 - sin_2angle * e2  # Rotated e1
        output[1] = sin_2angle * e1 + cos_2angle * e2  # Rotated e2

        return output

    @staticmethod
    def _get_angle_from_pixel(x, y, x_cen, y_cen):
        """
        Get the angle from the pixel coordinates
        the output is in radians between -pi and pi
        """
        return np.arctan2(y - y_cen, x - x_cen)

    @staticmethod
    def _get_radial_shear(
        eT, w, e1, e2, e1_g1, e2_g2, w_g1, w_g2, dist, radial_bin_edges
    ):
        """
        Get the radial shear profile
        """

        n_bins = len(radial_bin_edges) - 1
        shear_list = []
        for i_bin in range(n_bins):
            mask = (dist >= radial_bin_edges[i_bin]) & (
                dist < radial_bin_edges[i_bin + 1]
            )
            e = np.sum(eT[mask] * w[mask])
            # we use the mean of R11 and R22 as an estimator of Rt
            r11 = np.sum(e1_g1[mask] * w[mask] + e1[mask] * w_g1[mask])
            r22 = np.sum(e2_g2[mask] * w[mask] + e2[mask] * w_g2[mask])
            r_t = (r11 + r22) / 2
            shear_list.append(e / r_t)

        assert ~np.any(np.isnan(shear_list)), "shear_list contains NaN values"

        return np.array(shear_list)

    def run(self, skymap, src00List, src01List):

        pixel_scale = skymap.config.pixelScale  # arcsec per pixel
        image_dim = skymap.config.patchInnerDimensions[0]  # in pixels

        max_pixel = np.sqrt(2) * image_dim

        n_bins = 10
        pixel_bin_edges = np.linspace(0, max_pixel, n_bins + 1)
        angular_bin_edges = pixel_bin_edges * pixel_scale
        angular_bin_mids = (angular_bin_edges[1:] + angular_bin_edges[:-1]) / 2

        # get theory shear
        def _get_gt(mass, conc, z_lens, z_source, angular_dist, cosmo):
            lens = LensModel(lens_model_list=["NFW"])
            lens_cosmo = LensCosmo(
                z_lens=z_lens, z_source=z_source, cosmo=cosmo
            )
            pos_lens = galsim.PositionD(0, 0)
            r = galsim.PositionD(angular_dist, 0) - pos_lens
            rs_angle, alpha_rs = lens_cosmo.nfw_physical2angle(M=mass, c=conc)
            kwargs = [{"Rs": rs_angle, "alpha_Rs": alpha_rs}]
            f_xx, f_xy, f_yx, f_yy = lens.hessian(r.x, r.y, kwargs)
            gamma1 = 1.0 / 2 * (f_xx - f_yy)
            gamma2 = f_xy
            kappa = 1.0 / 2 * (f_xx + f_yy)

            g1 = gamma1 / (1 - kappa)
            g2 = gamma2 / (1 - kappa)
            mu = 1.0 / ((1 - kappa) ** 2 - gamma1**2 - gamma2**2)

            return g1, g2
            # if g1**2.0 + g2**2.0 > 0.95:
            #     return gso, shift
            # dra, ddec = self.lens.alpha(r.x, r.y, kwargs)
            # gso = gso.lens(g1=g1, g2=g2, mu=mu)
            # shift = shift + galsim.PositionD(dra, ddec)

        true_gt = []
        true_gx = []
        for bin_mid in angular_bin_mids:
            gt, gx = _get_gt(
                mass=self.config.mass,
                conc=self.config.conc,
                z_lens=self.config.z_lens,
                z_source=self.config.z_source,
                angular_dist=bin_mid,
                cosmo=Planck18,
            )
            true_gt.append(gt)
            true_gx.append(gx)
        true_gt = np.array(true_gt)
        true_gx = np.array(true_gx)

        en = self.ename
        e1n = en + "1"
        e2n = en + "2"

        e1g1n = self.egname(1, 1)
        e2g2n = self.egname(2, 2)

        xn = self.xname
        yn = self.yname

        print("The length of source list is", len(src00List), len(src01List))

        shear_list = np.empty((len(src00List), n_bins))

        for i, src in enumerate(zip(src00List, src01List)):
            src00, src01 = src[0], src[1]
            # get all res first
            sr_00_res = src00.get()
            sr_01_res = src01.get()
            e1 = np.concatenate([sr_00_res[e1n], sr_01_res[e1n]])
            e2 = np.concatenate([sr_00_res[e2n], sr_01_res[e2n]])
            e1_g1 = np.concatenate([sr_00_res[e1g1n], sr_01_res[e1g1n]])
            e2_g2 = np.concatenate([sr_00_res[e2g2n], sr_01_res[e2g2n]])
            w = np.concatenate([sr_00_res["w"], sr_01_res["w"]])
            w_g1 = np.concatenate([sr_00_res["w_g1"], sr_01_res["w_g1"]])
            w_g2 = np.concatenate([sr_00_res["w_g2"], sr_01_res["w_g2"]])
            x = np.concatenate([sr_00_res[xn], sr_01_res[xn]])
            y = np.concatenate([sr_00_res[yn], sr_01_res[yn]])

            angle = self._get_angle_from_pixel(
                x, y, image_dim / 2, image_dim / 2
            )
            # negative since we are rotating axes
            eT, eX = self._rotate_spin_2(e1, e2, -angle)
            # another negaive since we are rotating derivative
            # w_gT, w_gX = self._rotate_spin_2(w_g1, w_g2, angle)
            # w are scalar so no need to rotate
            dist = np.sqrt(x**2 + y**2)

            shear_list[i, :] = self._get_radial_shear(
                eT, w, e1, e2, e1_g1, e2_g2, w_g1, w_g2, dist, pixel_bin_edges
            )

        m_bias_array = shear_list / true_gt - 1
        mean_m_bias = np.mean(m_bias_array, axis=0)
        std_m_bias = np.std(m_bias_array, axis=0)
        print("m_bias", mean_m_bias, "+-", std_m_bias)

        # i_realization += 1
        # src00 = src00.get()
        # src01 = src01.get()
        # src00_dist = np.sqrt(src00[xn] ** 2 + src00[yn] ** 2)
        # src01_dist = np.sqrt(src01[xn] ** 2 + src01[yn] ** 2)

        # ind_shear = np.empty(n_bins)
        # for i_bin in range(len(pixel_bins_edges) - 1):
        #     mask_00 = (src00_dist >= pixel_bins_edges[i_bin]) & (
        #         src00_dist < pixel_bins_edges[i_bin + 1]
        #     )
        #     mask_01 = (src01_dist >= pixel_bins_edges[i_bin]) & (
        #         src01_dist < pixel_bins_edges[i_bin + 1]
        #     )

        #     e = np.sum(src00[en][mask_00] * src00["w"][mask_00]) + np.sum(
        #         src01[en][mask_01] * src01["w"][mask_01]
        #     )
        #     r = np.sum(
        #         src00[egn][mask_00] * src00["w"][mask_00]
        #         + src00[en][mask_00] * src00["w_g1"][mask_00]
        #     ) + np.sum(
        #         src01[egn][mask_01] * src01["w"][mask_01]
        #         + src01[en][mask_01] * src01["w_g1"][mask_01]
        #     )
        #     ind_shear[i_bin] = e / r
        # shear_list.append(ind_shear)

        return
