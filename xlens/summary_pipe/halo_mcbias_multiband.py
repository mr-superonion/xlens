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

    def validate(self):
        super().validate()
        if len(self.connections.dataType) == 0:
            raise ValueError("connections.dataTape missing")


class HaloMcBiasMultibandPipe(PipelineTask):
    _DefaultName = "FpfsTask"
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

        en = self.ename
        e1n = en + "1"
        e2n = en + "2"

        e1g1n = self.egname(1, 1)
        e2g2n = self.egname(2, 2)

        xn = self.xname
        yn = self.yname

        pixel_scale = skymap.config.pixelScale  # arcsec per pixel
        image_dim = skymap.config.patchInnerDimensions[0]  # in pixels

        max_pixel = np.sqrt(2) * image_dim

        src00_res = []
        src01_res = []
        for src00, src01 in zip(src00List, src01List):
            # get all res first
            src00_res.append(src00.get())
            src01_res.append(src01.get())

        def _join_list(src00_res, src01_res, key):
            return np.concatenate(
                [src00[key] for src00 in src00_res]
                + [src01[key] for src01 in src01_res]
            )

        e1 = _join_list(src00_res, src01_res, e1n)
        e2 = _join_list(src00_res, src01_res, e2n)
        x = _join_list(src00_res, src01_res, xn)
        y = _join_list(src00_res, src01_res, yn)
        e1_g1 = _join_list(src00_res, src01_res, e1g1n)
        e2_g2 = _join_list(src00_res, src01_res, e2g2n)
        w = _join_list(src00_res, src01_res, "w")
        w_g1 = _join_list(src00_res, src01_res, "w_g1")
        w_g2 = _join_list(src00_res, src01_res, "w_g2")

        print(e1.shape, e2.shape, x.shape, y.shape)

        angle = self._get_angle_from_pixel(x, y, image_dim / 2, image_dim / 2)
        angle = angle - np.pi / 2  # need the tangential shear to be positive
        # negative since we are rotating axes
        eT, eX = self._rotate_spin_2(e1, e2, -angle)
        # another negaive since we are rotating derivative
        # w_gT, w_gX = self._rotate_spin_2(w_g1, w_g2, angle)
        # w are scalar so no need to rotate

        dist = np.sqrt(x**2 + y**2)

        n_bins = 4
        pixel_bin_edges = np.linspace(0, max_pixel, n_bins + 1)
        angular_bin_edges = pixel_bin_edges * pixel_scale

        shear_list = self._get_radial_shear(
            eT, w, e1, e2, e1, e2, w_g1, w_g2, dist, pixel_bin_edges
        )

        print(shear_list)

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
