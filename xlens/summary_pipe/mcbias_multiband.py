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
    "McBiasMultibandPipeConfig",
    "McBiasMultibandPipe",
    "McBiasMultibandPipeConnections",
]

import logging
logger = logging.getLogger(__name__)
from typing import Any

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.pex.config import Field, FieldValidationError
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.utils.logging import LsstLogAdapter
from lsst.skymap import BaseSkyMap


class McBiasMultibandPipeConnections(
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

    src10List = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{inputCoaddName}Coadd_anacal_meas{dataType}_1_rot0",
        dimensions=("skymap", "band", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    src11List = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{inputCoaddName}Coadd_anacal_meas{dataType}_1_rot1",
        dimensions=("skymap", "band", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    outputSummary = cT.Output(
        doc="Summary of the results",
        name="{inputCoaddName}_summary_stats{dataType}",
        storageClass="ArrowAstropy",
        dimensions=("skymap",),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class McBiasMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=McBiasMultibandPipeConnections,
):
    shape_name = Field[str](
        doc="ellipticity column name",
        default="e1",
    )
    shear_name = Field[str](
        doc="the shear component to test",
        default="g1",
    )
    shear_value = Field[float](
        doc="absolute value of the shear",
        default=0.02,
    )

    def validate(self):
        super().validate()
        if len(self.connections.dataType) == 0:
            raise ValueError("connections.dataTape missing")

        if self.shear_name not in ["g1", "g2", "gt", "gx"]:
            raise FieldValidationError(
                self.__class__.shear_name,
                self,
                "shear_name can only be 'g1', 'g2', 'gt' or 'gx'",
            )

        if self.shape_name not in ["q1", "q2", "e1", "e2"]:
            raise FieldValidationError(
                self.__class__.shear_name,
                self,
                "shape_name can only be 'e1', 'e2', 'q1' or 'q2'",
            )

        if self.shear_value < 0.0 or self.shear_value > 0.10:
            raise FieldValidationError(
                self.__class__.shear_value,
                self,
                "shear_value should be in [0.00, 0.10]",
            )


class McBiasMultibandPipe(PipelineTask):
    _DefaultName = "FpfsTask"
    ConfigClass = McBiasMultibandPipeConfig

    def __init__(
        self,
        *,
        config: McBiasMultibandPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, McBiasMultibandPipeConfig)

        self.ename = self.config.shape_name
        self.sname = self.config.shear_name
        self.svalue = self.config.shear_value
        self.egname = self.ename + "_g" + self.ename[-1]
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, McBiasMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)
        return

    @staticmethod
    def _rotate_spin_2_vec(e1, e2, angle):
        e1 = np.asarray(e1)
        e2 = np.asarray(e2)
        angle = np.asarray(angle)

        output = np.zeros((2, len(e1)))
        cos_2angle = np.cos(2*angle)
        sin_2angle = np.sin(2*angle)
        output[0] = (cos_2angle * e1 - sin_2angle * e2) * (-1)
        output[1] = (sin_2angle * e1 + cos_2angle * e2) * (-1)
        return output
    
    @staticmethod
    def _get_angle_from_pixel(x, y, x_cen, y_cen):
        return np.arctan2(y - y_cen, x - x_cen)

    @staticmethod
    def _get_response_from_w_and_der(e1, e2, w, e1_g1, e2_g2, w_g1, w_g2):
        R11 = e1_g1 * w + e1 * w_g1
        R22 = e2_g2 * w + e2 * w_g2
        return R11, R22

    @staticmethod
    def _rotate_spin_2_matrix(R11, R22, angle):
        output = np.zeros((2, len(R11)))
        output[0] = np.cos(2*angle) ** 2 * R11 + np.sin(2*angle) ** 2 * R22
        output[1] = np.sin(2*angle) ** 2 * R11 + np.cos(2*angle) ** 2 * R22
        return output

    @staticmethod
    def _get_eT_eX_rT_rX_sum(eT, eX, w, rT, rX, dist, radial_bin_edges):
        n_bins = len(radial_bin_edges) - 1
        eT_list = []
        eX_list = []
        rT_list = []
        rX_list = []
        eT_std_list = []
        eX_std_list = []
        ngal_in_bin = []

        for i in range(n_bins):
            mask = (dist >= radial_bin_edges[i]) & (
                    dist < radial_bin_edges[i + 1])
            eT_sum = np.sum(eT[mask] * w[mask])
            eX_sum = np.sum(eX[mask] * w[mask])
            rT_sum = np.sum(rT[mask])
            rX_sum = np.sum(rX[mask])

            eT_list.append(eT_sum)
            eX_list.append(eX_sum)
            rT_list.append(rT_sum)
            rX_list.append(rX_sum)
            eT_std_list.append(np.std(eT[mask]))
            eX_std_list.append(np.std(eX[mask]))
            ngal_in_bin.append(np.sum(mask))

        return (
                np.array(eT_list),
                np.array(eX_list),
                np.array(rT_list),
                np.array(rX_list),
                np.array(eT_std_list),
                np.array(eX_std_list),
                np.array(ngal_in_bin),
                )

    @staticmethod
    def get_summary_struct(n_bins):
        dt = [
            ("angular_bin", f"({n_bins},)f8"),
            ("gT+", f"({n_bins},)f8"),
            ("gT+_std", f"({n_bins},)f8"),
            ("gX+", f"({n_bins},)f8"),
            ("gX+_std", f"({n_bins},)f8"),
            ("gT-", f"({n_bins},)f8"),
            ("gT-_std", f"({n_bins},)f8"),
            ("gX-", f"({n_bins},)f8"),
            ("gX-_std", f"({n_bins},)f8"),
            ("m_T", f"({n_bins},)f8"),
            ("m_T_std", f"({n_bins},)f8"),
            ("m_X", f"({n_bins},)f8"),
            ("m_X_std", f"({n_bins},)f8"),
            ("c_T", f"({n_bins},)f8"),
            ("c_T_std", f"({n_bins},)f8"),
            ("c_X", f"({n_bins},)f8"),
            ("c_X_std", f"({n_bins},)f8"),
            ("ngal_in_binp", f"({n_bins},)f8"),
            ("ngal_in_binm", f"({n_bins},)f8"),
        ]
        return np.zeros(1, dtype=dt)

    def run(self, skymap, src00List, src01List, src10List, src11List):
        n_realization = len(src00List)
        logger.info("n_realization", n_realization)

        pixel_scale = skymap.config.pixelScale # arcsec per pixel
        image_dim = skymap.config.patchInnerDimensions[0] # in pixels
        max_pixel = (image_dim - 40) / 2

        logger.info("image dim", image_dim)
        logger.info("pixel scale", pixel_scale)
        logger.info("max pixel", max_pixel)

        n_bins = 10
        pixel_bin_edges = np.linspace(0, max_pixel, n_bins + 1)
        angular_bin_edges = pixel_bin_edges * pixel_scale
        angular_bin_mids = (angular_bin_edges[1:] + angular_bin_edges[:-1]) / 2

        eTp_ensemble = np.empty((n_realization, n_bins))
        eXp_ensemble = np.empty((n_realization, n_bins))
        RTp_ensemble = np.empty((n_realization, n_bins))
        RXp_ensemble = np.empty((n_realization, n_bins))
        ngal_in_binp = np.empty((n_realization, n_bins))

        eTm_ensemble = np.empty((n_realization, n_bins))
        eXm_ensemble = np.empty((n_realization, n_bins))
        RTm_ensemble = np.empty((n_realization, n_bins))
        RXm_ensemble = np.empty((n_realization, n_bins))
        ngal_in_binm = np.empty((n_realization, n_bins))


        for i, (src00, src01, src10, src11) in enumerate(zip(
            src00List, src01List, src10List, src11List
        )):
            src00 = src00.get()
            src01 = src01.get()
            src10 = src10.get()
            src11 = src11.get()

            xp = np.concatenate([src00["x"], src01["x"]])
            yp = np.concatenate([src00["y"], src01["y"]])
            distp = np.sqrt((xp - image_dim/2) ** 2 + (yp - image_dim/2) ** 2)
            anglep = self._get_angle_from_pixel(xp, yp, image_dim/2, image_dim/2)
            e1p = np.concatenate([src00["e1"], src01["e1"]])
            e2p = np.concatenate([src00["e2"], src01["e2"]])
            e1_g1p = np.concatenate([src00["e1_g1"], src01["e1_g1"]])
            e2_g2p = np.concatenate([src00["e2_g2"], src01["e2_g2"]])
            wp = np.concatenate([src00["w"], src01["w"]])
            w_g1p = np.concatenate([src00["w_g1"], src01["w_g1"]])
            w_g2p = np.concatenate([src00["w_g2"], src01["w_g2"]])

            eTp, eXp = self._rotate_spin_2_vec(e1p, e2p, -anglep)
            R1p, R2p = self._get_response_from_w_and_der(
                    e1p, e2p, wp, e1_g1p, e2_g2p, w_g1p, w_g2p
                    )
            RTp, RXp = self._rotate_spin_2_matrix(R1p, R2p, anglep)

            (
                eT_list,
                eX_list,
                rT_list,
                rX_list,
                eT_std_list,
                eX_std_list,
                ngal_in_bin,
            ) = self._get_eT_eX_rT_rX_sum(
                eTp, eXp, wp, RTp, RXp, distp, pixel_bin_edges
            )

            eTp_ensemble[i, :] = eT_list
            eXp_ensemble[i, :] = eX_list
            RTp_ensemble[i, :] = rT_list
            RXp_ensemble[i, :] = rX_list
            ngal_in_binp[i, :] = ngal_in_bin

            xm = np.concatenate([src10["x"], src11["x"]])
            ym = np.concatenate([src10["y"], src11["y"]])
            distm = np.sqrt((xm - image_dim/2) ** 2 + (ym - image_dim/2) ** 2)
            anglem = self._get_angle_from_pixel(xm, ym, image_dim/2, image_dim/2)
            e1m = np.concatenate([src10["e1"], src11["e1"]])
            e2m = np.concatenate([src10["e2"], src11["e2"]])
            e1_g1m = np.concatenate([src10["e1_g1"], src11["e1_g1"]])
            e2_g2m = np.concatenate([src10["e2_g2"], src11["e2_g2"]])
            wm = np.concatenate([src10["w"], src11["w"]])
            w_g1m = np.concatenate([src10["w_g1"], src11["w_g1"]])
            w_g2m = np.concatenate([src10["w_g2"], src11["w_g2"]])

            eTm, eXm = self._rotate_spin_2_vec(e1m, e2m, -anglem)
            R1m, R2m = self._get_response_from_w_and_der(
                    e1m, e2m, wm, e1_g1m, e2_g2m, w_g1m, w_g2m
                    )
            RTm, RXm = self._rotate_spin_2_matrix(R1m, R2m, anglem)

            (
                eT_list,
                eX_list,
                rT_list,
                rX_list,
                eT_std_list,
                eX_std_list,
                ngal_in_bin,
            ) = self._get_eT_eX_rT_rX_sum(
                eTm, eXm, wm, RTm, RXm, distm, pixel_bin_edges
            )

            eTm_ensemble[i, :] = eT_list
            eXm_ensemble[i, :] = eX_list
            RTm_ensemble[i, :] = rT_list
            RXm_ensemble[i, :] = rX_list
            ngal_in_binm[i, :] = ngal_in_bin

        denom_T = np.average(RTp_ensemble + RTm_ensemble, axis=0) / 2.0
        denom_X = np.average(RXp_ensemble + RXm_ensemble, axis=0) / 2.0
        summary_stats = self.get_summary_struct(len(angular_bin_edges) - 1)
        summary_stats["angular_bin"] = angular_bin_mids
        summary_stats["gT+"] = np.average(eTp_ensemble, axis=0) / denom_T
        summary_stats["gT+_std"] = np.std(eTp_ensemble, axis=0) / denom_T / np.sqrt(n_realization)
        summary_stats["gX+"] = np.average(eXp_ensemble, axis=0) / denom_X
        summary_stats["gX+_std"] = np.std(eXp_ensemble, axis=0) / denom_X / np.sqrt(n_realization)
        summary_stats["gT-"] = np.average(eTm_ensemble, axis=0) / denom_T
        summary_stats["gT-_std"] = np.std(eTm_ensemble, axis=0) / denom_T / np.sqrt(n_realization)
        summary_stats["gX-"] = np.average(eXm_ensemble, axis=0) / denom_X
        summary_stats["gX-_std"] = np.std(eXm_ensemble, axis=0) / denom_X / np.sqrt(n_realization)  

        if self.sname[-1] == "t":
            summary_stats["m_T"] =  np.average(eTp_ensemble - eTm_ensemble, axis=0) / denom_T / self.svalue / 2.0 - 1
            summary_stats["m_T_std"] = np.std(eTp_ensemble - eTm_ensemble, axis=0) / denom_T / np.sqrt(n_realization) / self.svalue / 2.0
        else:
            summary_stats["m_X"] =  np.average(eXp_ensemble - eXm_ensemble, axis=0) / denom_X / self.svalue / 2.0 - 1
            summary_stats["m_X_std"] = np.std(eXp_ensemble - eXm_ensemble, axis=0) / denom_X / np.sqrt(n_realization) / self.svalue / 2.0

        summary_stats["c_T"] = np.average(eTp_ensemble + eTm_ensemble, axis=0) / denom_T
        summary_stats["c_T_std"] = np.std(eTp_ensemble + eTm_ensemble, axis=0) / denom_T / np.sqrt(n_realization)
        summary_stats["c_X"] = np.average(eXp_ensemble + eXm_ensemble, axis=0) / denom_X
        summary_stats["c_X_std"] = np.std(eXp_ensemble + eXm_ensemble, axis=0) / denom_X / np.sqrt(n_realization)

        summary_stats["ngal_in_binp"] = np.average(ngal_in_binp, axis=0)
        summary_stats["ngal_in_binm"] = np.average(ngal_in_binm, axis=0)
        return Struct(outputSummary=summary_stats)
