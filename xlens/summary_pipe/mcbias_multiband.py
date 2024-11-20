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
from typing import Any

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.pex.config import Field, FieldValidationError
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
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
        if self.sname[-1] in ["t", "x"]:
            self.run_tx(**inputs)
        else:
            self.run_12(**inputs)
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
    def _compute_additive_bias(ep, em, Rp, Rm):
        return (ep + em) / (Rp + Rm)


    def run_tx(self, skymap, src00List, src01List, src10List, src11List):
        image_dim = skymap.config.patchInnerDimensions[0] # in pixels
        print('running tx')
        up1 = []
        up2 = []
        down = []

        eTp_list = []
        eXp_list = []
        RTp_list = []
        RXp_list = []
        eTm_list = []
        eXm_list = []
        RTm_list = []
        RXm_list = []

        for src00, src01, src10, src11 in zip(
            src00List, src01List, src10List, src11List
        ):
            src00 = src00.get()
            src01 = src01.get()
            src10 = src10.get()
            src11 = src11.get()

            xp = np.concatenate([src00["x"], src01["x"]])
            yp = np.concatenate([src00["y"], src01["y"]])
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

            eTp_sum = np.sum(eTp * wp)
            eXp_sum = np.sum(eXp * wp)
            RTp_sum = np.sum(RTp)
            RXp_sum = np.sum(RXp)

            xm = np.concatenate([src10["x"], src11["x"]])
            ym = np.concatenate([src10["y"], src11["y"]])
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

            eTm_sum = np.sum(eTm * wm)
            eXm_sum = np.sum(eXm * wm)
            RTm_sum = np.sum(RTm)
            RXm_sum = np.sum(RXm)

            eTp_list.append(eTp_sum)
            eXp_list.append(eXp_sum)
            RTp_list.append(RTp_sum)
            RXp_list.append(RXp_sum)
            eTm_list.append(eTm_sum)
            eXm_list.append(eXm_sum)
            RTm_list.append(RTm_sum)
            RXm_list.append(RXm_sum)

        eTp = np.array(eTp_list)
        eXp = np.array(eXp_list)
        RTp = np.array(RTp_list)
        RXp = np.array(RXp_list)
        eTm = np.array(eTm_list)
        eXm = np.array(eXm_list)
        RTm = np.array(RTm_list)
        RXm = np.array(RXm_list)

        nsim = len(src00List)
        print(
            "Positive tangential shear:",
            np.average(eTp) / np.average(RTp),
            "+-",
            np.std(eTp) / np.average(RTp) / np.sqrt(nsim),
        )
        print(
            "Negative tangential shear:",
            np.average(eTm) / np.average(RTm),
            "+-",
            np.std(eTm) / np.average(RTm) / np.sqrt(nsim),
        )
        print(
            "Positive cross shear:",
            np.average(eXp) / np.average(RXp),
            "+-",
            np.std(eXp) / np.average(RXp) / np.sqrt(nsim),
        )
        print(
            "Negative cross shear:",
            np.average(eXm) / np.average(RXm),
            "+-",
            np.std(eXm) / np.average(RXm) / np.sqrt(nsim),
        )
        print(
            "Multiplicative bias:",
            np.average(eTp - eTm) / np.average(RTp + RTm) / self.svalue / 2.0 - 1,
            "+-",
            np.std(eTp - eTm) / np.average(RTp + RTm) / np.sqrt(nsim) / self.svalue / 2.0,
        )
        print(
            "Tangential additive bias:",
            np.average(eTp + eTm) / np.average(RTp + RTm),
            "+-",
            np.std(eTp + eTm) / np.average(RTp + RTm) / np.sqrt(nsim),
        )
        print(
            "Cross additive bias:",
            np.average(eXp + eXm) / np.average(RXp + RXm),
            "+-",
            np.std(eXp + eXm) / np.average(RXp + RXm) / np.sqrt(nsim),
        )
        return

    def run_12(self, skymap, src00List, src01List, src10List, src11List):
        en = self.ename
        egn = self.egname
        up1 = []
        up2 = []
        down = []
        for src00, src01, src10, src11 in zip(
            src00List, src01List, src10List, src11List
        ):
            src00 = src00.get()
            src01 = src01.get()
            src10 = src10.get()
            src11 = src11.get()
            em = np.sum(src00[en] * src00["w"]) + np.sum(src01[en] * src01["w"])
            ep = np.sum(src10[en] * src10["w"]) + np.sum(src11[en] * src11["w"])
            rm = np.sum(
                src00[egn] * src00["w"] + src00[en] * src00["w_g1"]
            ) + np.sum(src01[egn] * src01["w"] + src01[en] * src01["w_g1"])
            rp = np.sum(
                src10[egn] * src10["w"] + src10[en] * src10["w_g1"]
            ) + np.sum(src11[egn] * src11["w"] + src11[en] * src11["w_g1"])

            up1.append(ep - em)
            up2.append((em + ep) / 2.0)
            down.append((rm + rp) / 2.0)

        nsim = len(src00List)
        denom = np.average(down)
        tmp = np.array(up1) / 2.0 + np.array(up2)
        print(
            "Positive shear:",
            np.average(tmp) / denom,
            "+-",
            np.std(tmp) / denom / np.sqrt(nsim),
        )
        tmp = -np.array(up1) / 2.0 + np.array(up2)
        print(
            "Negative shear:",
            np.average(tmp) / denom,
            "+-",
            np.std(tmp) / denom / np.sqrt(nsim),
        )
        if self.sname[-1] == self.ename[-1]:
            print(
                "Multiplicative bias:",
                np.average(up1) / denom / self.svalue / 2.0 - 1,
                "+-",
                np.std(up1) / denom / np.sqrt(nsim) / self.svalue / 2.0,
            )
        else:
            print(
                "We do not estimate multiplicative bias:",
            )
        print(
            "Additive bias:",
            np.average(up2) / denom,
            "+-",
            np.std(up2) / denom / np.sqrt(nsim),
        )
        return
