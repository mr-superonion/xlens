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


class McBiasMultibandPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "band"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "dataType": "",
    },
):
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

        if self.shear_name not in ["g1", "g2"]:
            raise FieldValidationError(
                self.__class__.shear_name,
                self,
                "shear_name can only be 'g1' or 'g2'",
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
        self.run(**inputs)
        return

    def run(self, src00List, src01List, src10List, src11List):
        en = self.ename
        egn = self.egname
        up1 = []
        up2 = []
        down = []
        gt1 = []
        gt2 = []
        print(len(src00List))
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

            g1_00 = src00["e1"].value * src00["w"] / np.average(src00["e1_g1"] * src00["w"] + src00["e1"] * src00["w_g1"])
            g2_00 = src00["e2"] * src00["w"] / np.average(src00["e2_g2"] * src00["w"] + src00["e2"] * src00["w_g2"])
            theta_00 = np.arctan2(src00["y"], src00["x"])
            gt_00 = g1_00 * np.cos(2.0 * theta_00) + g2_00 * np.sin(2.0 * theta_00)

            g1_01 = src01["e1"] * src01["w"] / np.average(src01["e1_g1"] * src01["w"] + src01["e1"] * src01["w_g1"])
            g2_01 = src01["e2"] * src01["w"] / np.average(src01["e2_g2"] * src01["w"] + src01["e2"] * src01["w_g2"])
            theta_01 = np.arctan2(src01["y"], src01["x"])
            gt_01 = g1_01 * np.cos(2.0 * theta_01) + g2_01 * np.sin(2.0 * theta_01)

            gt1.append(np.average(np.concatenate([gt_00, gt_01])))

            g1_10 = src10["e1"] * src10["w"] / np.average(src10["e1_g1"] * src10["w"] + src10["e1"] * src10["w_g1"])
            g2_10 = src10["e2"] * src10["w"] / np.average(src10["e2_g2"] * src10["w"] + src10["e2"] * src10["w_g2"])
            theta_10 = np.arctan2(src10["y"], src10["x"])
            gt_10 = g1_10 * np.cos(2.0 * theta_10) + g2_10 * np.sin(2.0 * theta_10)

            g1_11 = src11["e1"] * src11["w"] / np.average(src11["e1_g1"] * src11["w"] + src11["e1"] * src11["w_g1"])
            g2_11 = src11["e2"] * src11["w"] / np.average(src11["e2_g2"] * src11["w"] + src11["e2"] * src11["w_g2"])
            theta_11 = np.arctan2(src11["y"], src11["x"])
            gt_11 = g1_11 * np.cos(2.0 * theta_11) + g2_11 * np.sin(2.0 * theta_11)

            gt2.append(np.average(np.concatenate([gt_10, gt_11])))


        nsim = len(src00List)
        denom = np.average(down)
        tmp = np.array(up1) / 2.0 + np.array(up2)
        print(gt1)
        gt1 = np.array(gt1)
        gt2 = np.array(gt2)
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
        print(
                "Tangential shear+:",
                np.average(gt1, axis=0),
                "Tangetial shear-:",
                np.average(gt2, axis=0),
                )
        return
