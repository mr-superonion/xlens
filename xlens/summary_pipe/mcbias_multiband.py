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
        "coaddName": "deep",
        "dataType": "",
    },
):
    src00_list = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_0_rot0_Coadd_anacal_{dataType}",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    src01_list = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_0_rot1_Coadd_anacal_{dataType}",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    src10_list = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_1_rot0_Coadd_anacal_{dataType}",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    src11_list = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_1_rot1_Coadd_anacal_{dataType}",
        dimensions=("skymap", "tract", "patch", "band"),
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
        default="fpfs_e1",
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
            raise ValueError("connections.dataType missing")

        if self.shear_name not in ["g1", "g2"]:
            raise FieldValidationError(
                self.__class__.shear_name,
                self,
                "shear_name can only be 'g1' or 'g2'",
            )

        if self.shear_value < 0.0 or self.shear_value > 0.10:
            raise FieldValidationError(
                self.__class__.shear_value,
                self,
                "shear_value should be in [0.00, 0.10]",
            )


def name_add_d(ename, nchars):
    return ename[:-nchars] + "d" + ename[-nchars:]


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

        self.svalue = self.config.shear_value
        self.sname = self.config.shear_name
        self.ename = self.config.shape_name
        ins = "_dg" + self.ename[-1]
        self.egname = name_add_d(self.ename, 2) + ins
        self.wname = "fpfs_w"
        self.wgname = name_add_d(self.wname, 1) + ins
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, McBiasMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        self.run(**inputs)
        return

    def run(self, src00_list, src01_list, src10_list, src11_list):
        en = self.ename
        egn = self.egname
        wn = self.wname
        wgn = self.wgname
        up1 = []
        up2 = []
        down = []
        for src00, src01, src10, src11 in zip(
            src00_list, src01_list, src10_list, src11_list
        ):
            src00 = src00.get()
            src01 = src01.get()
            src10 = src10.get()
            src11 = src11.get()
            em = np.sum(src00[en] * src00[wn]) + np.sum(src01[en] * src01[wn])
            ep = np.sum(src10[en] * src10[wn]) + np.sum(src11[en] * src11[wn])
            rm = np.sum(
                src00[egn] * src00[wn] + src00[en] * src00[wgn]
            ) + np.sum(src01[egn] * src01[wn] + src01[en] * src01[wgn])
            rp = np.sum(
                src10[egn] * src10[wn] + src10[en] * src10[wgn]
            ) + np.sum(src11[egn] * src11[wn] + src11[en] * src11[wgn])

            up1.append(ep - em)
            up2.append((em + ep) / 2.0)
            down.append((rm + rp) / 2.0)
        nsim = len(src00_list)
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
