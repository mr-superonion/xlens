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
from lsst.pex.config import Field
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
    ename = Field[str](
        doc="ellipticity column name",
        default="e1",
    )

    def validate(self):
        super().validate()
        if len(self.connections.dataType) == 0:
            raise ValueError("connections.dataTape missing")


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

        self.ename = self.config.ename
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
        denom = np.average(down)
        print(
            "Multiplicative bias:",
            np.average(up1) / denom / 0.02 / 2.0 - 1, "+-",
            np.std(up1) / denom / np.sqrt(len(up1)) / 0.02 / 2.0,
        )
        print(
            "Additive bias:",
            np.average(up2) / denom, "+-",
            np.std(up2) / denom / np.sqrt(len(up2)),
        )
        return
