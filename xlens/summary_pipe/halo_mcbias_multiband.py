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
        default="e1",
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
        self.egname = self.ename + "_g" + self.ename[-1]
        self.xname = self.config.xname
        self.yname = self.config.yname
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, HaloMcBiasMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        self.run(**inputs)
        return

    def run(self, skymap, src00List, src01List):

        en = self.ename
        egn = self.egname
        xn = self.xname
        yn = self.yname
        
        print("skymap is passed",skymap)
        

        pixel_scale = skymap.config.pixelScale # arcsec per pixel
        max_pixel = np.sqrt(2) * image_dim
        image_dim = skymap.config.patchInnerDimensions[0] # in pixels
        n_bins = 4
        pixel_bins_edges = np.linspace(0, max_pixel, n_bins + 1)
        # angular_bins_edges = pixel_bins_edges * pixel_scale

        shear_list = []

        i_realization = 0
        for src00, src01 in zip(src00List, src01List):
            # loop though all realizations
            i_realization += 1
            src00 = src00.get()
            src01 = src01.get()
            src00_dist = np.sqrt(src00[xn] ** 2 + src00[yn] ** 2)
            src01_dist = np.sqrt(src01[xn] ** 2 + src01[yn] ** 2)

            ind_shear = np.empty(n_bins)
            for i_bin in range(len(pixel_bins_edges) - 1):
                mask_00 = (src00_dist >= pixel_bins_edges[i_bin]) & (
                    src00_dist < pixel_bins_edges[i_bin + 1]
                )
                mask_01 = (src01_dist >= pixel_bins_edges[i_bin]) & (
                    src01_dist < pixel_bins_edges[i_bin + 1]
                )

                e = np.sum(src00[en][mask_00] * src00["w"][mask_00]) + np.sum(
                    src01[en][mask_01] * src01["w"][mask_01]
                )
                r = np.sum(
                    src00[egn][mask_00] * src00["w"][mask_00]
                    + src00[en][mask_00] * src00["w_g1"][mask_00]
                ) + np.sum(
                    src01[egn][mask_01] * src01["w"][mask_01]
                    + src01[en][mask_01] * src01["w_g1"][mask_01]
                )
                ind_shear[i_bin] = e / r
            shear_list.append(ind_shear)

        print(np.mean(np.array(shear_list), axis=0))

        return
