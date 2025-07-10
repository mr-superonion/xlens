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
    "RegisterSimPipeConfig",
    "RegisterSimPipe",
    "RegisterSimPipeConnections",
]

import glob
import logging
import os
from typing import Any

import fitsio
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter



class RegisterSimPipeConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates={"coaddName": "deep"},
):
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    outputExposure = cT.Output(
        doc="Input coadd image",
        name="{coaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap"),
    )
    outputCatalog = cT.Output(
        doc=("original measurement catalog"),
        name="{coaddName}Coadd_meas",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class RegisterSimPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=RegisterSimPipeConnections,
):

    def validate(self):
        super().validate()


class RegisterSimPipe(PipelineTask):
    _DefaultName = "FpfsTask"
    ConfigClass = RegisterSimPipeConfig

    def __init__(
        self,
        *,
        config: RegisterSimPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, RegisterSimPipeConfig)

        self.pt_data = fitsio.read(
            os.path.join(
                "/work/xiangchong.li/work/hsc_s23b_sim/catalogs/",
                "tracts_fdfc_v1_trim2_sim.fits",
            )
        )
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, RegisterSimPipeConfig)
        # Retrieve the filename of the input exposure
        assert butlerQC.quantum.dataId is not None
        tract = butlerQC.quantum.dataId["tract"]
        patch = butlerQC.quantum.dataId["patch"]
        patch_list = self.pt_data[self.pt_data["tract"] == tract]["patch"]
        patch_y = int(patch) // 9
        patch_x = int(patch) % 9
        patch_db = patch_x * 100 + patch_y

        if patch_db not in patch_list:
            return

        band = butlerQC.quantum.dataId["band"]
        hsc_dir = "/lustre/HSC_DR/hsc_ssp/dr4/s23b/data/s23b_wide/unified/"

        exp_file_name = glob.glob(
            os.path.join(
                hsc_dir,
                f"deepCoadd_calexp/{tract}/{patch}/{band}/",
                f"deepCoadd_calexp_{tract}_{patch}_{band}_*.fits",
            )
        )
        if len(exp_file_name) > 0:
            exp_file_name = exp_file_name[0]
        else:
            return
            # raise IOError("Cannot find exposure")
        exposure = afwImage.ExposureF.readFits(exp_file_name)
        if exposure.getPsf() is None:
            return

        cat_file_name = glob.glob(
            os.path.join(
                hsc_dir,
                f"deepCoadd_meas/{tract}/{patch}/{band}/",
                f"deepCoadd_meas_{tract}_{patch}_{band}_*.fits",
            )
        )
        if len(cat_file_name) > 0:
            cat_file_name = cat_file_name[0]
        else:
            return
        catalog = afwTable.SourceCatalog.readFits(cat_file_name)
        outputs = Struct(
            outputExposure=exposure,
            outputCatalog=catalog,
        )
        butlerQC.put(outputs, outputRefs)
        return
