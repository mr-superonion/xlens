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
    "FpfsMultibandPipeConfig",
    "FpfsMultibandPipe",
    "FpfsMultibandPipeConnections",
]

import logging
from typing import Any

import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.pipe.base.connectionTypes as cT
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.meas.deblender import SourceDeblendTask
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.utils.logging import LsstLogAdapter

from ..processor.fpfs import FpfsMeasurementTask


class FpfsMultibandPipeConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "outputCoaddName": "deep",
        "dataType": "",
    },
):
    exposure = cT.Input(
        doc="Input coadd image",
        name="{inputCoaddName}Coadd_calexp{dataType}",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=False,
    )
    # inputCatalog = cT.Input(
    #     doc=("original measurement catalog"),
    #     name="{inputCoaddName}Coadd_meas",
    #     storageClass="SourceCatalog",
    #     dimensions=("tract", "patch", "band", "skymap"),
    # )
    outputCatalog = cT.Output(
        doc="Source catalog with all the measurement generated in this task",
        name="{outputCoaddName}Coadd_anacal_meas{dataType}",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class FpfsMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=FpfsMultibandPipeConnections,
):
    do_dm_detection = Field[bool](
        doc="whether to do detection",
        default=False,
    )
    detection = ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect Sources Task",
    )
    deblend = ConfigurableField(
        target=SourceDeblendTask,
        doc="Deblending Task",
    )
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )
    psfCache = Field[int](
        doc="Size of psfCache",
        default=100,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()
        if not self.fpfs.do_adding_noise:
            if len(self.connections.dataType) == 0:
                raise ValueError(
                    "Only set fpfs.do_adding_noise=False on simulation"
                )


class FpfsMultibandPipe(PipelineTask):
    _DefaultName = "FpfsMultibandPipe"
    ConfigClass = FpfsMultibandPipeConfig

    def __init__(
        self,
        *,
        config: FpfsMultibandPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, FpfsMultibandPipeConfig)
        if self.config.do_dm_detection:
            self.schema = afwTable.SourceTable.makeMinimalSchema()
            self.algMetadata = dafBase.PropertyList()
            self.makeSubtask("detection", schema=self.schema)
            self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask("fpfs")
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, FpfsMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        data = self.prepare_data(butlerQC=butlerQC, inputRefs=inputRefs)
        outputs = Struct(outputCatalog=self.fpfs.run(**data))
        butlerQC.put(outputs, outputRefs)
        return

    def prepare_data(
        self,
        *,
        butlerQC,
        inputRefs,
    ):
        assert isinstance(self.config, FpfsMultibandPipeConfig)

        inputs = butlerQC.get(inputRefs)

        # Set psfCache
        # move this to run after gen2 deprecation
        inputs["exposure"].getPsf().setCacheCapacity(self.config.psfCache)

        # Get unique integer ID for IdFactory and RNG seeds; only the latter
        # should really be used as the IDs all come from the input catalog.
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        inputs["seed"] = idGenerator.catalog_id
        inputs["detection"] = None
        inputs["noise_corr"] = None
        return self.fpfs.prepare_data(**inputs)
