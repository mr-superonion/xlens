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

import lsst.pipe.base.connectionTypes as cT
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
        "coaddName": "deep",
    },
):
    exposure = cT.Input(
        doc="Input coadd image",
        name="{coaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=False,
    )
    detection = cT.Input(
        doc="Source catalog with detection",
        name="{coaddName}Coadd_anacal_detection",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        minimum=0,
        multiple=False,
    )
    catalog = cT.Output(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}Coadd_anacal_meas",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class FpfsMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=FpfsMultibandPipeConnections,
):
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

    def setDefaults(self):
        super().setDefaults()


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
        self.makeSubtask("fpfs")
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, FpfsMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        data = self.prepare_data(butlerQC=butlerQC, inputRefs=inputRefs)
        outputs = Struct(catalog=self.fpfs.run(**data))
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
        inputs["noise_corr"] = None
        return self.fpfs.prepare_data(**inputs)
