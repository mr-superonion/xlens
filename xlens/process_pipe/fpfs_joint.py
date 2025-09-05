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
    "FpfsJointPipeConfig",
    "FpfsJointPipe",
    "FpfsJointPipeConnections",
]

import logging
from typing import Any

import lsst.pipe.base.connectionTypes as cT
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.utils.logging import LsstLogAdapter

from ..processor.fpfs import FpfsMeasurementTask


class FpfsJointPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "coaddName": "deep",
    },
):
    exposure = cT.Input(
        doc="Input coadd image",
        name="{coaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        deferLoad=True,
    )
    noise_corr = cT.Input(
        doc="noise correlation function",
        name="{coaddName}Coadd_systematics_noisecorr",
        storageClass="ImageF",
        dimensions=("skymap", "tract"),
        minimum=0,
        multiple=True,
        deferLoad=True,
    )
    joint_catalog = cT.Output(
        doc="Source catalog with joint detection and measurement",
        name="{coaddName}Coadd_anacal_joint",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class FpfsJointPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=FpfsJointPipeConnections,
):
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )
    psfCache = Field[int](
        doc="Size of PSF cache",
        default=100,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()
    use_truth_detection = Field[bool](
        doc="whether to use truth catalog as detection",
        default=False,
    )
    use_dm_detection = Field[bool](
        doc="whether to use dm catalog as detection",
        default=False,
    )

    def validate(self):
        super().validate()

    def setDefaults(self):
        super().setDefaults()
        self.fpfs.sigma_arcsec1 = -1
        self.fpfs.sigma_arcsec2 = -1
        self.fpfs.do_compute_detect_weight = True


class FpfsJointPipe(PipelineTask):
    _DefaultName = "FpfsJointPipe"
    ConfigClass = FpfsJointPipeConfig

    def __init__(
        self,
        *,
        config: FpfsJointPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, FpfsJointPipeConfig)
        self.makeSubtask("fpfs")
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, FpfsJointPipeConfig)
        inputs = butlerQC.get(inputRefs)
        exposure_handles = inputs["exposure"]
        exposure_handles_dict = {
            handle.dataId["band"]: handle for handle in exposure_handles
        }
        correlation_handles = inputs["noise_corr"]
        if len(correlation_handles) == 0:
            correlation_handles_dict = None
        else:
            correlation_handles_dict = {
                handle.dataId["band"]: handle for handle in correlation_handles
            }
        outputs = self.run(
            exposure_handles_dict=exposure_handles_dict,
            correlation_handles_dict=correlation_handles_dict,
        )
        butlerQC.put(outputs, outputRefs)
        return

    def run(
        self,
        *,
        exposure_handles_dict: dict,
        correlation_handles_dict: dict | None,
    ):
        assert isinstance(self.config, FpfsJointPipeConfig)
        band = "i"
        handle = exposure_handles_dict[band]
        exposure = handle.get()
        exposure.getPsf().setCacheCapacity(self.config.psfCache)
        if correlation_handles_dict is not None:
            handle = correlation_handles_dict[band]
            noise_corr = handle.get()
        else:
            noise_corr = None

        idGenerator = self.config.idGenerator.apply(handle.dataId)
        seed = idGenerator.catalog_id
        data = self.fpfs.prepare_data(
            band=band,
            exposure=exposure,
            seed=seed,
            noise_corr=noise_corr,
            detection=None,
        )
        catalog = self.fpfs.run(**data)
        return Struct(joint_catalog=catalog)
