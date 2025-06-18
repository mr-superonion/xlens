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
    "AnacalForcePipeConfig",
    "AnacalForcePipe",
    "AnacalForcePipeConnections",
]

import logging
from typing import Any

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn

from ..processor.anacal import AnacalTask


class AnacalForcePipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "coaddName": "deep",
    },
):
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    input_catalog = cT.Input(
        doc="Source catalog with joint detection and measurement",
        name="{coaddName}Coadd_anacal_joint",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=False,
    )
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
        name="deepCoadd_systematics_noisecorr",
        storageClass="ImageF",
        dimensions=("skymap", "tract", "patch", "band"),
        minimum=0,
        multiple=True,
        deferLoad=True,
    )
    catalog = cT.Output(
        doc="Source catalog with joint detection and measurement",
        name="{coaddName}Coadd_anacal_force",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class AnacalForcePipeConfig(
    PipelineTaskConfig,
    pipelineConnections=AnacalForcePipeConnections,
):
    anacal = ConfigurableField(
        target=AnacalTask,
        doc="AnaCal Task Force",
    )
    psfCache = Field[int](
        doc="Size of PSF cache",
        default=100,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()

    def setDefaults(self):
        super().setDefaults()
        self.anacal.force_size = True
        self.anacal.force_center = True


class AnacalForcePipe(PipelineTask):
    _DefaultName = "AnacalForcePipe"
    ConfigClass = AnacalForcePipeConfig

    def __init__(
        self,
        *,
        config: AnacalForcePipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, AnacalForcePipeConfig)
        self.makeSubtask("anacal")
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, AnacalForcePipeConfig)
        inputs = butlerQC.get(inputRefs)
        tract = butlerQC.quantum.dataId["tract"]
        patch = butlerQC.quantum.dataId["patch"]
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
        skyMap = inputs["skyMap"]
        detection = inputs["input_catalog"].as_array()
        outputs = self.run(
            detection=detection,
            exposure_handles_dict=exposure_handles_dict,
            correlation_handles_dict=correlation_handles_dict,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
        )
        butlerQC.put(outputs, outputRefs)
        return

    def run(
        self,
        *,
        detection,
        exposure_handles_dict: dict,
        correlation_handles_dict: dict | None,
        skyMap,
        tract: int,
        patch: int,
    ):
        assert isinstance(self.config, AnacalForcePipeConfig)
        colnames = ["flux", "dflux_dg1", "dflux_dg2", "dflux_dj1", "dflux_dj2"]
        catalog = [detection]
        for band in exposure_handles_dict.keys():
            handle = exposure_handles_dict[band]
            exposure = handle.get()
            exposure.getPsf().setCacheCapacity(self.config.psfCache)
            if correlation_handles_dict is not None:
                noise_corr = correlation_handles_dict[band].get().getArray()
                variance = np.amax(noise_corr)
                noise_corr = noise_corr / variance
                ny, nx = noise_corr.shape
                assert noise_corr[ny // 2, nx // 2] == 1
                self.log.debug("With correlation, variance:", variance)
            else:
                noise_corr = None
            idGenerator = self.config.idGenerator.apply(handle.dataId)
            seed = idGenerator.catalog_id
            data = self.anacal.prepare_data(
                exposure=exposure,
                seed=seed,
                noise_corr=noise_corr,
                detection=detection,
                band=band,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
            )
            cat = rfn.repack_fields(self.anacal.run(**data)[colnames])
            map_dict = {name: f"{band}_" + name for name in colnames}
            renamed = rfn.rename_fields(cat, map_dict)
            catalog.append(renamed)
            del exposure, data
        catalog = rfn.merge_arrays(catalog, flatten=True)
        return Struct(catalog=catalog)
