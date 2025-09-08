#!/usr/bin/env python
#
# simple example with ring test (rotating intrinsic galaxies)
# Copyright 20230916 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
import os
from typing import Any

import fitsio
import lsst.pipe.base.connectionTypes as cT
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import ConfigurableField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
)
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.skymap import BaseSkyMap

from ..simulator.multiband import MultibandSimHaloTask, MultibandSimShearTask


class MultibandSimPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch", "band"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "outputCoaddName": "sim",
        "mode": 0,
        "rotId": 0,
    },
):
    skymap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    exposure = cT.Input(
        doc="Input coadd exposure",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=False,
        minimum=0,
    )
    noiseCorrImage = cT.Input(
        doc="image for noise correlation function",
        name="{inputCoaddName}Coadd_systematics_noisecorr",
        dimensions=("skymap", "tract"),
        storageClass="ImageF",
        multiple=False,
        minimum=0,
    )
    psfImage = cT.Input(
        doc="image for PSF model for simulation",
        name="{inputCoaddName}Coadd_systematics_psfcentered",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ImageF",
        multiple=False,
        minimum=0,
    )
    outputExposure = cT.Output(
        doc="Output simulated coadd exposure",
        name="{outputCoaddName}_{mode}_rot{rotId}_Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
    )
    outputTruthCatalog = cT.Output(
        doc="Output truth catalog",
        name="{outputCoaddName}_{mode}_rot{rotId}_Coadd_truthCatalog",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class MultibandSimShearPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=MultibandSimPipeConnections,
):
    simulator = ConfigurableField(
        target=MultibandSimShearTask,
        doc="Simulation task for shear test",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()

    def setDefaults(self):
        super().setDefaults()


class MultibandSimShearPipe(PipelineTask):
    _DefaultName = "MultibandSimShearPipe"
    ConfigClass = MultibandSimShearPipeConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.makeSubtask("simulator")
        fname = os.path.join(
            "/lustre/work/xiangchong.li/work",
            "hsc_s23b_data/sim_v1/success.fits",
        )
        if os.path.isfile(fname):
            self.pt_data = fitsio.read(fname)
        else:
            self.pt_data = None

    def runQuantum(self, butlerQC, inputRefs, outputRefs) -> None:
        assert butlerQC.quantum.dataId is not None
        tract = butlerQC.quantum.dataId["tract"]
        patch = butlerQC.quantum.dataId["patch"]
        # if self.pt_data is not None:
        #     patch_list = self.pt_data[self.pt_data["tract"] == tract]["patch"]
        #     if patch not in patch_list:
        #         print(patch)
        #         print("failed..............")
        #         return

        inputs = butlerQC.get(inputRefs)

        # band name
        assert butlerQC.quantum.dataId is not None
        band = butlerQC.quantum.dataId["band"]
        patch = butlerQC.quantum.dataId["patch"]
        inputs["band"] = band
        inputs["patch"] = patch

        # Get unique integer ID for IdFactory and RNG seeds; only the latter
        # should really be used as the IDs all come from the input catalog.
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        seed = idGenerator.catalog_id
        inputs["seed"] = seed

        skymap = butlerQC.get(inputRefs.skymap)
        sky_info = makeSkyInfo(
            skymap,
            tractId=butlerQC.quantum.dataId["tract"],
            patchId=butlerQC.quantum.dataId["patch"],
        )
        boundaryBox = sky_info.bbox
        inputs["boundaryBox"] = boundaryBox

        # Obtain the WCS for the patch
        tract_info = sky_info.tractInfo
        wcs = tract_info.getWcs()
        inputs["wcs"] = wcs

        outputs = self.simulator.run(**inputs)
        butlerQC.put(outputs, outputRefs)
        return


class MultibandSimHaloPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=MultibandSimPipeConnections,
):
    simulator = ConfigurableField(
        target=MultibandSimHaloTask,
        doc="Multiband Halo Simulation Task",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()

    def setDefaults(self):
        super().setDefaults()


class MultibandSimHaloPipe(MultibandSimShearPipe):
    _DefaultName = "MultibandSimHaloPipe"
    ConfigClass = MultibandSimHaloPipeConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.makeSubtask("simulator")
