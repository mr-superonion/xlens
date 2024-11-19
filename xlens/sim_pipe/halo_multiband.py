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
from typing import Any

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

from ..simulator.multiband import MultibandSimHaloTask


class MultibandSimHaloPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch", "band"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "outputCoaddName": "sim",
        "mode": 0,
        "irot": 0,
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
        name="{outputCoaddName}Coadd_systematics_noisecorr",
        dimensions=("tract", "patch", "band", "skymap"),
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
        name="{outputCoaddName}_{mode}_rot{irot}_Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
    )
    outputTruthCatalog = cT.Output(
        doc="Output truth catalog",
        name="{outputCoaddName}_{mode}_rot{irot}_Coadd_truthCatalog",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class MultibandSimHaloPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=MultibandSimHaloPipeConnections,
):
    simulator = ConfigurableField(
        target=MultibandSimHaloTask,
        doc="Multiband Halo Simulation Task",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()

    def setDefaults(self):
        print("Configuration starts")
        super().setDefaults()


class MultibandSimHaloPipe(PipelineTask):
    _DefaultName = "MultibandSimHaloPipe"
    ConfigClass = MultibandSimHaloPipeConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert isinstance(self.config, MultibandSimHaloPipeConfig)
        self.makeSubtask("simulator")

    def runQuantum(self, butlerQC, inputRefs, outputRefs) -> None:
        assert isinstance(self.config, MultibandSimHaloPipeConfig)

        # band name
        assert butlerQC.quantum.dataId is not None
        band = butlerQC.quantum.dataId["band"]

        # Get unique integer ID for IdFactory and RNG seeds; only the latter
        # should really be used as the IDs all come from the input catalog.
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        seed = idGenerator.catalog_id

        psfImage = butlerQC.get(inputRefs.psfImage)

        skymap = butlerQC.get(inputRefs.skymap)
        sky_info = makeSkyInfo(
            skymap,
            tractId=butlerQC.quantum.dataId["tract"],
            patchId=butlerQC.quantum.dataId["patch"],
        )
        boundaryBox = sky_info.bbox

        # Obtain the WCS for the patch
        tract_info = sky_info.tractInfo
        wcs = tract_info.getWcs()

        exposure = butlerQC.get(inputRefs.exposure)
        noiseCorr = butlerQC.get(inputRefs.noiseCorrImage)

        outputs = self.simulator.run(
            band=band,
            seed=seed,
            boundaryBox=boundaryBox,
            wcs=wcs,
            psfImage=psfImage,
            noiseCorr=noiseCorr,
            exposure=exposure,
        )
        butlerQC.put(outputs, outputRefs)
        return
