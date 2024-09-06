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
from lsst.daf.butler import ValidationError
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import ConfigurableField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
)
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.skymap import BaseSkyMap

from ..simulator.multiband import MultibandSimShearTask


class MultibandSimShearPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch", "band"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "outputCoaddName": "deep",
        "psfType": "moffat",
        "simType": "_sim",
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
    simType = "{simType}"
    if simType == "_sim":
        exposure = cT.Input(
            doc="Input coadd exposure",
            name="{inputCoaddName}Coadd_calexp",
            storageClass="ExposureF",
            dimensions=("skymap", "tract", "patch", "band"),
            multiple=False,
        )
        noiseCorrImage = cT.Input(
            doc="image for noise correlation function",
            name="{outputCoaddName}Coadd_systematics_noisecorr",
            dimensions=("tract", "patch", "band", "skymap"),
            storageClass="ImageF",
            multiple=False,
        )
    psfType = "{psfType}"
    if psfType in ["psf", "star"]:
        psfImage = cT.Input(
            doc="image for PSF model for simulation",
            name="{inputCoaddName}Coadd_systematics_psfcentered",
            dimensions=("skymap", "tract", "patch", "band"),
            storageClass="ImageF",
            multiple=False,
        )
    outputExposure = cT.Output(
        doc="Output simulated coadd exposure",
        name="{outputCoaddName}Coadd_calexp{simType}_{mode}_rot{irot}",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
    )
    outputTruthCatalog = cT.Output(
        doc="Output truth catalog",
        name="{outputCoaddName}Coadd_truthCatalog{simType}_{mode}_rot{irot}",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class MultibandSimShearPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=MultibandSimShearPipeConnections,
):
    simulator = ConfigurableField(
        target=MultibandSimShearTask,
        doc="Fpfs Source Measurement Task",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()
        psf_type_list = ["moffat", "psf", "star"]
        if self.connections.psfType not in psf_type_list:
            raise ValidationError("connection.psfType is incorrect")

    def setDefaults(self):
        super().setDefaults()
        self.simulator.irot = int(self.connections.irot)
        self.simulator.mode = int(self.connections.mode)
        # print(int(self.connections.mode))
        # print(self.simulator.mode)


class MultibandSimShearPipe(PipelineTask):
    _DefaultName = "MultibandSimShearPipe"
    ConfigClass = MultibandSimShearPipeConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert isinstance(self.config, MultibandSimShearPipeConfig)
        self.makeSubtask("simulator")

    def runQuantum(self, butlerQC, inputRefs, outputRefs) -> None:
        assert isinstance(self.config, MultibandSimShearPipeConfig)

        # band name
        assert butlerQC.quantum.dataId is not None
        band = butlerQC.quantum.dataId["band"]

        # Get unique integer ID for IdFactory and RNG seeds; only the latter
        # should really be used as the IDs all come from the input catalog.
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        seed = idGenerator.catalog_id

        # Extract tract and patch IDs from dataId
        skymap = butlerQC.get(inputRefs.skymap)
        if self.config.connections.psfType in ["psf", "star"]:
            psfImage = butlerQC.get(inputRefs.psfImage)
        else:
            psfImage = None

        sky_info = makeSkyInfo(
            skymap,
            tractId=butlerQC.quantum.dataId["tract"],
            patchId=butlerQC.quantum.dataId["patch"],
        )
        boundaryBox = sky_info.bbox

        # Obtain the WCS for the patch
        tract_info = sky_info.tractInfo
        wcs = tract_info.getWcs()

        if hasattr(inputRefs, "exposure") and hasattr(
            inputRefs, "noiseCorrImage"
        ):
            exposure = butlerQC.get(inputRefs.exposure)
            noiseCorr = butlerQC.get(inputRefs.noiseCorrImage)
        else:
            exposure = None
            noiseCorr = None

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
