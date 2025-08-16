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

import lsst.afw.image as afwImage
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import Field, FieldValidationError
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)

from ..simulator.multiband import get_noise_array
from ..simulator.multiband_defaults import noise_variance_defaults
from ..utils.random import get_noise_seed, num_rot


class MultibandSimPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch", "band"),
    defaultTemplates={
        "coaddName": "deep",
        "noiseId": 0,
    },
):
    noiseCorrImage = cT.Input(
        doc="image for noise correlation function",
        name="deepCoadd_systematics_noisecorr",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ImageF",
        multiple=False,
        minimum=0,
    )
    exposure = cT.Input(
        doc="Input simulated coadd exposure",
        name="{coaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
    )
    outputExposure = cT.Output(
        doc="Output simulated coadd exposure",
        name="{coaddName}noise{noiseId}_Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class AddNoisePipeConfig(
    PipelineTaskConfig,
    pipelineConnections=MultibandSimPipeConnections,
):
    idGenerator = SkyMapIdGeneratorConfig.make_field()
    survey_name = Field[str](
        doc="Name of the survey",
        default="lsst",
    )

    rotId = Field[int](
        doc="number of rotations",
        default=0,
    )

    noiseId = Field[int](
        doc="random seed for noise, 0 <= noiseId < 10",
        default=0,
    )

    def validate(self):
        super().validate()
        if self.noiseId < 0 or self.noiseId >= 10:
            raise FieldValidationError(
                self.__class__.noiseId, self, "We require 0 <= noiseId < 10"
            )

    def setDefaults(self):
        super().setDefaults()


class AddNoisePipe(PipelineTask):
    _DefaultName = "AddNoisePipe"
    ConfigClass = AddNoisePipeConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs) -> None:
        assert isinstance(self.config, AddNoisePipeConfig)
        inputs = butlerQC.get(inputRefs)

        # band name
        assert butlerQC.quantum.dataId is not None
        # Get unique integer ID for IdFactory and RNG seeds; only the latter
        # should really be used as the IDs all come from the input catalog.
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        seed = idGenerator.catalog_id
        inputs["seed"] = seed
        band = butlerQC.quantum.dataId["band"]
        inputs["band"] = band
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)
        return

    def run(
        self,
        *,
        exposure: afwImage.ExposureF,
        seed: int,
        band: str,
        noiseCorrImage: afwImage.ImageF | None = None,
        **kwargs,
    ):
        assert isinstance(self.config, AddNoisePipeConfig)
        # Obtain Noise correlation array
        if noiseCorrImage is None:
            noise_corr = None
            variance = noise_variance_defaults[band][self.config.survey_name]
            self.log.debug("No correlation, variance:", variance)
        else:
            noise_corr = noiseCorrImage.getArray()
            variance = np.amax(noise_corr)
            noise_corr = noise_corr / variance
            ny, nx = noise_corr.shape
            assert noise_corr[ny // 2, nx // 2] == 1
            self.log.debug("With correlation, variance:", variance)
        noise_std = np.sqrt(variance)
        seed_noise = get_noise_seed(
            seed=seed,
            noiseId=self.config.noiseId,
            rotId=self.config.rotId,
        )
        height, width = exposure.getMaskedImage().image.array.shape
        wcs = exposure.getWcs()
        pixel_scale = wcs.getPixelScale().asArcseconds()
        noise_array = get_noise_array(
            seed_noise=seed_noise,
            noise_std=noise_std,
            noise_corr=noise_corr,
            shape=(height, width),
            pixel_scale=pixel_scale,
        )
        exposure.getMaskedImage().image.array[:, :] = (
            exposure.getMaskedImage().image.array[:, :] + noise_array
        )

        return Struct(outputExposure=exposure)
