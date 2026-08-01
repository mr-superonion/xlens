#!/usr/bin/env python
# This file is part of xlens.
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
"""Pipeline task for adding correlated or uncorrelated noise to
coadd exposures."""

from typing import Any

import anacal
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
from numpy.typing import NDArray

from ..utils.random import gal_seed_base, get_noise_seed
from .defaults import noise_variance_defaults


def get_noise_array(
    *,
    seed_noise: int,
    noise_std: float,
    noise_corr: NDArray | None,
    shape: tuple[int, int],
    pixel_scale: float,
) -> NDArray:
    """Generate a noise realisation, optionally with spatial correlation.

    Parameters
    ----------
    seed_noise : int
        Random seed for reproducibility.
    noise_std : float
        Standard deviation of the noise (before correlation scaling).
    noise_corr : NDArray or None
        Normalised noise correlation kernel.  When *None*, independent
        Gaussian noise is drawn.
    shape : tuple[int, int]
        ``(height, width)`` of the output array.
    pixel_scale : float
        Pixel scale in arcseconds, passed to ``anacal.noise.simulate_noise``.

    Returns
    -------
    NDArray
        Two-dimensional noise array.
    """
    if noise_corr is None:
        noise_array = np.random.RandomState(seed_noise).normal(
            scale=noise_std,
            size=shape,
        )
    else:
        noise_array = (
            anacal.noise.simulate_noise(
                seed=seed_noise,
                correlation=noise_corr,
                nx=shape[1],
                ny=shape[0],
                scale=pixel_scale,
            )
            * noise_std
        )
    return noise_array


class AddNoisePipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch", "band"),
    defaultTemplates={
        "coaddName": "deep",
        "noiseId": 0,
    },
):
    """Butler connections for :class:`AddNoisePipe`."""

    noiseCorrImage = cT.Input(
        doc="image for noise correlation function",
        name="deep_coadd_systematics_noisecorr",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ImageF",
        multiple=False,
        minimum=0,
    )
    exposure = cT.Input(
        doc="Input simulated coadd exposure",
        name="{coaddName}_coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
    )
    simExposure = cT.Output(
        doc="Output simulated coadd exposure",
        name="{coaddName}noise{noiseId}_coadd",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class AddNoisePipeConfig(
    PipelineTaskConfig,
    pipelineConnections=AddNoisePipeConnections,
):
    """Configuration for :class:`AddNoisePipe`."""

    idGenerator = SkyMapIdGeneratorConfig.make_field()
    survey_name = Field[str](
        doc="Name of the survey",
        default="lsst",
    )
    galId = Field[int](
        doc="random seed index for galaxy, 0 <= galId < 10",
        default=0,
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
            raise FieldValidationError(self.__class__.noiseId, self, "We require 0 <= noiseId < 10")

    def setDefaults(self):
        super().setDefaults()


class AddNoisePipe(PipelineTask):
    """Pipeline task that adds noise to a simulated coadd exposure."""

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
        """Add noise to an exposure and return the result.

        Parameters
        ----------
        exposure : afwImage.ExposureF
            Input simulated coadd exposure (modified in place).
        seed : int
            Base seed from the pipeline's ID generator.
        band : str
            Photometric band label.
        noiseCorrImage : afwImage.ImageF or None, optional
            Noise correlation kernel image.  When *None*, uncorrelated
            Gaussian noise with default survey variance is used.

        Returns
        -------
        lsst.pipe.base.Struct
            Struct with ``simExposure`` attribute containing the noisy
            exposure.
        """
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
            if noise_corr[ny // 2, nx // 2] != 1:
                raise RuntimeError("noise correlation peak is not at the center pixel")
            self.log.debug("With correlation, variance:", variance)
        noise_std = np.sqrt(variance)
        galaxy_seed = seed * gal_seed_base + self.config.galId
        seed_noise = get_noise_seed(
            galaxy_seed=galaxy_seed,
            noiseId=self.config.noiseId,
            rotId=self.config.rotId,
            band=band,
            survey=self.config.survey_name,
            is_sim=True,
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
        return Struct(simExposure=exposure)
