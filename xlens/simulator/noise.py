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
from .defaults import (
    mag_zero_defaults,
    noise_variance_defaults,
    source_noise_coeff_defaults,
)


def get_noise_array(
    *,
    seed_noise: int,
    noise_std: float,
    noise_corr: NDArray | None,
    shape: tuple[int, int],
    pixel_scale: float,
    variance_array: NDArray | None = None,
) -> NDArray:
    """Generate a noise realisation, optionally with spatial correlation.

    Parameters
    ----------
    seed_noise : int
        Random seed for reproducibility.
    noise_std : float
        Standard deviation of the sky noise (before correlation scaling).
    noise_corr : NDArray or None
        Normalised noise correlation kernel.  When *None*, independent
        Gaussian noise is drawn.
    shape : tuple[int, int]
        ``(height, width)`` of the output array.
    pixel_scale : float
        Pixel scale in arcseconds, passed to ``anacal.noise.simulate_noise``.
    variance_array : NDArray or None
        Per-pixel TOTAL noise variance, sky + source photon noise, as
        ``get_total_variance_array`` returns it.  When given it replaces
        ``noise_std``: one standard-normal field is scaled by its square
        root, so the noise is white but non-stationary.  Not supported
        together with ``noise_corr``.

    Returns
    -------
    NDArray
        Two-dimensional noise array.
    """
    if variance_array is not None:
        if noise_corr is not None:
            raise RuntimeError(
                "source photon noise is not supported with correlated "
                "noise: a stationary correlated field cannot be rescaled "
                "pixel by pixel without destroying the correlation it was "
                "built to have"
            )
        # One field, not a sky field plus a source field: there is a single
        # noise, whose variance happens to vary across the image.
        # NB numpy's normal(scale=s) is s * gauss off the same stream, so
        # where the flux is zero this reproduces the sky-only realisation
        # for the same seed bit for bit; only pixels with flux change.
        eps = np.random.RandomState(seed_noise).normal(size=shape)
        return eps * np.sqrt(variance_array)
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


def get_total_variance_array(
    *,
    image_array: NDArray,
    band: str,
    survey_name: str,
    sky_variance: float,
    mag_zero: float,
) -> NDArray:
    """Per-pixel noise variance, sky + source photon noise, in the image's
    own units.

    This is ``defaults.get_variance_array`` generalised to a zero point other
    than the tabulated one.  ``source_noise_coeff_defaults`` is a gain, 1/g,
    tabulated at ``mag_zero_defaults[survey_name]``; a gain is flux-like, so
    it rescales as ``10**(0.4 dZ)`` -- NOT as the ``10**(0.8 dZ)`` of the
    variance it multiplies into.  Rescaling the coefficient once is why the
    image never has to be carried to the tabulated zero point and the noise
    carried back.

    ``sky_variance`` is passed in rather than looked up, because the caller
    has already formed it at the image's own zero point (and, for correlated
    noise, may have taken it from the peak of a correlation function instead
    of from the defaults table).

    Negative pixels are clipped -- a variance cannot be negative.  A noiseless
    simulated image has none, but this is cheap insurance for the case where
    the input is not in fact noiseless.
    """
    coeff = source_noise_coeff_defaults[band][survey_name] * 10.0 ** (
        0.4 * (mag_zero - mag_zero_defaults[survey_name])
    )
    image = np.maximum(np.asarray(image_array, dtype=np.float64), 0.0)
    return sky_variance + coeff * image


class AddNoisePipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch", "band"),
    defaultTemplates={
        "inputName": "deep_coadd",
        "noiseId": 0,
    },
):
    """Butler connections for :class:`AddNoisePipe`.

    The output is named after the input, as every derived product is:
    ``{inputName}`` -> ``{inputName}_noise{noiseId}_coadd``, e.g.
    ``deep_coadd`` -> ``deep_coadd_noise0_coadd`` and, after a noiseless
    simulation, ``sim_0_rot0_coadd`` -> ``sim_0_rot0_coadd_noise0_coadd``,
    which MeasureCoaddsPipe then reads through its own ``inputName``.
    """

    noiseCorrImage = cT.Input(
        doc="image for noise correlation function",
        name="{inputName}_systematics_noisecorr",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ImageF",
        multiple=False,
        minimum=0,
    )
    exposure = cT.Input(
        doc="Input (noiseless) coadd exposure",
        name="{inputName}",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
    )
    simExposure = cT.Output(
        doc="The exposure with the noise realisation noiseId added",
        name="{inputName}_noise{noiseId}_coadd",
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
    survey_in_seed = Field[bool](
        doc=(
            "Mix survey_name into the noise seed. MultibandSimTask does NOT "
            "(it calls get_noise_seed without a survey), so leave this False "
            "to add, to a noiseless simulation, exactly the realisation that "
            "draw_image_noise=True would have drawn for the same seed, "
            "galId, noiseId, rotId and band."
        ),
        default=False,
    )
    mag_zero = Field[float](
        doc=(
            "Magnitude zero point the exposure was rendered at; the white-"
            "noise variance is noise_variance_defaults scaled by "
            "10**(0.8 (mag_zero - survey default)), exactly as "
            "MultibandSimTask computes it. Negative (default) means the "
            "survey default, as in the simulator."
        ),
        default=-1.0,
    )
    draw_source_noise = Field[bool](
        doc=(
            "Add source photon noise on top of the sky noise, making the "
            "image noise NON-STATIONARY: the per-pixel variance becomes "
            "defaults.get_variance_array, sky + c * flux, with the flux read "
            "off the (noiseless) input image. One white field is drawn, "
            "scaled by sqrt(sky + c * flux); this is NOT supported together "
            "with a noise correlation kernel. The variance PLANE is left at "
            "the constant sky value regardless -- see the note on the plane "
            "below. LSST only. Do NOT set this on an input that already "
            "contains noise (a real deep_coadd): the flux it reads would be "
            "that noise, and the photon term would be drawn from it."
        ),
        default=False,
    )
    check_variance_plane = Field[bool](
        doc=(
            "If the input already carries a positive variance plane, require "
            "it to agree with the noise variance to be added (float32 "
            "precision); a disagreement means the image was rendered for a "
            "different noise level. The plane is set to the noise variance "
            "in any case, so an input without one comes out correct."
        ),
        default=True,
    )

    def validate(self):
        super().validate()
        if self.draw_source_noise and self.survey_name != "lsst":
            raise FieldValidationError(
                self.__class__.draw_source_noise,
                self,
                "source photon noise is only calibrated for lsst, not "
                f"{self.survey_name}",
            )
        if self.noiseId < 0 or self.noiseId >= 10:
            raise FieldValidationError(self.__class__.noiseId, self, "We require 0 <= noiseId < 10")
        # the realisation in the output NAME must be the one that seeds the noise
        if int(self.connections.noiseId) != int(self.noiseId):
            raise FieldValidationError(
                self.__class__.noiseId,
                self,
                f"noiseId ({self.noiseId}) must equal connections.noiseId "
                f"({self.connections.noiseId}); set both",
            )

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
        if self.config.draw_source_noise and noiseCorrImage is not None:
            raise RuntimeError(
                "draw_source_noise is not supported with a noise "
                "correlation kernel; use one or the other"
            )
        # Obtain Noise correlation array
        if noiseCorrImage is None:
            noise_corr = None
            # the simulator's own arithmetic, so the float64 noise_std -- and
            # hence every noise sample -- is bit-identical to draw_image_noise
            mag_zero_default = mag_zero_defaults[self.config.survey_name]
            if self.config.mag_zero < 0:
                var_ratio = 1.0
            else:
                var_ratio = 10.0 ** ((self.config.mag_zero - mag_zero_default) * 0.8)
            variance = noise_variance_defaults[band][self.config.survey_name] * var_ratio
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
        # The variance plane carries the SKY noise variance only -- the
        # constant (for correlated noise, the peak of the correlation
        # function) -- even when draw_source_noise has made the noise in the
        # image non-stationary.  That is deliberate: downstream,
        # ``estimate_noise_variance`` collapses this plane to one float per
        # band, which anacal uses for the noise-bias correction and for the
        # renoising spectrum, and both of those model stationary sky noise.
        # They must be told the sky level and nothing more; handing them the
        # source term would have them correct for noise the correction does
        # not apply to.  An input that already has a plane is checked first,
        # since a different value means the galaxies were rendered for
        # another noise level.
        var_plane = exposure.getMaskedImage().variance.array
        if self.config.check_variance_plane:
            plane = float(np.median(var_plane))
            if plane > 0.0 and not np.isclose(plane, variance, rtol=1e-6, atol=0.0):
                raise RuntimeError(
                    f"variance plane ({plane:g}) disagrees with the noise "
                    f"variance to add ({variance:g}) for band {band}; the "
                    "exposure was rendered for a different noise level "
                    "(survey_name / mag_zero)"
                )
        var_plane[:, :] = variance
        galaxy_seed = seed * gal_seed_base + self.config.galId
        seed_noise = get_noise_seed(
            galaxy_seed=galaxy_seed,
            noiseId=self.config.noiseId,
            rotId=self.config.rotId,
            band=band,
            survey=self.config.survey_name if self.config.survey_in_seed else None,
            is_sim=True,
        )
        height, width = exposure.getMaskedImage().image.array.shape
        wcs = exposure.getWcs()
        pixel_scale = wcs.getPixelScale().asArcseconds()
        variance_array = None
        if self.config.draw_source_noise:
            # the flux is read off the image while it is still noiseless
            variance_array = get_total_variance_array(
                image_array=exposure.getMaskedImage().image.array,
                band=band,
                survey_name=self.config.survey_name,
                sky_variance=variance,
                mag_zero=(
                    mag_zero_defaults[self.config.survey_name]
                    if self.config.mag_zero < 0
                    else self.config.mag_zero
                ),
            )
        noise_array = get_noise_array(
            seed_noise=seed_noise,
            noise_std=noise_std,
            noise_corr=noise_corr,
            shape=(height, width),
            pixel_scale=pixel_scale,
            variance_array=variance_array,
        )
        exposure.getMaskedImage().image.array[:, :] = (
            exposure.getMaskedImage().image.array[:, :] + noise_array
        )
        return Struct(simExposure=exposure)
