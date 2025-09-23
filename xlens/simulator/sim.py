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

import galsim
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as meaAlg
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.afw.cameraGeom.testUtils import DetectorWrapper
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import Field, FieldValidationError
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.skymap import BaseSkyMap

from ..utils.random import (
    gal_seed_base,
    get_noise_seed,
    num_rot,
)
from .defaults import (
    mag_zero_defaults,
    noise_variance_defaults,
    psf_fwhm_defaults,
    sys_npix,
)
from .galaxies import CatSim2017Catalog, OpenUniverse2024RubinRomanCatalog
from .noise import get_noise_array


class MultibandSimConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch", "band"),
    defaultTemplates={
        "coaddName": "deep",
        "simCoaddName": "sim",
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
    truthCatalog = cT.Input(
        doc="Output truth catalog",
        name="{simCoaddName}_{mode}_rot{rotId}_Coadd_truthCatalog",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
    )
    exposure = cT.Input(
        doc="Input coadd exposure",
        name="{coaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=False,
        minimum=0,
    )
    noiseCorrImage = cT.Input(
        doc="image for noise correlation function",
        name="{coaddName}Coadd_systematics_noisecorr",
        dimensions=("skymap", "tract"),
        storageClass="ImageF",
        multiple=False,
        minimum=0,
    )
    psfImage = cT.Input(
        doc="image for PSF model for simulation",
        name="{coaddName}Coadd_systematics_psfcentered",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ImageF",
        multiple=False,
        minimum=0,
    )
    simExposure = cT.Output(
        doc="Output simulated coadd exposure",
        name="{simCoaddName}_{mode}_rot{rotId}_Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class MultibandSimConfig(
    PipelineTaskConfig,
    pipelineConnections=MultibandSimConnections,
):

    galaxy_type = Field[str](
        doc="galaxy type",
        default="catsim2017",
    )
    survey_name = Field[str](
        doc="Name of the survey",
        default="LSST",
    )
    include_pixel_masks = Field[bool](
        doc="whether to include pixel masks in the simulation",
        default=False,
    )
    draw_image_noise = Field[bool](
        doc="Whether to draw image noise in the simulation",
        default=False,
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
        doc="random seed index for noise, 0 <= noiseId < 10",
        default=0,
    )
    use_real_psf = Field[bool](
        doc="whether to use real PSF",
        default=False,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()
        if self.galId >= gal_seed_base or self.galId < 0:
            raise FieldValidationError(
                self.__class__.galId,
                self,
                "We require 0 <= galId < %d" % (gal_seed_base),
            )
        if self.rotId >= num_rot:
            raise FieldValidationError(
                self.__class__.rotId,
                self,
                "rotId needs to be smaller than 2",
            )
        if self.noiseId < 0:
            raise FieldValidationError(
                self.__class__.noiseId,
                self,
                "We require noiseId >=0 ",
            )
        if self.galaxy_type not in ["catsim2017", "RomanRubin2024"]:
            raise FieldValidationError(
                self.__class__.galaxy_type,
                self,
                "We require galaxy_type in ['catsim2017', 'RomanRubin2024']",
            )

    def setDefaults(self):
        super().setDefaults()
        self.survey_name = self.survey_name.lower()


class MultibandSimTask(PipelineTask):
    _DefaultName = "MultibandSimTask"
    ConfigClass = MultibandSimConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, MultibandSimConfig)
        self.rotate_list = [np.pi / num_rot * i for i in range(num_rot)]
        pass

    def simulate_images(
        self,
        *,
        catalog,
        psf_obj,
        tract_info,
        patch_id: int,
        band: str,
        mag_zero: float,
        draw_method: str = "auto",
        **kwargs,
    ):
        assert isinstance(self.config, MultibandSimConfig)
        if self.config.galaxy_type == "catsim2017":
            GalClass = CatSim2017Catalog
        elif self.config.galaxy_type == "RomanRubin2024":
            GalClass = OpenUniverse2024RubinRomanCatalog
        else:
            raise ValueError("invalid galaxy_type")
        galaxy_catalog = GalClass.from_array(
            tract_info=tract_info,
            table=catalog,
        )
        gal_array = galaxy_catalog.draw(
            patch_id=patch_id,
            psf_obj=psf_obj,
            mag_zero=mag_zero,
            band=band,
            draw_method=draw_method,
        )
        return gal_array

    def runQuantum(self, butlerQC, inputRefs, outputRefs) -> None:
        assert butlerQC.quantum.dataId is not None
        inputs = butlerQC.get(inputRefs)

        # band name
        assert butlerQC.quantum.dataId is not None
        band = butlerQC.quantum.dataId["band"]
        patch_id = butlerQC.quantum.dataId["patch"]
        inputs["band"] = band
        inputs["patch_id"] = patch_id

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
        tract_info = sky_info.tractInfo
        inputs["tract_info"] = tract_info

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)
        return

    def run(
        self,
        *,
        tract_info,
        patch_id: int,
        band: str,
        seed: int,
        truthCatalog,
        psfImage: afwImage.ImageF | None = None,
        noiseCorrImage: afwImage.ImageF | None = None,
        exposure: afwImage.ExposureF | None = None,
        **kwargs,
    ):
        assert isinstance(self.config, MultibandSimConfig)
        if self.config.use_real_psf:
            if psfImage is None:
                raise IOError("Do not have PSF input model")

        # Prepare the random number generator and basic parameters
        survey_name = self.config.survey_name

        boundary_box = tract_info[patch_id].getOuterBBox()
        wcs = tract_info.getWcs()
        pixel_scale = wcs.getPixelScale().asArcseconds()

        mag_zero = mag_zero_defaults[self.config.survey_name]
        zero_flux = 10.0 ** (0.4 * mag_zero)
        photo_calib = afwImage.makePhotoCalibFromCalibZeroPoint(zero_flux)

        if exposure is not None:
            self.log.debug("Using the real pixel mask")
            mask_array = exposure.getMaskedImage().mask.array
            assert mag_zero == 2.5 * np.log10(
                exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()
            )
        else:
            self.log.debug("Do not use the real pixel mask")
            mask_array = 0.0

        # Obtain PSF object for Galsim
        if psfImage is not None and self.config.use_real_psf:
            psf_galsim = galsim.InterpolatedImage(
                galsim.Image(psfImage.getArray()),
                scale=pixel_scale,
                flux=1.0,
            )
            draw_method = "no_pixel"
        else:
            psf_fwhm = psf_fwhm_defaults[band][survey_name]
            psf_galsim = galsim.Moffat(fwhm=psf_fwhm, beta=2.5)
            psf_array = psf_galsim.drawImage(
                nx=sys_npix,
                ny=sys_npix,
                scale=pixel_scale,
                wcs=None,
            ).array
            psfImage = afwImage.ImageF(sys_npix, sys_npix)
            assert psfImage is not None
            psfImage.array[:, :] = psf_array
            draw_method = "auto"

        # and psf kernel for the LSST exposure
        kernel = afwMath.FixedKernel(psfImage.convertD())
        kernel_psf = meaAlg.KernelPsf(kernel)

        galaxy_array = self.simulate_images(
            catalog=truthCatalog,
            psf_obj=psf_galsim,
            tract_info=tract_info,
            patch_id=patch_id,
            band=band,
            mag_zero=mag_zero,
            draw_method=draw_method,
        )

        # Obtain Noise correlation array
        if noiseCorrImage is None:
            noise_corr = None
            variance = noise_variance_defaults[band][survey_name]
            self.log.debug("No correlation, variance:", variance)
        else:
            noise_corr = noiseCorrImage.getArray()
            variance = np.amax(noise_corr)
            noise_corr = noise_corr / variance
            ny, nx = noise_corr.shape
            assert noise_corr[ny // 2, nx // 2] == 1
            self.log.debug("With correlation, variance:", variance)
        noise_std = np.sqrt(variance)

        exp_out = afwImage.ExposureF(boundary_box)
        exp_out.getMaskedImage().image.array[:, :] = galaxy_array

        exp_out.setPhotoCalib(photo_calib)
        exp_out.setPsf(kernel_psf)
        exp_out.setWcs(wcs)
        exp_out.getMaskedImage().variance.array[:, :] = variance
        filter_label = afwImage.FilterLabel(band=band, physical=band)
        exp_out.setFilter(filter_label)
        detector = DetectorWrapper().detector
        exp_out.setDetector(detector)
        del photo_calib, kernel_psf, filter_label, detector

        if self.config.draw_image_noise:
            galaxy_seed = seed * gal_seed_base + self.config.galId
            seed_noise = get_noise_seed(
                galaxy_seed=galaxy_seed,
                noiseId=self.config.noiseId,
                rotId=self.config.rotId,
                band=band,
                is_sim=True,
            )
            noise_array = get_noise_array(
                seed_noise=seed_noise,
                noise_std=noise_std,
                noise_corr=noise_corr,
                shape=galaxy_array.shape,
                pixel_scale=pixel_scale,
            )
            exp_out.getMaskedImage().image.array[:, :] = (
                exp_out.getMaskedImage().image.array[:, :] + noise_array
            )
            del noise_array
        exp_out.getMaskedImage().mask.array[:, :] = mask_array
        del mask_array, galaxy_array

        outputs = Struct(
            simExposure=exp_out,
        )
        return outputs
