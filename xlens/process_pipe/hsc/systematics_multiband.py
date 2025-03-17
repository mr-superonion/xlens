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
    "SystematicsMultibandPipeConfig",
    "SystematicsMultibandPipe",
    "SystematicsMultibandPipeConnections",
]

import glob
import logging
import os
from typing import Any

import fitsio
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.geom import Box2I, Extent2I, Point2D, Point2I
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import Field, FieldValidationError
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter

from . import resize_array, subpixel_shift


class SystematicsMultibandPipeConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates={"coaddName": "deep"},
):
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )

    exposure = cT.Input(
        doc="Input coadd image",
        name="{coaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap"),
    )
    catalog = cT.Input(
        doc=("original measurement catalog"),
        name="{coaddName}Coadd_meas",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap"),
    )
    outputNoiseCorr = cT.Output(
        doc="noise correlation function",
        name="{coaddName}Coadd_systematics_noisecorr",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="ImageF",
    )
    outputPsfCentered = cT.Output(
        doc="noise correlation function",
        name="{coaddName}Coadd_systematics_psfcentered",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="ImageF",
    )
    outputStarCentered = cT.Output(
        doc="noise correlation function",
        name="{coaddName}Coadd_systematics_starcentered",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="ImageF",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class SystematicsMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=SystematicsMultibandPipeConnections,
):
    psfCache = Field[int](
        doc="Size of psfCache",
        default=100,
    )

    npix = Field[int](
        doc="number of pixels for the length of stamp",
        default=49,
    )

    star_snr_min = Field[float](
        doc="minimum (aperture) snr threshold of stars",
        default=100.0,
    )

    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()
        if self.npix % 2 == 0:
            raise FieldValidationError(
                self.__class__.npix, self, "npix should be odd number"
            )


class SystematicsMultibandPipe(PipelineTask):
    _DefaultName = "FpfsTask"
    ConfigClass = SystematicsMultibandPipeConfig

    def __init__(
        self,
        *,
        config: SystematicsMultibandPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, SystematicsMultibandPipeConfig)

        self.pt_data = fitsio.read(
            os.path.join(
                "/work/xiangchong.li/work/hsc_s23b_sim/catalogs/",
                "tracts_fdfc_v1_trim2_sim.fits",
            )
        )
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, SystematicsMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        # assert butlerQC.quantum.dataId is not None
        # tract = butlerQC.quantum.dataId["tract"]
        # patch = butlerQC.quantum.dataId["patch"]
        # patch_list = self.pt_data[self.pt_data["tract"] == tract]["patch"]
        # patch_y = int(patch) // 9
        # patch_x = int(patch) % 9
        # patch_db = patch_x * 100 + patch_y
        # if patch_db not in patch_list:
        #     return

        inputs = butlerQC.get(inputRefs)
        inputs["exposure"].getPsf().setCacheCapacity(self.config.psfCache)
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        seed = idGenerator.catalog_id
        outputs = self.run(
            seed=seed,
            **inputs
        )
        butlerQC.put(outputs, outputRefs)
        return

    def run(self, *, exposure, catalog, seed, **kwargs):
        noise_corr = self.get_noise_corr(exposure)
        psf_image, star_image = self.get_psf_systematics(
            exposure,
            catalog,
            seed,
        )
        del exposure, catalog
        return Struct(
            outputNoiseCorr=noise_corr,
            outputPsfCentered=psf_image,
            outputStarCentered=star_image,
        )

    def get_noise_corr(self, exposure):
        assert isinstance(self.config, SystematicsMultibandPipeConfig)
        variance_array = exposure.getMaskedImage().variance.array[
            1000:3000, 1000:3000
        ]
        window_array = (exposure.mask.array == 0).astype(np.float32)[
            1000:3000, 1000:3000
        ]

        noise_array = np.asarray(
            exposure.getMaskedImage().image.array,
            dtype=np.float32,
        )[1000:3000, 1000:3000]
        window_array = (
            window_array * (noise_array ** 2.0 < variance_array * 9)
            * (variance_array < 5.0) * (~np.isnan(variance_array))
        )

        noise_array[~window_array.astype(bool)] = 0.0
        noise_variance = np.average(
            variance_array[window_array.astype(bool)]
        )
        if noise_variance < 1e-20:
            raise ValueError(
                "the estimated image noise variance should be positive."
            )

        pad_width = ((10, 10), (10, 10))  # ((top, bottom), (left, right))
        window_array = np.pad(
            window_array,
            pad_width=pad_width,
            mode="constant",
            constant_values=0.0,
        )
        noise_array = np.pad(
            noise_array,
            pad_width=pad_width,
            mode="constant",
            constant_values=0.0,
        )
        ny, nx = window_array.shape

        npixl = int(self.config.npix // 2)
        npixr = int(self.config.npix // 2 + 1)
        noise_corr = np.fft.fftshift(
            np.fft.ifft2(np.abs(np.fft.fft2(noise_array)) ** 2.0)
        ).real[
            ny // 2 - npixl : ny // 2 + npixr, nx // 2 - npixl : nx // 2 + npixr
        ]
        window_corr = np.fft.fftshift(
            np.fft.ifft2(np.abs(np.fft.fft2(window_array)) ** 2.0)
        ).real[
            ny // 2 - npixl : ny // 2 + npixr, nx // 2 - npixl : nx // 2 + npixr
        ]
        noise_corr = noise_corr / window_corr
        del window_array, noise_array, window_corr

        noise_image = afwImage.ImageF(self.config.npix, self.config.npix)
        noise_image.array[:, :] = noise_corr

        return noise_image

    def get_psf_systematics(self, exposure, catalog, seed):
        assert isinstance(self.config, SystematicsMultibandPipeConfig)
        npixl = int(self.config.npix // 2)
        npixr = int(self.config.npix // 2 + 1)

        catalog = catalog.asAstropy().as_array()
        msk = catalog["calib_psf_reserved"] & catalog["detect_isPrimary"]
        catalog = catalog[msk]
        snr = (
            catalog["base_CircularApertureFlux_3_0_instFlux"]
            / catalog["base_CircularApertureFlux_3_0_instFluxErr"]
        )
        bbox = exposure.getBBox()
        xmin_exp, ymin_exp = bbox.getMinX(), bbox.getMinY()
        xmax_exp, ymax_exp = bbox.getMaxX(), bbox.getMaxY()
        msk2 = (
            (catalog["base_SdssShape_x"] > xmin_exp + npixl)
            & (catalog["base_SdssShape_y"] > ymin_exp + npixl)
            & (catalog["base_SdssShape_x"] < xmax_exp - npixr)
            & (catalog["base_SdssShape_y"] < ymax_exp - npixr)
            & (snr > self.config.star_snr_min)
        )
        catalog = catalog[msk2]
        nstars = len(catalog)

        if nstars >= 1:
            np.random.seed(seed)
            ind = np.random.randint(0, nstars)
            src = catalog[ind]

            # Collect the PSF image
            exposure.getPsf().setCacheCapacity(self.config.psfCache)
            lsst_psf = exposure.getPsf()
            psf_array = lsst_psf.computeImage(
                Point2D(
                    int(src["base_SdssShape_x"]),
                    int(src["base_SdssShape_y"]),
                )
            ).getArray()
            psf_array = resize_array(
                psf_array,
                (self.config.npix, self.config.npix),
            )
            psf_image = afwImage.ImageF(self.config.npix, self.config.npix)
            psf_image.array[:, :] = psf_array

            bbox = Box2I(
                Point2I(
                    int(src["base_SdssShape_x"]) - npixl,
                    int(src["base_SdssShape_y"]) - npixl,
                ),
                Extent2I(self.config.npix, self.config.npix),
            )

            # Collect the star image
            # Extract the sub-image using the BBox
            star_image = exposure.Factory(exposure, bbox).getImage()
            # Get the image component and convert to a NumPy array
            star_array = star_image.getArray()
            offset_x = src["base_SdssShape_x"] - int(src["base_SdssShape_x"])
            offset_y = src["base_SdssShape_y"] - int(src["base_SdssShape_y"])
            star_array = subpixel_shift(star_array, -offset_x, -offset_y)
            star_image.array[:, :] = star_array
        else:
            psf_image = None
            star_image = None
        return psf_image, star_image
