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
"""Pipeline task that simulates multi-band LSST coadd images.

The :class:`MultibandSimTask` task orchestrates drawing galaxy catalogs,
convolving them with PSF models, and optionally adding realistic noise and
pixel masks.  The code mirrors the Rubin Science Pipelines interface while
providing a self-contained set of utilities that are convenient for unit
tests and tutorials bundled with ``xlens``.
"""

import os
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
from numpy.typing import NDArray

from ..utils.random import (
    gal_seed_base,
    get_noise_seed,
    num_rot,
)
from ..wcs import tanwcs_dm2galsim
from .bat import draw_ia
from .defaults import (
    mag_zero_defaults,
    noise_variance_defaults,
    psf_fwhm_defaults,
    sys_npix,
)
from .galaxies import (
    CatSim2017Catalog,
    DiffskyCatalog,
    Flagship2025Catalog,
)
from .noise import get_noise_array
from ..foreground.star import bright_star

SIM_INCLUSION_PADDING = 200  # pixels
DEFAULT_BAT_STAMP_SIZE = 64
# Band ordering used to index stacked (per-band) PSF / noise-correlation
# arrays. Ground-based surveys use the 6-band optical order; Euclid uses
# its own bands.
band_order_by_survey = {
    "lsst": ["u", "g", "r", "i", "z", "y"],
    "hsc": ["u", "g", "r", "i", "z", "y"],
    "euclid": ["vis", "nisp_y", "nisp_j", "nisp_h"],
}


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
    """Define the Butler datasets consumed and produced by ``MultibandSim``."""

    skymap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    truthCatalog = cT.Input(
        doc="Output truth catalog",
        name="{simCoaddName}_{mode}_rot{rotId}_coadd_truthCatalog",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
        multiple=False,
    )
    mask = cT.Input(
        doc="Input coadd systematics mask",
        name="{coaddName}_coadd_systematics_mask",
        storageClass="Mask",
        dimensions=("skymap", "tract", "patch"),
        multiple=False,
        minimum=0,
    )
    noiseCorrArray = cT.Input(
        doc="Stacked noise correlation array (6 x npix x npix).",
        name="deep_coadd_systematics_noisecorr_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
        multiple=False,
        minimum=0,
    )
    psfArray = cT.Input(
        doc="Stacked PSF image array (6 x npix x npix).",
        name="deep_coadd_systematics_psfcentered_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
        multiple=False,
        minimum=0,
    )
    simExposure = cT.Output(
        doc="Output simulated coadd exposure",
        name="{simCoaddName}_{mode}_rot{rotId}_coadd",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class MultibandSimConfig(
    PipelineTaskConfig,
    pipelineConnections=MultibandSimConnections,
):
    """Configuration options controlling the multi-band simulation task."""

    catsim_dir = Field[str](
        doc="Directory containing input galaxy catalogs.",
        default=os.environ.get("CATSIM_DIR", "."),
    )
    galaxy_type = Field[str](
        doc="galaxy type",
        default="catsim2017",
    )
    survey_name = Field[str](
        doc="Name of the survey",
        default="LSST",
    )
    mag_zero = Field[float](
        doc="magnitude zero point",
        default=-1,
    )
    include_pixel_masks = Field[bool](
        doc="whether to include pixel masks in the simulation",
        default=False,
    )
    draw_image_noise = Field[bool](
        doc="Whether to draw image noise in the simulation",
        default=False,
    )
    use_field_distortion = Field[bool](
        doc="Whether to include field distortion when drawing objects",
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
    use_mog = Field[bool](
        doc="whether to use use multi-Gaussian approximation",
        default=False,
    )
    force_isotropic = Field[bool](
        doc="force all input catalog to be isotropic",
        default=False,
    )
    psf_e1 = Field[float](
        doc="psf ellipticity, first component e1",
        default=0.0,
    )
    psf_e2 = Field[float](
        doc="psf ellipticity, second component e2",
        default=0.0,
    )
    include_point_source = Field[bool](
        doc="whether to include point sources in galaxies (agn or knots)",
        default=True,
    )
    truncate_stamp_size = Field[int](
        doc="truncation size of stamps",
        default=-1,
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
        if self.galaxy_type not in ["catsim2017", "flagship2025", "diffsky"]:
            raise FieldValidationError(
                self.__class__.galaxy_type,
                self,
                "We require galaxy_type in " "['catsim2017', 'flagship2025', 'diffsky']",
            )
        if self.survey_name not in ["lsst", "hsc", "euclid"]:
            raise FieldValidationError(
                self.__class__.survey_name,
                self,
                "survey_name must be one of ['lsst', 'hsc', 'euclid']",
            )
        # Only the Flagship catalog carries the euclid_* magnitude columns,
        # so Euclid simulations must use it (see Flagship2025Catalog, which
        # reads ``entry[f"{survey_name}_{band}"]``).
        if self.survey_name == "euclid" and self.galaxy_type != "flagship2025":
            raise FieldValidationError(
                self.__class__.galaxy_type,
                self,
                "Euclid simulations require galaxy_type='flagship2025' "
                "(only Flagship provides euclid_* magnitudes)",
            )

    def setDefaults(self):
        super().setDefaults()
        self.survey_name = self.survey_name.lower()


class MultibandSimTask(PipelineTask):
    _DefaultName = "MultibandSimTask"
    ConfigClass = MultibandSimConfig

    """Task that draws simulated coadd images for a single patch and band."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, MultibandSimConfig)
        self.rotate_list = [np.pi / num_rot * i for i in range(num_rot)]
        pass

    def simulate_foreground(self, pixel_scale, model, psf_obj, bbox_outer, band, mag_zero, draw_method, **kwargs):
        assert isinstance(self.config, MultibandSimConfig)
        xmin = bbox_outer.getMinX()
        ymin = bbox_outer.getMinY()
        xmax = bbox_outer.getMaxX()
        ymax = bbox_outer.getMaxY()
        width = bbox_outer.getWidth()
        height = bbox_outer.getHeight()

        pix_x = xmin + width//2 #center pix +- sub pixel shift
        pix_y = ymin + height//2 #center pix +- sub pixel shift

        image_pos = galsim.PositionD(x=pix_x, y=pix_y)
        foreground_obj = bright_star(model=model, mag_zero=mag_zero, band=band)
        convolved_object = galsim.Convolve([foreground_obj, psf_obj])

        stamp = convolved_object.drawImage(
            center=image_pos,
            wcs=None,
            method=draw_method,
            scale=pixel_scale,
            nx=width,
            ny=height,
        )

        return stamp
    
    def simulate_images(
        self,
        *,
        galaxy_catalog,
        psf_obj,
        wcs,
        bbox_outer,
        band: str,
        mag_zero: float,
        draw_method: str = "auto",
        **kwargs,
    ):
        """Render a galaxy catalog into an image array.

        Parameters
        ----------
        galaxy_catalog
            Galaxy catalog object (e.g. from ``from_array``) to draw.
        psf_obj
            ``galsim.GSObject`` describing the PSF to use when rendering.
        wcs
            LSST ``SkyWcs`` object that provides the sky-to-pixel mapping.
        bbox_outer
            ``lsst.geom.Box2I`` giving the outer bounding box of the patch.
        band
            Name of the photometric band (``"r"``, ``"i"``, ...).
        mag_zero
            Zeropoint magnitude used for converting fluxes.
        draw_method
            Rendering method passed to the ``galsim`` drawing routines.

        Returns
        -------
        numpy.ndarray
            Two-dimensional array with simulated pixel values for the requested
            patch.
        """
        assert isinstance(self.config, MultibandSimConfig)
        if self.config.truncate_stamp_size <= 0:
            nn_trunc = None
        else:
            nn_trunc = self.config.truncate_stamp_size

        return self.draw_catalog(
            galaxy_catalog=galaxy_catalog,
            wcs=wcs,
            bbox_outer=bbox_outer,
            psf_obj=psf_obj,
            mag_zero=mag_zero,
            band=band,
            draw_method=draw_method,
            nn_trunc=nn_trunc,
        )

    def draw_catalog(
        self,
        *,
        galaxy_catalog,
        wcs,
        bbox_outer,
        psf_obj,
        mag_zero: float,
        band: str,
        draw_method: str = "auto",
        nn_trunc: None | int = None,
        **kwargs,
    ):
        """Iterate over galaxies in the catalog and render them into an image.

        Parameters
        ----------
        galaxy_catalog
            Galaxy catalog with ``data`` array and ``get_obj`` method.
        wcs
            LSST ``SkyWcs`` for the tangent-plane projection.
        bbox_outer
            ``lsst.geom.Box2I`` outer bounding box of the patch.
        psf_obj
            GalSim PSF object used for convolution.
        mag_zero : float
            Zeropoint magnitude.
        band : str
            Photometric band.
        draw_method : str, optional
            GalSim rendering method.
        nn_trunc : int or None, optional
            Stamp truncation size in pixels.  ``None`` means no truncation.

        Returns
        -------
        numpy.ndarray
            Two-dimensional pixel array.
        """
        assert isinstance(self.config, MultibandSimConfig)
        xmin = bbox_outer.getMinX()
        ymin = bbox_outer.getMinY()
        xmax = bbox_outer.getMaxX()
        ymax = bbox_outer.getMaxY()
        width = bbox_outer.getWidth()
        height = bbox_outer.getHeight()
        wcs_gs = tanwcs_dm2galsim(wcs)
        image = galsim.ImageF(width, height, xmin=xmin, ymin=ymin, wcs=wcs_gs)
        survey_name = self.config.survey_name

        # Convert ra/dec to pixel positions using the provided WCS
        pix_x, pix_y = wcs.skyToPixelArray(
            galaxy_catalog.data["ra"],
            galaxy_catalog.data["dec"],
            degrees=True,
        )

        for i, src in enumerate(galaxy_catalog.data):
            ix = pix_x[i]
            iy = pix_y[i]
            if (
                ((xmin - SIM_INCLUSION_PADDING) < ix < (xmax + SIM_INCLUSION_PADDING))
                and ((ymin - SIM_INCLUSION_PADDING) < iy < (ymax + SIM_INCLUSION_PADDING))
                and src["has_finite_shear"]
            ):
                image_pos = galsim.PositionD(x=ix, y=iy)
                gal_obj = galaxy_catalog.get_obj(
                    ind=i,
                    mag_zero=mag_zero,
                    band=band,
                    use_mog=self.config.use_mog,
                    force_isotropic=self.config.force_isotropic,
                    include_point_source=self.config.include_point_source,
                    survey_name=survey_name,
                )
                convolved_object = galsim.Convolve([gal_obj, psf_obj])
                if self.config.use_field_distortion:
                    local_wcs = wcs_gs.local(image_pos=image_pos)
                    stamp = convolved_object.drawImage(
                        center=image_pos,
                        wcs=local_wcs,
                        method=draw_method,
                        nx=nn_trunc,
                        ny=nn_trunc,
                    )
                else:
                    stamp = convolved_object.drawImage(
                        center=image_pos,
                        wcs=None,
                        method=draw_method,
                        scale=galaxy_catalog.pixel_scale,
                        nx=nn_trunc,
                        ny=nn_trunc,
                    )
                b = stamp.bounds & image.bounds
                if b.isDefined():
                    image[b] += stamp[b]
        return image.array

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
        psfArray: NDArray | None = None,
        noiseCorrArray: NDArray | None = None,
        mask: afwImage.MaskX | None = None,
        **kwargs,
    ):
        """Simulate an LSST coadd exposure for a specific tract patch.

        Parameters
        ----------
        tract_info
            ``TractInfo`` describing the region of sky being simulated.
        patch_id
            Identifier of the patch inside ``tract_info`` to draw.
        band
            Photometric band label.
        seed
            Deterministic seed generated by the pipeline's ID generator.
        truthCatalog
            Truth catalog produced by :class:`CatalogTask` containing the
            galaxies to render.
        psfArray, noiseCorrArray, mask
            Optional inputs that provide measured PSFs, noise correlation
            images, or systematics masks from real observations.

        Returns
        -------
        lsst.pipe.base.Struct
            Struct with a single ``simExposure`` attribute holding the
            simulated ``ExposureF`` object.
        """
        assert isinstance(self.config, MultibandSimConfig)
        if self.config.use_real_psf:
            if psfArray is None:
                raise IOError("Do not have PSF input model")

        # Prepare the random number generator and basic parameters
        survey_name = self.config.survey_name

        boundary_box = tract_info[patch_id].getOuterBBox()
        wcs = tract_info.getWcs()
        pixel_scale = wcs.getPixelScale().asArcseconds()

        mag_zero_survey_default = mag_zero_defaults[self.config.survey_name]
        mag_zero = self.config.mag_zero
        if mag_zero < 0:
            mag_zero = mag_zero_survey_default
            var_ratio = 1.0
        else:
            var_ratio = 10.0 ** ((mag_zero - mag_zero_survey_default) * 0.8)
        zero_flux = 10.0 ** (0.4 * mag_zero)
        photo_calib = afwImage.makePhotoCalibFromCalibZeroPoint(zero_flux)

        if mask is not None:
            self.log.debug("Using the real pixel mask")
            mask_array = mask.getArray()
        else:
            self.log.debug("Do not use the real pixel mask")
            mask_array = 0.0

        isys = band_order_by_survey[survey_name].index(band)
        # Obtain PSF object for Galsim
        if psfArray is not None and self.config.use_real_psf:
            draw_method = "no_pixel"
            if len(psfArray.shape) == 3:
                psf_array = psfArray[isys]
            else:
                psf_array = psfArray
            assert abs(np.sum(psf_array) - 1.0) < 1e-2
            psf_galsim = galsim.InterpolatedImage(
                galsim.Image(psf_array),
                scale=pixel_scale,
                flux=1.0,
            )
            psfImage = afwImage.ImageF(psf_array.shape[0], psf_array.shape[1])
        else:
            draw_method = "auto"
            psf_fwhm = psf_fwhm_defaults[band][survey_name]
            psf_galsim = galsim.Moffat(fwhm=psf_fwhm, beta=2.5).shear(
                e1=self.config.psf_e1,
                e2=self.config.psf_e2,
            )
            psf_array = psf_galsim.drawImage(
                nx=sys_npix,
                ny=sys_npix,
                scale=pixel_scale,
                wcs=None,
            ).array
            psfImage = afwImage.ImageF(sys_npix, sys_npix)

        assert psfImage is not None
        psfImage.array[:, :] = psf_array
        # and psf kernel for the LSST exposure
        kernel = afwMath.FixedKernel(psfImage.convertD())
        kernel_psf = meaAlg.KernelPsf(kernel)

        if self.config.galaxy_type == "catsim2017":
            GalClass = CatSim2017Catalog
        elif self.config.galaxy_type == "flagship2025":
            GalClass = Flagship2025Catalog
        elif self.config.galaxy_type == "diffsky":
            GalClass = DiffskyCatalog
        else:
            raise ValueError("invalid galaxy_type")
        galaxy_catalog = GalClass.from_array(
            tract_info=tract_info,
            truthCatalog=truthCatalog,
            catsim_dir=self.config.catsim_dir,
        )

        galaxy_array = self.simulate_images(
            galaxy_catalog=galaxy_catalog,
            psf_obj=psf_galsim,
            wcs=wcs,
            bbox_outer=boundary_box,
            band=band,
            mag_zero=mag_zero,
            draw_method=draw_method,
        )

        ###### TEMP ########
        foreground_array = self.simulate_foreground(
            galaxy_catalog.pixel_scale, 
            None, 
            psf_galsim,
            boundary_box,
            band,
            mag_zero,
            draw_method
        )
        galaxy_array += foreground_array # add to galsim image before noise and afw conversion

        # Obtain Noise correlation array
        if noiseCorrArray is None:
            noise_corr = None
            variance = noise_variance_defaults[band][survey_name] * var_ratio
            self.log.debug("No correlation, variance:", variance)
        else:
            if len(noiseCorrArray.shape) == 3:
                noise_corr = noiseCorrArray[isys]
            else:
                noise_corr = noiseCorrArray
            # Peak of the auto-covariance is the per-pixel noise variance
            # at the input data's zero point; normalise to 1 at the
            # centre and use that peak directly as the variance. Caller
            # is responsible for ensuring ``mag_zero`` matches the data
            # behind the array (no ``var_ratio`` rescaling here).
            variance = float(np.amax(noise_corr))
            noise_corr = noise_corr / variance
            ny, nx = noise_corr.shape
            if noise_corr[ny // 2, nx // 2] != 1:
                raise RuntimeError("noise correlation peak is not at the center pixel")
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


class IASimConnections(MultibandSimConnections):
    """Butler connections for :class:`IASimTask`.

    The intrinsic-alignment simulator uses the same datasets as
    :class:`MultibandSimTask` so this subclass only exists for clarity.
    """


class IASimConfig(MultibandSimConfig):
    """Configuration for :class:`IASimTask` including IA parameters."""

    pipelineConnections = IASimConnections

    ia_amplitude = Field[float](
        doc="Amplitude of the BATSim intrinsic-alignment distortion.",
        default=0.0,
    )
    ia_beta = Field[float](
        doc="Beta parameter passed to the BATSim IA transform.",
        default=0.0,
    )
    ia_phi = Field[float](
        doc="Orientation angle (radians) for the IA distortion field.",
        default=0.0,
    )
    ia_clip_radius = Field[float](
        doc="Clip radius in units of half-light radii for the IA transform.",
        default=3.0,
    )

    def validate(self):  # noqa: D401
        super().validate()


class IASimTask(MultibandSimTask):
    """Task that draws coadds using intrinsic-alignment distortions."""

    _DefaultName = "IASimTask"
    ConfigClass = IASimConfig

    def draw_catalog(
        self,
        *,
        galaxy_catalog,
        wcs,
        bbox_outer,
        psf_obj,
        mag_zero: float,
        band: str,
        draw_method: str = "auto",
        **kwargs,
    ):
        """Render galaxies with intrinsic-alignment distortions via BATSim."""
        assert isinstance(self.config, IASimConfig)
        if self.config.use_field_distortion:
            raise RuntimeError("IASimTask does not yet support use_field_distortion=True.")

        xmin = bbox_outer.getMinX()
        ymin = bbox_outer.getMinY()
        xmax = bbox_outer.getMaxX()
        ymax = bbox_outer.getMaxY()
        width = bbox_outer.getWidth()
        height = bbox_outer.getHeight()

        wcs_gs = tanwcs_dm2galsim(wcs)
        image = galsim.ImageF(width, height, xmin=xmin, ymin=ymin, wcs=wcs_gs)
        survey_name = self.config.survey_name

        # Convert ra/dec to pixel positions using the provided WCS
        pix_x, pix_y = wcs.skyToPixelArray(
            galaxy_catalog.data["ra"],
            galaxy_catalog.data["dec"],
            degrees=True,
        )

        for i, src in enumerate(galaxy_catalog.data):
            ix = pix_x[i]
            iy = pix_y[i]
            if (
                ((xmin - SIM_INCLUSION_PADDING) < ix < (xmax + SIM_INCLUSION_PADDING))
                and ((ymin - SIM_INCLUSION_PADDING) < iy < (ymax + SIM_INCLUSION_PADDING))
                and src["has_finite_shear"]
            ):
                image_pos = galsim.PositionD(x=ix, y=iy)
                gal_obj = galaxy_catalog.get_obj(
                    ind=i,
                    mag_zero=mag_zero,
                    band=band,
                    use_mog=self.config.use_mog,
                    force_isotropic=self.config.force_isotropic,
                    include_point_source=self.config.include_point_source,
                    survey_name=survey_name,
                )
                stamp = draw_ia(
                    amplitude=self.config.ia_amplitude,
                    beta=self.config.ia_beta,
                    phi=self.config.ia_phi,
                    clip_radius=self.config.ia_clip_radius,
                    stamp_size=DEFAULT_BAT_STAMP_SIZE,
                    gal_obj=gal_obj,
                    psf_obj=psf_obj,
                    image_pos=image_pos,
                    draw_method=draw_method,
                    pixel_scale=galaxy_catalog.pixel_scale,
                    entry=src,
                )
                b = stamp.bounds & image.bounds
                if b.isDefined():
                    image[b] += stamp[b]
        return image.array
