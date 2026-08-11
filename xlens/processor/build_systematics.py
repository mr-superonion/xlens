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

from typing import Any

import numpy as np
from lsst.afw.image import MaskX
from lsst.geom import Box2I, Extent2I, Point2D, Point2I
from lsst.meas.algorithms import ReferenceObjectLoader
from lsst.pex.config import Field
from lsst.pipe.base import (
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base import connectionTypes as cT

from xlens.utils.image import resize_array, subpixel_shift

from .systematics_base import (
    BuildSystematicsConfigBase,
    BuildSystematicsTaskBase,
    band_order,
)


class BuildSystematicsConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "coaddName": "deep",
    },
):
    exposure = cT.Input(
        doc="Input coadd exposure to build systematics mask from.",
        name="{coaddName}_coadd",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        deferLoad=True,
    )
    catalog = cT.Input(
        doc="Catalog containing single-band measurement information.",
        name="object",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
        minimum=0,
    )
    gaia = cT.PrerequisiteInput(
        doc="GAIA sources to load",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
        minimum=0,
    )
    outputMask = cT.Output(
        doc="Combined mask from bad pixels and bright stars across all bands.",
        name="deep_coadd_systematics_mask",
        storageClass="Mask",
        dimensions=("skymap", "tract", "patch"),
    )
    outputNoiseCorr = cT.Output(
        doc="Stacked noise correlation array (6 x npix x npix).",
        name="deep_coadd_systematics_noisecorr_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
    )
    outputPsfCentered = cT.Output(
        doc="Stacked PSF image array (6 x npix x npix).",
        name="deep_coadd_systematics_psfcentered_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
    )
    outputStarCentered = cT.Output(
        doc="Stacked star image array (6 x npix x npix).",
        name="deep_coadd_systematics_starcentered_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class BuildSystematicsConfig(
    BuildSystematicsConfigBase,
    PipelineTaskConfig,
    pipelineConnections=BuildSystematicsConnections,
):
    """Configuration for :class:`BuildSystematicsTask`."""

    psfCache = Field[int](
        doc="Size of PSF cache",
        default=100,
    )
    star_snr_min = Field[float](
        doc="minimum (aperture) snr threshold of stars",
        default=150.0,
    )


class BuildSystematicsTask(BuildSystematicsTaskBase):
    """Collect mask information from exposures, including bright star
    masking.
    """

    _DefaultName = "BuildSystematicsTask"
    ConfigClass = BuildSystematicsConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs, **kwargs):
        tract = int(butlerQC.quantum.dataId["tract"])
        patch = int(butlerQC.quantum.dataId["patch"])
        inputs = butlerQC.get(inputRefs)
        if len(inputs["gaia"]) > 0:
            gaia_loader = ReferenceObjectLoader(
                dataIds=[ref.datasetRef.dataId for ref in inputRefs.gaia],
                refCats=inputs.pop("gaia"),
                name="gaia_dr3_20230707",
                config=self.config.gaiaLoader,
            )
        else:
            gaia_loader = None
        id_generator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        seed = id_generator.catalog_id
        exposure_handles = inputs["exposure"]
        exposure_handles_dict = {handle.dataId["band"]: handle for handle in exposure_handles}

        outputs = self.run(
            exposure_handles_dict=exposure_handles_dict,
            tract=tract,
            patch=patch,
            gaia_loader=gaia_loader,
            catalog=inputs["catalog"],
            seed=seed,
        )
        butlerQC.put(outputs, outputRefs)
        return

    def run(
        self,
        *,
        tract: int,
        patch: int,
        exposure_handles_dict: dict[str, Any],
        gaia_loader: ReferenceObjectLoader | None = None,
        catalog=None,
        seed: int | None = None,
        **kwargs,
    ) -> Struct:
        assert isinstance(self.config, BuildSystematicsConfig)

        mask_array: np.ndarray | None = None
        template_wcs = None
        template_bbox = None

        npix = self.config.npix
        noise_corr_array = np.zeros((6, npix, npix))
        psf_centered_array = np.zeros((6, npix, npix))
        star_centered_array = np.zeros((6, npix, npix))
        if catalog is not None:
            catalog = catalog[catalog["patch"] == patch]
            ngood = np.zeros(len(catalog))
            for b in band_order:
                snr = catalog[f"{b}_psfFlux"] / catalog[f"{b}_psfFluxErr"]
                test = (
                    (catalog[f"{b}_calib_psf_candidate"])
                    & (snr > self.config.star_snr_min)
                    & (~catalog[f"{b}_psfFlux_flag"])
                    & (~catalog[f"{b}_hsmShapeRegauss_flag"])
                )
                ngood += (test).astype(int)
            catalog = catalog[ngood == ngood.max()]

        # Mask, PSF and Stars
        for band, exp_handle in exposure_handles_dict.items():
            exp = exp_handle.get()

            if (template_wcs is None) and (template_bbox is None):
                template_wcs = exp.getWcs()
                template_bbox = exp.getBBox()

            band_mask = self._build_mask_band(exp, band)
            mask_array = self._merge_mask(mask_array, band_mask)

            if band in band_order:
                i = band_order.index(band)
                if catalog is not None:
                    psf_array, star_array = self.get_psf_systematics(
                        exp,
                        catalog,
                        seed,
                        band=band,
                    )
                    if psf_array is not None:
                        psf_centered_array[i] = psf_array
                    if star_array is not None:
                        star_centered_array[i] = star_array
            del exp, band_mask
        assert mask_array is not None
        self._apply_gaia_mask(
            mask_array=mask_array,
            bbox=template_bbox,
            wcs=template_wcs,
            gaia_loader=gaia_loader,
        )
        h, w = mask_array.shape
        output_msk = MaskX(width=w, height=h)
        output_msk.getArray()[:, :] = mask_array.astype(output_msk.getArray().dtype, copy=False)

        # noise correlation
        for band, exp_handle in exposure_handles_dict.items():
            exp = exp_handle.get()
            if band in band_order:
                i = band_order.index(band)
                noise_corr_array[i] = self.get_noise_corr(exp, mask_array)
            del exp

        return Struct(
            outputMask=output_msk,
            outputNoiseCorr=noise_corr_array,
            outputPsfCentered=psf_centered_array,
            outputStarCentered=star_centered_array,
        )

    def get_noise_corr(self, exposure, mask_array):
        """Noise correlation from a fixed central [1000:3000] window.

        NOTE the window's plane list below is hardcoded and does NOT
        follow ``badMaskPlanes``; ``BuildCellSystematicsTask`` derives
        its own window from the config instead.  Left as-is so this
        refactor does not silently change the patch-path numerics.
        """
        assert isinstance(self.config, BuildSystematicsConfig)
        mask = exposure.mask

        planes = ["BAD", "CR", "NO_DATA", "SAT", "UNMASKEDNAN", "DETECTED", "DETECTED_NEGATIVE"]
        avail = set(mask.getMaskPlaneDict().keys())
        planes = [p for p in planes if p in avail]

        bits = mask.getPlaneBitMask(planes)
        variance_array = exposure.getMaskedImage().variance.array[1000:3000, 1000:3000]
        window_array = (((mask.array & bits) == 0) & (mask_array == 0)).astype(np.float32)[
            1000:3000, 1000:3000
        ]

        # ``.copy()`` is load-bearing: the exposure image is already
        # float32, so np.asarray returns a VIEW and the zeroing below
        # would write through into the CALLER's pixels -- blanking every
        # DETECTED source in the central window of an exposure the
        # caller still intends to measure.  (build_cell_systematics
        # copies here for the same reason.)
        noise_array = np.array(
            exposure.getMaskedImage().image.array[1000:3000, 1000:3000],
            dtype=np.float32,
        )
        window_array = window_array * (noise_array**2.0 < variance_array * 9) * (~np.isnan(variance_array))

        noise_array[~window_array.astype(bool)] = 0.0
        noise_variance = np.average(variance_array[window_array.astype(bool)])
        if noise_variance < 1e-20:
            raise ValueError("the estimated image noise variance should be positive.")

        return self._correlate(noise_array, window_array, self.config.npix)

    def get_psf_systematics(self, exposure, catalog, seed, band):
        assert isinstance(self.config, BuildSystematicsConfig)
        if seed is None:
            raise ValueError("Seed is required to select a random star.")
        npixl = int(self.config.npix // 2)
        npixr = int(self.config.npix // 2 + 1)
        bbox = exposure.getBBox()
        xmin_exp, ymin_exp = bbox.getMinX(), bbox.getMinY()
        xmax_exp, ymax_exp = bbox.getMaxX(), bbox.getMaxY()
        msk = (
            (catalog[f"{band}_centroid_x"] > xmin_exp + npixl)
            & (catalog[f"{band}_centroid_y"] > ymin_exp + npixl)
            & (catalog[f"{band}_centroid_x"] < xmax_exp - npixr)
            & (catalog[f"{band}_centroid_y"] < ymax_exp - npixr)
        )
        catalog = catalog[msk]
        nstars = len(catalog)

        if nstars >= 1:
            rng = np.random.RandomState(seed)
            ind = rng.randint(0, nstars)
            src = catalog[ind]
            # Collect the PSF image
            lsst_psf = exposure.getPsf()
            psf_array = lsst_psf.computeImage(
                Point2D(
                    int(src[f"{band}_centroid_x"]),
                    int(src[f"{band}_centroid_y"]),
                )
            ).getArray()
            psf_array = resize_array(
                psf_array,
                (self.config.npix, self.config.npix),
            )

            bbox = Box2I(
                Point2I(
                    int(src[f"{band}_centroid_x"]) - npixl,
                    int(src[f"{band}_centroid_y"]) - npixl,
                ),
                Extent2I(self.config.npix, self.config.npix),
            )
            # Collect the star image
            # Extract the sub-image using the BBox
            star_image = exposure.Factory(exposure, bbox).getImage()
            # Get the image component and convert to a NumPy array
            star_array = star_image.getArray()
            xn = f"{band}_centroid_x"
            yn = f"{band}_centroid_y"
            offset_x = src[xn] - int(src[xn])
            offset_y = src[yn] - int(src[yn])
            star_array = subpixel_shift(star_array, -offset_x, -offset_y)
        else:
            psf_array = None
            star_array = None
        return psf_array, star_array
