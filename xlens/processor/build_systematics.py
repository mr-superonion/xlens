from typing import Any

import anacal
import astropy.units as u
import lsst.afw.image as afwImage
import numpy as np
from lsst.afw.image import ExposureF, MaskX
from lsst.geom import Box2I, Extent2I, Point2D, Point2I
from lsst.meas.algorithms import (
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
)
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import ConfigField, Field, ListField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base import connectionTypes as cT

import xlens
from xlens.utils.image import resize_array, subpixel_shift


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
    cellexposure = cT.Input(
        doc="Input cell coadd exposure to build systematics mask from.",
        name="{coaddName}_coadd_cell_predetection",
        storageClass="MultipleCellCoadd",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        deferLoad=True,
        minimum=0,
    )
    catalog = cT.Input(
        doc="Catalog containing single-band measurement information.",
        name="{coaddName}_coadd_meas",
        storageClass="SourceCatalog",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        deferLoad=True,
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
    outputPsf = cT.Output(
        doc="Stacked PSF array (6 x npix x npix).",
        name="deep_coadd_systematics_psfcentered",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
    )
    outputNoiseCorr = cT.Output(
        doc="Stacked noise correlation array (6 x npix x npix).",
        name="deep_coadd_systematics_noisecorr_stack",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
    )
    outputPsfCentered = cT.Output(
        doc="Stacked PSF image array (6 x npix x npix).",
        name="deep_coadd_systematics_psfcentered_stack",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
    )
    outputStarCentered = cT.Output(
        doc="Stacked star image array (6 x npix x npix).",
        name="deep_coadd_systematics_starcentered_stack",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class BuildSystematicsConfig(
    PipelineTaskConfig, pipelineConnections=BuildSystematicsConnections
):
    """Configuration for :class:`BuildSystematicsTask`."""

    npix = Field[int](
        doc="number of pixels in stamp",
        default=64,
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes used to reject bad pixels.",
        default=["BAD", "CR", "NO_DATA", "SAT", "UNMASKEDNAN",],
    )
    gaiaPadding = Field[int](
        doc="Padding (pixels) when selecting GAIA sources around the patch.",
        default=300,
    )
    psfCache = Field[int](
        doc="Size of PSF cache",
        default=100,
    )
    star_snr_min = Field[float](
        doc="minimum (aperture) snr threshold of stars",
        default=100.0,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()
    gaiaLoader = ConfigField(
        dtype=LoadReferenceObjectsConfig,
        doc="Reference catalog loader",
    )

    def setDefaults(self):
        super().setDefaults()
        self.gaiaLoader.requireProperMotion = False
        self.gaiaLoader.anyFilterMapsToThis = "phot_g_mean"


class BuildSystematicsTask(PipelineTask):
    """Collect mask information from exposures, including bright star
    masking.
    """

    _DefaultName = "BuildSystematicsTask"
    ConfigClass = BuildSystematicsConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs, **kwargs):
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
        exposure_handles_dict = {
            handle.dataId["band"]: handle for handle in exposure_handles
        }

        cell_handles = inputs["cellexposure"]
        if len(cell_handles) == 0:
            cell_handles_dict = None
        else:
            cell_handles_dict = {h.dataId["band"]: h for h in cell_handles}
        catalog_handles = inputs["catalog"]
        if len(catalog_handles) == 0:
            catalog_handles_dict = None
        else:
            catalog_handles_dict = {
                handle.dataId["band"]: handle for handle in catalog_handles
            }
        outputs = self.run(
            exposure_handles_dict=exposure_handles_dict,
            gaia_loader=gaia_loader,
            cell_handles_dict=cell_handles_dict,
            catalog_handles_dict=catalog_handles_dict,
            seed=seed,
        )
        butlerQC.put(outputs, outputRefs)
        return

    def run(
        self,
        *,
        exposure_handles_dict: dict[str, Any],
        gaia_loader: ReferenceObjectLoader | None = None,
        cell_handles_dict: None | dict[str, Any] = None,
        catalog_handles_dict: None | dict[str, Any] = None,
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

        band_order = "ugrizy"
        for band, exp_handle in exposure_handles_dict.items():
            exp = exp_handle.get()

            if (template_wcs is None) and (template_bbox is None):
                template_wcs = exp.getWcs()
                template_bbox = exp.getBBox()

            band_mask = self._build_mask_band(
                exposure=exp,
            )
            mask_array = self._merge_mask(mask_array, band_mask)

            if band in band_order:
                i = band_order.index(band)
                noise_image = self.get_noise_corr(exp)
                noise_corr_array[i] = noise_image.array

                if (
                    catalog_handles_dict is not None
                    and band in catalog_handles_dict
                ):
                    catalog = catalog_handles_dict[band].get()
                    psf_image, star_image = self.get_psf_systematics(
                        exp,
                        catalog,
                        seed,
                    )
                    if psf_image is not None:
                        psf_centered_array[i] = psf_image.array
                    if star_image is not None:
                        star_centered_array[i] = star_image.array
                    del catalog
            del exp, band_mask

        if (
            template_wcs is not None
            and template_bbox is not None
            and gaia_loader is not None
        ):
            gaia = gaia_loader.loadPixelBox(
                bbox=template_bbox,
                filterName="phot_g_mean",
                wcs=template_wcs,
                bboxToSpherePadding=self.config.gaiaPadding,
            ).refCat
            gaia_array = self._get_gaia_mask_sources(
                wcs=template_wcs,
                bbox=template_bbox,
                gaia_catalog=gaia,
            )
            if gaia_array is not None:
                anacal.mask.add_bright_star_mask(
                    mask_array=mask_array, star_array=gaia_array
                )

        assert mask_array is not None
        h, w = mask_array.shape
        output_msk = MaskX(width=w, height=h)
        output_msk.getArray()[:, :] = mask_array.astype(
            output_msk.getArray().dtype,
            copy=False
        )

        if cell_handles_dict is not None:
            psf_array = np.zeros((6, npix, npix))
            for i, band in enumerate("ugrizy"):
                if band in cell_handles_dict.keys():
                    cell_coadd = cell_handles_dict[band].get()
                    psf_array[i] = xlens.utils.image.stack_psfs_cells(
                        cell_coadd=cell_coadd,
                        npix=npix,

                    )
                    del cell_coadd
        else:
            psf_array = None
        return Struct(
            outputMask=output_msk,
            outputPsf=psf_array,
            outputNoiseCorr=noise_corr_array,
            outputPsfCentered=psf_centered_array,
            outputStarCentered=star_centered_array,
        )

    def _merge_mask(
        self, global_mask: np.ndarray | None, band_mask: np.ndarray,
    ):
        if global_mask is None:
            return band_mask.astype(np.int16)
        return (global_mask | band_mask).astype(np.int16)

    def _build_mask_band(self, *, exposure: ExposureF) -> np.ndarray:
        assert isinstance(self.config, BuildSystematicsConfig)
        bitv = exposure.mask.getPlaneBitMask(self.config.badMaskPlanes)
        mask_band = ((exposure.mask.array & bitv) != 0).astype(np.int16)
        return mask_band

    def _get_gaia_mask_sources(
        self,
        *,
        wcs,
        bbox,
        gaia_catalog: Any,
    ) -> np.ndarray | None:
        assert isinstance(self.config, BuildSystematicsConfig)

        gaia_astropy = gaia_catalog.asAstropy()
        flux = gaia_astropy["phot_g_mean_flux"]
        mag = (np.asarray(flux) * u.nJy).to_value(u.ABmag)
        x, y = wcs.skyToPixelArray(
            ra=gaia_astropy["coord_ra"] * 180 / np.pi,
            dec=gaia_astropy["coord_dec"] * 180 / np.pi,
            degrees=True,
        )
        mask = (mag <= 17.0)
        if not np.any(mask):
            return None

        x = x[mask] - bbox.getBeginX()
        y = y[mask] - bbox.getBeginY()
        mag = mag[mask]
        conds = [
            mag <= 11.0, (mag > 11.0) & (mag <= 14.0),
            (mag > 14.0) & (mag <= 17.0)
        ]
        choices = [450.0, 200.0, 100.0]
        r = np.select(conds, choices, default=100.0)
        dtype = np.dtype([("x", float), ("y", float), ("r", float)])
        xy_r = np.zeros(len(x), dtype=dtype)
        xy_r["x"] = x
        xy_r["y"] = y
        xy_r["r"] = r
        return xy_r

    def get_noise_corr(self, exposure):
        assert isinstance(self.config, BuildSystematicsConfig)
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
            window_array
            * (noise_array**2.0 < variance_array * 9)
            * (variance_array < 5.0)
            * (~np.isnan(variance_array))
        )

        noise_array[~window_array.astype(bool)] = 0.0
        noise_variance = np.average(variance_array[window_array.astype(bool)])
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
            ny // 2 - npixl : ny // 2 + npixr,
            nx // 2 - npixl : nx // 2 + npixr,
        ]
        window_corr = np.fft.fftshift(
            np.fft.ifft2(np.abs(np.fft.fft2(window_array)) ** 2.0)
        ).real[
            ny // 2 - npixl : ny // 2 + npixr,
            nx // 2 - npixl : nx // 2 + npixr,
        ]
        noise_corr = noise_corr / window_corr
        del window_array, noise_array, window_corr

        noise_image = afwImage.ImageF(self.config.npix, self.config.npix)
        noise_image.array[:, :] = noise_corr

        return noise_image

    def get_psf_systematics(self, exposure, catalog, seed):
        assert isinstance(self.config, BuildSystematicsConfig)
        if seed is None:
            raise ValueError("Seed is required to select a random star.")
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
