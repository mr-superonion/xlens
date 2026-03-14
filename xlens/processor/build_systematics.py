from typing import Any

import anacal
import astropy.units as u
import numpy as np
from lsst.afw.image import ExposureF, MaskX
from lsst.geom import Box2I, Extent2I, Point2D, Point2I
from lsst.meas.algorithms import (
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
)
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import (
    ConfigField,
    Field,
    FieldValidationError,
    ListField,
)
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base import connectionTypes as cT

import xlens
from xlens.utils.image import resize_array, subpixel_shift

band_order = "ugrizy"


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
    outputPsf = cT.Output(
        doc="Stacked PSF array (6 x npix x npix).",
        name="deep_coadd_systematics_psfcentered_6bands_cell",
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
        default=49,
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
        default=150.0,
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

    def validate(self):
        super().validate()
        if self.npix % 2 == 0:
            raise FieldValidationError(
                self.__class__.npix, self, "npix should be odd number"
            )


class BuildSystematicsTask(PipelineTask):
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
        exposure_handles_dict = {
            handle.dataId["band"]: handle for handle in exposure_handles
        }

        cell_handles = inputs["cellexposure"]
        if len(cell_handles) == 0:
            cell_handles_dict = None
        else:
            cell_handles_dict = {h.dataId["band"]: h for h in cell_handles}
        outputs = self.run(
            exposure_handles_dict=exposure_handles_dict,
            tract=tract,
            patch=patch,
            gaia_loader=gaia_loader,
            cell_handles_dict=cell_handles_dict,
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
        cell_handles_dict: None | dict[str, Any] = None,
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

            band_mask = self._build_mask_band(
                exposure=exp,
            )
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
            for i, band in enumerate(band_order):
                if band in cell_handles_dict.keys():
                    cell_coadd = cell_handles_dict[band].get()
                    psf_array[i] = xlens.utils.image.stack_psfs_cells(
                        cell_coadd=cell_coadd,
                        npix=npix,

                    )
                    del cell_coadd
        else:
            psf_array = None

        # noise correlation
        for band, exp_handle in exposure_handles_dict.items():
            exp = exp_handle.get()
            if band in band_order:
                i = band_order.index(band)
                noise_corr_array[i] = self.get_noise_corr(exp, mask_array)
            del exp

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

    def get_noise_corr(self, exposure, mask_array):
        assert isinstance(self.config, BuildSystematicsConfig)
        mask = exposure.mask

        # Always check what planes exist in this exposure:
        print(mask.getMaskPlaneDict().keys())

        planes = [
            "BAD", "CR", "NO_DATA", "SAT", "UNMASKEDNAN",
            "DETECTED", "DETECTED_NEGATIVE"
        ]
        avail = set(mask.getMaskPlaneDict().keys())
        planes = [p for p in planes if p in avail]

        bits = mask.getPlaneBitMask(planes)
        variance_array = exposure.getMaskedImage().variance.array[
            1000:3000, 1000:3000
        ]
        window_array = (
            ((mask.array & bits) == 0) & (mask_array == 0)
        ).astype(np.float32)[
            1000:3000, 1000:3000
        ]

        noise_array = np.asarray(
            exposure.getMaskedImage().image.array,
            dtype=np.float32,
        )[1000:3000, 1000:3000]
        window_array = (
            window_array
            * (noise_array**2.0 < variance_array * 9)
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
        good = window_corr > 0
        noise_corr2 = np.zeros_like(window_corr, dtype=np.float32)
        noise_corr2[good] = noise_corr[good] / window_corr[good]
        del window_array, noise_array, window_corr
        return noise_corr2

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
            np.random.seed(seed)
            ind = np.random.randint(0, nstars)
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
