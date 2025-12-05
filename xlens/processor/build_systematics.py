from typing import Any

import astropy.units as u
import anacal
import numpy as np
from lsst.afw.image import ExposureF, ImageI
from lsst.pex.config import Field, ListField, ConfigField
from lsst.meas.algorithms import (
    ReferenceObjectLoader, LoadReferenceObjectsConfig,
)
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base import connectionTypes as cT
from lsst.afw.image import MaskX


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
    gaia = cT.PrerequisiteInput(
        doc="GAIA sources to load",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    outputMask = cT.Output(
        doc="Combined mask from bad pixels and bright stars across all bands.",
        name="{coaddName}_systematics_mask",
        storageClass="Mask",
        dimensions=("skymap", "tract", "patch"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)



class BuildSystematicsConfig(
    PipelineTaskConfig, pipelineConnections=BuildSystematicsConnections
):
    """Configuration for :class:`BuildSystematicsTask`."""

    badMaskPlanes = ListField[str](
        doc="Mask planes used to reject bad pixels.",
        default=["BAD", "CR", "NO_DATA", "SAT", "UNMASKEDNAN",],
    )
    gaiaPadding = Field[int](
        doc="Padding (pixels) when selecting GAIA sources around the patch.",
        default=300,
    )
    gaiaLoader = ConfigField(
        dtype=LoadReferenceObjectsConfig,
        doc="Reference catalog loader",
    )

    def setDefaults(self):
        super().setDefaults()
        self.gaiaLoader.requireProperMotion = False
        self.gaiaLoader.anyFilterMapsToThis = "phot_g_mean"


class BuildSystematicsTask(PipelineTask):
    """Collect mask information from exposures, including bright star masking."""

    _DefaultName = "BuildSystematicsTask"
    ConfigClass = BuildSystematicsConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs, **kwargs):
        inputs = butlerQC.get(inputRefs)
        dataId = butlerQC.quantum.dataId
        gaia_loader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.gaia],
            refCats=inputs.pop("gaia"),
            name="gaia_dr3_20230707",
            config=self.config.gaiaLoader,
        )
        exposure_handles = inputs["exposure"]
        exposure_handles_dict = {
            handle.dataId["band"]: handle for handle in exposure_handles
        }
        outputs = self.run(
            exposureHandles=exposure_handles_dict,
            gaia_loader=gaia_loader,
        )
        butlerQC.put(outputs, outputRefs)
        return

    def run(
        self,
        *,
        exposureHandles: dict[str, Any],
        gaia_loader: ReferenceObjectLoader,
        **kwargs,
    ) -> Struct:
        assert isinstance(self.config, BuildSystematicsConfig)

        mask_array: np.ndarray | None = None
        template_wcs = None
        template_bbox = None

        for band, exp_handle in exposureHandles.items():
            exp = exp_handle.get()

            if (template_wcs is None) and (template_bbox is None):
                template_wcs = exp.getWcs()
                template_bbox = exp.getBBox()

            band_mask = self._build_mask_band(
                exposure=exp,
            )
            mask_array = self._merge_mask(mask_array, band_mask)
            del exp, band_mask

        if template_wcs is not None and template_bbox is not None:
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
        return Struct(outputMask=output_msk)

    def _merge_mask(self, global_mask: np.ndarray | None, band_mask: np.ndarray):
        if global_mask is None:
            return band_mask.astype(np.int16)
        return (global_mask | band_mask).astype(np.int16)

    def _build_mask_band(self, *, exposure: ExposureF) -> np.ndarray:
        assert isinstance(self.config, BuildSystematicsConfig)
        bitv = exposure.mask.getPlaneBitMask(self.config.badMaskPlanes)
        mask_band = (
            ((exposure.mask.array & bitv) != 0)
            | (
                np.abs(exposure.image.array) > (
                    6.0 * np.sqrt(
                        np.where(
                            exposure.variance.array < 0,
                            0, exposure.variance.array,
                        )
                    )
                )
            )
        ).astype(np.int16)
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
        print(len(gaia_astropy))
        mask = (mag <= 17.0)
        if not np.any(mask):
            return None
        print(np.sum(mask))

        x = x[mask] - bbox.getBeginX()
        y = y[mask] - bbox.getBeginY()
        mag = mag[mask]
        conds = [
            mag <= 11.0, (mag > 11.0) & (mag <= 14.0),
            (mag > 14.0) & (mag <= 17.0)
        ]
        choices = [266.0, 185.0, 100.0]
        r = np.select(conds, choices, default=100.0)
        dtype = np.dtype([("x", float), ("y", float), ("r", float)])
        xy_r = np.zeros(len(x), dtype=dtype)
        xy_r["x"] = x
        xy_r["y"] = y
        xy_r["r"] = r
        return xy_r
