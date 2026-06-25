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

"""Build systematics products from cell-based coadds.

Estimates noise correlation functions per band from stitched cell coadd
images, stacks per-cell PSFs, and builds bright star masks from GAIA.
"""

__all__ = [
    "BuildCellSystematicsConfig",
    "BuildCellSystematicsTask",
    "BuildCellSystematicsConnections",
]

from typing import Any

import anacal
import numpy as np
from lsst.afw.image import MaskX
from lsst.meas.algorithms import (
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
)
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import (
    ChoiceField,
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
from lsst.skymap import BaseSkyMap

from xlens.utils.image import stack_psfs_cells
from xlens.utils.mask import (
    GAIA_TABLE_DTYPE,
    STAR_MASK_RADIUS_FUNCS,
    build_gaia_xyr,
    get_gaia_table,
)

band_order = "ugrizy"


class BuildCellSystematicsConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={"coaddName": "deep"},
):
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    cellCoadd = cT.Input(
        doc="Input cell-based coadd image",
        name="{coaddName}_coadd_cell_predetection",
        storageClass="MultipleCellCoadd",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        deferLoad=True,
    )
    gaia = cT.PrerequisiteInput(
        doc="GAIA sources for bright star masking",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
        minimum=0,
    )
    outputMask = cT.Output(
        doc="Combined mask from bad pixels across all bands on stitched image.",
        name="deep_coadd_cell_systematics_mask",
        storageClass="Mask",
        dimensions=("skymap", "tract", "patch"),
    )
    outputNoiseCorr = cT.Output(
        doc="Stacked noise correlation array (6 x npix x npix).",
        name="deep_coadd_cell_systematics_noisecorr_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
    )
    outputPsf = cT.Output(
        doc="Stacked PSF array from cell coadds (6 x npix x npix).",
        name="deep_coadd_cell_systematics_psfcentered_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
    )
    outputGaiaCatalog = cT.Output(
        doc=(
            "GAIA sources covering this patch. Columns: x_in_tract, "
            "y_in_tract (tract-pixel coordinates), gaia_g_mag, "
            "gaia_source_id (Gaia DR3 source_id, int64), ra, dec (deg). "
            "Empty when no GAIA refcat is in the inputs."
        ),
        name="deep_coadd_cell_systematics_gaia",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class BuildCellSystematicsConfig(
    PipelineTaskConfig,
    pipelineConnections=BuildCellSystematicsConnections,
):
    npix = Field[int](
        doc="Size of noise correlation and PSF stamps (must be odd).",
        default=49,
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes used to reject bad pixels.",
        default=[
            "BAD",
            "SAT",
            "CR",
            "NO_DATA",
            "UNMASKEDNAN",
            "CROSSTALK",
            "INTRP",
            "STREAK",
            "VIGNETTED",
            "CLIPPED",
        ],
    )
    gaiaPadding = Field[int](
        doc="Padding (pixels) when selecting GAIA sources around the patch.",
        default=300,
    )
    bands = ListField[str](
        doc=(
            "Bands required to be present in the input cell coadd dict. "
            "The task raises if the set of bands actually delivered by "
            "the butler does not match this list (no partial-band runs)."
        ),
        default=["g", "r", "i", "z"],
    )
    gaiaLoader = ConfigField(
        dtype=LoadReferenceObjectsConfig,
        doc="Reference catalog loader for GAIA",
    )
    starMaskType = ChoiceField[str](
        doc=(
            "Name of the GAIA halo-radius model in "
            "xlens.utils.mask.STAR_MASK_RADIUS_FUNCS. 'default' = "
            "450/200/100 px step for mag <= 11/14/20; 'no_mask' = "
            "flat 10 px for every GAIA star with mag <= 20."
        ),
        allowed={k: k for k in STAR_MASK_RADIUS_FUNCS},
        default="default",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def setDefaults(self):
        super().setDefaults()
        self.gaiaLoader.requireProperMotion = False
        self.gaiaLoader.anyFilterMapsToThis = "phot_g_mean"

    def validate(self):
        super().validate()
        if self.npix % 2 == 0:
            raise FieldValidationError(
                self.__class__.npix,
                self,
                "npix should be odd number",
            )


class BuildCellSystematicsTask(PipelineTask):
    """Build noise correlation and PSF systematics from cell-based coadds.

    For each band, the task stitches the full-patch cell coadd into a
    contiguous image and estimates the noise correlation function via
    FFT-based autocorrelation. PSFs are stacked from individual cell
    PSF images.
    """

    _DefaultName = "BuildCellSystematicsTask"
    ConfigClass = BuildCellSystematicsConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        patch = int(butlerQC.quantum.dataId["patch"])

        cell_handles = inputs["cellCoadd"]
        cell_handles_dict = {h.dataId["band"]: h for h in cell_handles}
        if len(inputs["gaia"]) > 0:
            gaia_loader = ReferenceObjectLoader(
                dataIds=[ref.datasetRef.dataId for ref in inputRefs.gaia],
                refCats=inputs.pop("gaia"),
                name="gaia_dr3_20230707",
                config=self.config.gaiaLoader,
            )
        else:
            gaia_loader = None
            self.log.warning(
                "No GAIA reference catalog found for tract=%d patch=%d. "
                "Bright star masking will be skipped. "
                "Ensure refcats/DM-39298/gaia_dr3_20230707 is in input "
                "collections.",
                tract,
                patch,
            )
        outputs = self.run(
            cell_handles_dict=cell_handles_dict,
            skyMap=inputs["skyMap"],
            tract=tract,
            patch=patch,
            gaia_loader=gaia_loader,
        )
        butlerQC.put(outputs, outputRefs)

    def get_noise_corr(self, stitched_coadd, mask_array, badMaskPlanes):
        """Estimate noise correlation from a stitched cell coadd.

        Parameters
        ----------
        stitched_coadd : StitchedCoadd
            Full-patch stitched cell coadd.
        mask_array : np.ndarray
            Combined systematics mask (incl. cross-band bad pixels and
            GAIA bright-star halos); nonzero pixels are excluded.
        badMaskPlanes : list of str
            Mask plane names to exclude.

        Returns
        -------
        noise_corr : np.ndarray
            Noise correlation array of shape (npix, npix).
        """
        assert isinstance(self.config, BuildCellSystematicsConfig)
        npix = self.config.npix

        exp = stitched_coadd.asExposure()
        mask = exp.mask
        image_array = np.asarray(exp.image.array, dtype=np.float32)
        variance_array = exp.variance.array

        # Build window mask
        avail = set(mask.getMaskPlaneDict().keys())
        planes = [p for p in badMaskPlanes if p in avail]
        # Also exclude detected sources
        for extra in ["DETECTED", "DETECTED_NEGATIVE"]:
            if extra in avail:
                planes.append(extra)
        bits = mask.getPlaneBitMask(planes)

        # Use central region to avoid edge effects
        ny, nx = image_array.shape
        y0 = min(1000, ny // 4)
        y1 = max(ny - 1000, 3 * ny // 4)
        x0 = min(1000, nx // 4)
        x1 = max(nx - 1000, 3 * nx // 4)

        noise_array = image_array[y0:y1, x0:x1].copy()
        variance_sub = variance_array[y0:y1, x0:x1]
        window_array = (((mask.array[y0:y1, x0:x1] & bits) == 0) & (mask_array[y0:y1, x0:x1] == 0)).astype(
            np.float32
        )
        window_array *= (noise_array**2.0 < variance_sub * 9) & (~np.isnan(variance_sub))

        # Mean-subtract over the kept pixels before zeroing the masked ones.
        # A nonzero DC offset would otherwise spread to a flat μ² pedestal
        # across every lag of the windowed autocorrelation.
        window_bool = window_array.astype(bool)
        if window_bool.any():
            noise_array -= noise_array[window_bool].mean()
        noise_array[~window_bool] = 0.0

        # Pad to avoid FFT wrap-around
        pad_width = ((10, 10), (10, 10))
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
        pny, pnx = window_array.shape

        npixl = npix // 2
        npixr = npix // 2 + 1

        noise_corr = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(noise_array)) ** 2.0)).real[
            pny // 2 - npixl : pny // 2 + npixr,
            pnx // 2 - npixl : pnx // 2 + npixr,
        ]
        window_corr = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(window_array)) ** 2.0)).real[
            pny // 2 - npixl : pny // 2 + npixr,
            pnx // 2 - npixl : pnx // 2 + npixr,
        ]

        good = window_corr > 0
        noise_corr2 = np.zeros_like(window_corr, dtype=np.float32)
        noise_corr2[good] = noise_corr[good] / window_corr[good]
        return noise_corr2

    def run(
        self,
        *,
        cell_handles_dict: dict,
        skyMap,
        tract: int,
        patch: int,
        gaia_loader: ReferenceObjectLoader | None = None,
        **kwargs,
    ) -> Struct:
        """Build noise correlation and PSF arrays from cell coadds.

        Parameters
        ----------
        cell_handles_dict : dict
            Mapping of band name to deferred MultipleCellCoadd handle.
        skyMap : BaseSkyMap
            Sky map used for processing.
        tract, patch : int
            Tract and patch identifiers.
        gaia_loader : ReferenceObjectLoader or None
            Loader for GAIA reference catalog for bright star masking.

        Returns
        -------
        Struct
            outputMask : MaskX
                Combined mask from stitched images across all bands,
                including bright star masks from GAIA.
            outputNoiseCorr : np.ndarray (6, npix, npix)
            outputPsf : np.ndarray (6, npix, npix)
        """
        assert isinstance(self.config, BuildCellSystematicsConfig)

        expected = set(self.config.bands)
        provided = set(cell_handles_dict.keys())
        if provided != expected:
            raise RuntimeError(
                f"band mismatch for tract={tract} patch={patch}: "
                f"expected {sorted(expected)}, "
                f"got {sorted(provided)} "
                f"(missing={sorted(expected - provided)}, "
                f"extra={sorted(provided - expected)})"
            )

        npix = self.config.npix

        noise_corr_array = np.zeros((6, npix, npix))
        psf_array = np.zeros((6, npix, npix))
        mask_array: np.ndarray | None = None
        stitched_bbox = None
        stitched_wcs = None

        # Pass 1: stitch each band ONE AT A TIME, accumulate the
        # combined bad-pixel mask + per-band PSF stack, then drop the
        # stitched coadd. Caching all bands' stitched coadds in memory
        # was costing ~6 * ~400 MB per quantum for the 6-band pipeline,
        # which OOMs the slurm worker block at 256-way concurrency.
        for band, handle in cell_handles_dict.items():
            if band not in band_order:
                continue
            i = band_order.index(band)
            self.log.info(
                "Pass 1 (mask + PSF), band %s for tract=%d patch=%d",
                band, tract, patch,
            )

            cell_coadd = handle.get()
            psf_array[i] = stack_psfs_cells(cell_coadd=cell_coadd, npix=npix)

            stitched = cell_coadd.stitch()
            exp = stitched.asExposure()

            if stitched_bbox is None:
                stitched_bbox = exp.getBBox()
                stitched_wcs = exp.getWcs()

            band_mask = self._build_mask_band(exp)
            if mask_array is None:
                mask_array = band_mask.astype(np.int16)
            else:
                mask_array = (mask_array | band_mask).astype(np.int16)

            # Critical: drop the heavy stitched coadd before the next band.
            del cell_coadd, exp, stitched, band_mask

        # Add GAIA bright-star wings to the combined mask BEFORE the
        # noise-correlation pass, so the bright-star halos don't leak
        # correlated power into the per-band estimate.
        assert mask_array is not None
        gaia_table = np.empty(0, dtype=GAIA_TABLE_DTYPE)
        if gaia_loader is not None and stitched_wcs is not None and stitched_bbox is not None:
            gaia = gaia_loader.loadPixelBox(
                bbox=stitched_bbox,
                filterName="phot_g_mean",
                wcs=stitched_wcs,
                bboxToSpherePadding=self.config.gaiaPadding,
            ).refCat
            gaia_table = get_gaia_table(gaia_catalog=gaia, wcs=stitched_wcs)
            gaia_array = build_gaia_xyr(
                gaia_table,
                bbox=stitched_bbox,
                star_mask_type=self.config.starMaskType,
            )
            if gaia_array is not None:
                self.log.info(
                    "Adding bright star mask for %d GAIA sources (starMaskType=%s)",
                    len(gaia_array), self.config.starMaskType,
                )
                anacal.mask.add_bright_star_mask(
                    mask_array=mask_array,
                    star_array=gaia_array,
                )

        # Pass 2: re-stitch ONE BAND AT A TIME and compute its noise
        # correlation against the augmented mask. Doubles the stitching
        # work vs. caching, but keeps peak memory at ~1 stitched coadd
        # instead of nbands.
        for band, handle in cell_handles_dict.items():
            if band not in band_order:
                continue
            i = band_order.index(band)
            self.log.info(
                "Pass 2 (noise corr), band %s for tract=%d patch=%d",
                band, tract, patch,
            )
            cell_coadd = handle.get()
            stitched = cell_coadd.stitch()
            noise_corr_array[i] = self.get_noise_corr(
                stitched,
                mask_array,
                self.config.badMaskPlanes,
            )
            del cell_coadd, stitched

        # Convert mask to MaskX
        h, w = mask_array.shape
        output_msk = MaskX(width=w, height=h)
        output_msk.getArray()[:, :] = mask_array.astype(
            output_msk.getArray().dtype,
            copy=False,
        )
        if stitched_bbox is not None:
            output_msk.setXY0(stitched_bbox.getMin())

        return Struct(
            outputMask=output_msk,
            outputNoiseCorr=noise_corr_array,
            outputPsf=psf_array,
            outputGaiaCatalog=gaia_table,
        )

    def _build_mask_band(self, exposure) -> np.ndarray:
        """Build a bad-pixel mask for one band from a stitched exposure."""
        assert isinstance(self.config, BuildCellSystematicsConfig)
        avail = set(exposure.mask.getMaskPlaneDict().keys())
        planes = [p for p in self.config.badMaskPlanes if p in avail]
        if not planes:
            return np.zeros(exposure.mask.array.shape, dtype=np.int16)
        bitv = exposure.mask.getPlaneBitMask(planes)
        return ((exposure.mask.array & bitv) != 0).astype(np.int16)
