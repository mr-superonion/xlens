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

import numpy as np
from lsst.afw.image import MaskX
from lsst.meas.algorithms import ReferenceObjectLoader
from lsst.pex.config import ListField
from lsst.pipe.base import (
    NoWorkFound,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base import connectionTypes as cT
from lsst.skymap import BaseSkyMap

from xlens.utils.image import (
    estimate_noise_variance,
    mask_to_rle_table,
    stack_psfs_cells,
)

from .systematics_base import (
    BuildSystematicsConfigBase,
    BuildSystematicsTaskBase,
    band_order,
)


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
        # Read the NATIVE cell coadd. Declaring MultipleCellCoadd makes
        # butler run the CellCoadd -> MultipleCellCoadd converter, which
        # raises "MultipleCellCoadd requires its bounding box to lie on
        # the cell grid" for patches whose bbox is not a whole number of
        # cells -- 36 hard failures in the first DP2 systematics run.
        # Nothing here needs per-cell image objects: the mask comes from
        # the full-patch exposure and the PSF from the cell grid.
        storageClass="CellCoadd",
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
        doc=(
            "Combined anacal bitmask on the stitched image, run-length "
            "encoded (y, x_start, x_end, value; x_end exclusive; shape "
            "decode with "
            "xlens.utils.image.rle_table_to_mask). Bit 0 (value 1): bad "
            "pixels across all bands, the only bit that cuts pixels. "
            "Bit 1 (value 2): union of the discontinuity planes "
            "(default INEXACT_PSF), stamped per source as "
            "n_mask_discontinuity, never cut."
        ),
        name="deep_coadd_cell_systematics_mask_rle",
        storageClass="ArrowAstropy",
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
        if config is not None and not config.do_noise_corr_estimation:
            # Do not advertise -- or write -- a product we did not
            # compute. Writing it as zeros would look like a real
            # estimate to anything that read it later; dropping the
            # connection makes the absence explicit, and the quantum
            # graph then simply has no such dataset.
            self.outputs.remove("outputNoiseCorr")


class BuildCellSystematicsConfig(
    BuildSystematicsConfigBase,
    PipelineTaskConfig,
    pipelineConnections=BuildCellSystematicsConnections,
):
    bands = ListField[str](
        doc=(
            "Bands required to be present in the input cell coadd dict. "
            "The task raises if the set of bands actually delivered by "
            "the butler does not match this list (no partial-band runs)."
        ),
        default=["g", "r", "i", "z"],
    )


class BuildCellSystematicsTask(BuildSystematicsTaskBase):
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
        # run() returns the two masks as pixel arrays (script-friendly);
        # the butler stores ONE combined bitmask (bit 0 = masked, bit 1
        # = discontinuity), run-length encoded with a value column.
        msk = outputs.outputMask
        combined = (msk.getArray() != 0).astype(np.uint8)
        combined |= (
            (np.asarray(outputs.outputDiscontinuityMask) != 0).astype(np.uint8)
            << 1
        )
        outputs.outputMask = mask_to_rle_table(combined)
        del outputs.outputDiscontinuityMask
        if not self.config.do_noise_corr_estimation:
            # matches the connection dropped in
            # BuildCellSystematicsConnections.__init__
            del outputs.outputNoiseCorr
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

        # Accepts either an afw Exposure (native CellCoadd via
        # to_legacy) or a StitchedCoadd (the legacy path).
        exp = (stitched_coadd if hasattr(stitched_coadd, "image")
               else stitched_coadd.asExposure())
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
        noise_variance = estimate_noise_variance(variance_array, mask, mask_array)
        window_array *= self._noise_window(noise_array, variance_sub, noise_variance)

        # Mean-subtract over the kept pixels before zeroing the masked ones.
        # A nonzero DC offset would otherwise spread to a flat μ² pedestal
        # across every lag of the windowed autocorrelation.
        window_bool = window_array.astype(bool)
        if window_bool.any():
            noise_array -= noise_array[window_bool].mean()
        noise_array[~window_bool] = 0.0

        return self._correlate(noise_array, window_array, npix)

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
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        if missing:
            # Incomplete coverage is data, not a mistake: a patch the
            # survey has not finished in every band simply has no work
            # here. NoWorkFound makes the executor SKIP the quantum
            # (and its downstream) instead of failing the run, so one
            # missing band does not take a whole submission down.
            raise NoWorkFound(
                f"tract={tract} patch={patch} is missing band(s) "
                f"{missing}; skipping (have {sorted(provided)})"
            )
        if extra:
            # Extra bands ARE a mistake: `bands` and the data query
            # disagree, and every patch would be measured with a
            # different band set. Fail loudly.
            raise RuntimeError(
                f"band mismatch for tract={tract} patch={patch}: "
                f"expected {sorted(expected)}, got {sorted(provided)} "
                f"(extra={extra}). Constrain the data query, e.g. "
                f"-d \"... AND band IN ('r','i','z')\"."
            )

        npix = self.config.npix

        noise_corr_array = (
            np.zeros((6, npix, npix))
            if self.config.do_noise_corr_estimation else None
        )
        psf_array = np.zeros((6, npix, npix))
        mask_array: np.ndarray | None = None
        disc_array: np.ndarray | None = None
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

            # to_legacy() gives the whole patch as an afw Exposure --
            # image, variance, mask planes and WCS -- with no stitching
            # and no cell-grid constraint.
            exp = cell_coadd.to_legacy()
            stitched = None

            if stitched_bbox is None:
                stitched_bbox = exp.getBBox()
                stitched_wcs = exp.getWcs()

            band_mask = self._build_mask_band(exp, band)
            mask_array = self._merge_mask(mask_array, band_mask)

            if self._discontinuity_band_selected(band):
                disc_band = self._plane_union_mask(exp, band)
                disc_array = (
                    disc_band if disc_array is None
                    else self._merge_mask(disc_array, disc_band)
                )
                del disc_band

            # Critical: drop the heavy stitched coadd before the next band.
            del cell_coadd, exp, stitched, band_mask

        # Add GAIA bright-star wings to the combined mask BEFORE the
        # noise-correlation pass, so the bright-star halos don't leak
        # correlated power into the per-band estimate.
        assert mask_array is not None
        gaia_table = self._apply_gaia_mask(
            mask_array=mask_array,
            bbox=stitched_bbox,
            wcs=stitched_wcs,
            gaia_loader=gaia_loader,
        )

        # Pass 2: re-stitch ONE BAND AT A TIME and compute its noise
        # correlation against the augmented mask. Doubles the stitching
        # work vs. caching, but keeps peak memory at ~1 stitched coadd
        # instead of nbands.
        if not self.config.do_noise_corr_estimation:
            self.log.info(
                "noise correlation estimation disabled; outputNoiseCorr "
                "stays zero for tract=%d patch=%d", tract, patch,
            )
        pass2 = (
            cell_handles_dict.items()
            if self.config.do_noise_corr_estimation else ()
        )
        for band, handle in pass2:
            if band not in band_order:
                continue
            i = band_order.index(band)
            self.log.info(
                "Pass 2 (noise corr), band %s for tract=%d patch=%d",
                band, tract, patch,
            )
            cell_coadd = handle.get()
            stitched = cell_coadd.to_legacy()
            noise_corr_array[i] = self.get_noise_corr(
                stitched,
                mask_array,
                self.config.mask_planes(band),
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

        if disc_array is None:
            disc_array = np.zeros_like(mask_array)
        return Struct(
            outputMask=output_msk,
            outputDiscontinuityMask=disc_array,
            outputNoiseCorr=noise_corr_array,
            outputPsf=psf_array,
            outputGaiaCatalog=gaia_table,
        )
