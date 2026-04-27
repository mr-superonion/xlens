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
    "MeasureCellCoaddsPipeConfig",
    "MeasureCellCoaddsPipe",
    "MeasureCellCoaddsPipeConnections",
]

import logging
from typing import Any

import anacal
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.meas.extensions.shapeHSM  # noqa: F401  (registers HSM plugins)
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.afw.math import FixedKernel
from lsst.meas.algorithms import KernelPsf
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
    SkyMapIdGeneratorConfig,
)
from lsst.pex.config import ConfigurableField, Field, FieldValidationError
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from .anacal import AnacalTask
from .fpfs import FpfsMeasurementTask
from ..utils.catalog import set_isPrimary
from ..utils.columns import (
    rename_flux_to_photoz_format,
    select_detection_columns,
)
from ..utils.image import prepare_data_cell


class MeasureCellCoaddsPipeConnections(
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
    mask = cT.Input(
        doc="Combined mask from cell-based systematics.",
        name="deep_coadd_cell_systematics_mask",
        storageClass="Mask",
        dimensions=("skymap", "tract", "patch"),
        minimum=0,
        multiple=False,
    )
    anacalCatalog = cT.Output(
        doc="anacal catalog",
        name="{coaddName}_cell_coadd_anacal_catalog",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class MeasureCellCoaddsPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=MeasureCellCoaddsPipeConnections,
):
    anacal = ConfigurableField(
        target=AnacalTask,
        doc="AnaCal Task for detection stage (i-band)",
    )
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )
    psfCache = Field[int](
        doc="Size of PSF cache",
        default=100,
    )
    do_measure_psf = Field[bool](
        doc=(
            "If True, run DM SDSS adaptive moments and HSM (incl. higher-"
            "order) moments on each cell's PSF stamp and broadcast the "
            "results as per-band columns on every object in that cell."
        ),
        default=False,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()
        if self.fpfs.sigma_shapelets1 < 0.0:
            raise FieldValidationError(
                self.fpfs.__class__.sigma_shapelets1,
                self,
                "sigma_shapelets1 in a wrong range",
            )

    def setDefaults(self):
        super().setDefaults()
        self.anacal.force_size = True
        self.anacal.force_center = True
        self.anacal.bound = 5
        self.fpfs.do_compute_detect_weight = False


class MeasureCellCoaddsPipe(PipelineTask):
    """Detect and measure sources on cell-based coadds.

    Each SingleCellCoadd has a 250x250 outer region and a 150x150 inner
    region with 50px padding on all sides. Detection and measurement are
    performed on the full outer region so that objects near inner-region
    boundaries have complete pixel context. The anacal block inner region
    (pad=50) keeps only objects whose centers fall within the 150x150
    inner region, preventing double-counting across neighboring cells.

    The noise realization stored in each cell coadd is passed directly
    to anacal for noise bias correction. The noise image is rotated by
    90 degrees inside ``prepare_data`` to remove anisotropy.
    """

    _DefaultName = "MeasureCellCoaddsPipe"
    ConfigClass = MeasureCellCoaddsPipeConfig

    def __init__(
        self,
        *,
        config: MeasureCellCoaddsPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log,
            initInputs=initInputs, **kwargs,
        )
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)

        self.makeSubtask("anacal")
        self.makeSubtask("fpfs")

        self._psf_meas_task = None
        self._psf_meas_schema = None
        self._psf_meas_columns: list[str] = []
        if self.config.do_measure_psf:
            self._psf_meas_schema, self._psf_meas_task = (
                self._build_psf_meas_task()
            )
            self._psf_meas_columns = self._collect_psf_meas_columns(
                self._psf_meas_schema,
            )

    @staticmethod
    def _build_psf_meas_task() -> tuple[afwTable.Schema,
                                        SingleFrameMeasurementTask]:
        """Construct a one-shot SingleFrameMeasurementTask that runs the
        SDSS shape and HSM (adaptive + higher-order) PSF plugins.
        """
        schema = afwTable.SourceTable.makeMinimalSchema()
        cfg = SingleFrameMeasurementConfig()
        cfg.plugins.names = [
            "base_SdssCentroid",
            "base_SdssShape",
            "ext_shapeHSM_HsmPsfMoments",
            "ext_shapeHSM_HigherOrderMomentsPSF",
        ]
        cfg.slots.shape = "base_SdssShape"
        cfg.slots.centroid = "base_SdssCentroid"
        cfg.slots.psfShape = "base_SdssShape_psf"
        cfg.slots.apFlux = None
        cfg.slots.modelFlux = None
        cfg.slots.psfFlux = None
        cfg.slots.gaussianFlux = None
        cfg.slots.calibFlux = None
        cfg.doReplaceWithNoise = False
        task = SingleFrameMeasurementTask(schema=schema, config=cfg)
        return schema, task

    @staticmethod
    def _collect_psf_meas_columns(schema: afwTable.Schema) -> list[str]:
        """Pick the per-cell scalar PSF columns we want to broadcast."""
        keep: list[str] = []
        for name in schema.getNames():
            n = name
            if n.startswith("base_SdssShape_") and (
                n.endswith("_xx") or n.endswith("_yy") or n.endswith("_xy")
            ):
                # Both source ('base_SdssShape_xx') and PSF model
                # ('base_SdssShape_psf_xx') versions; keep both.
                keep.append(n)
            elif n.startswith("ext_shapeHSM_HsmPsfMoments_") and (
                n.endswith("_xx") or n.endswith("_yy") or n.endswith("_xy")
            ):
                keep.append(n)
            elif n.startswith("ext_shapeHSM_HigherOrderMomentsPSF_"):
                # Higher-order moment values are named with two-digit
                # order codes (e.g. '_22', '_31', '_40').  Skip flag
                # columns.
                tail = n.rsplit("_", 1)[-1]
                if (
                    len(tail) == 2 and tail.isdigit()
                ):
                    keep.append(n)
        return sorted(keep)

    def _measure_psf_for_cell(self, cell) -> dict[str, float]:
        """Measure SDSS + HSM (adaptive + higher-order) moments on one
        cell's PSF stamp.  Returns a {column_name: scalar} dict of the
        values listed in ``self._psf_meas_columns``.

        Important: the PSF model is centered on a pixel center (LSST
        convention), so the synthetic source's centroid is set to that
        integer pixel.
        """
        assert self._psf_meas_task is not None

        psf_arr = np.asarray(cell.psf_image.array, dtype=np.float32)
        ny, nx = psf_arr.shape
        cx_pix = nx // 2  # PSF stamp center is at the integer pixel center
        cy_pix = ny // 2

        # Build a small ExposureF whose image content IS the PSF stamp;
        # also attach the same stamp as the PSF model so HSM PSF plugins
        # see a sensible PSF at the centroid.
        exp = afwImage.ExposureF(nx, ny)
        exp.image.array[:] = psf_arr
        # Tiny non-zero variance keeps SdssShape happy; PSFs are noiseless.
        peak_sq = float(np.max(psf_arr) ** 2) if psf_arr.size else 1.0
        exp.variance.array[:] = max(peak_sq * 1e-10, 1e-30)
        exp.mask.array[:] = 0

        psf_im = afwImage.ImageD(nx, ny)
        psf_im.array[:] = psf_arr.astype(np.float64)
        exp.setPsf(KernelPsf(FixedKernel(psf_im)))

        cat = afwTable.SourceCatalog(self._psf_meas_schema)
        src = cat.addNew()
        fp = afwDetection.Footprint(afwGeom.SpanSet(exp.getBBox()))
        peak_val = float(psf_arr[cy_pix, cx_pix])
        fp.addPeak(cx_pix, cy_pix, peak_val)
        src.setFootprint(fp)

        try:
            self._psf_meas_task.run(cat, exp)
        except Exception as e:
            self.log.warning("PSF measurement failed on cell: %s", e)
            return {n: float("nan") for n in self._psf_meas_columns}

        out: dict[str, float] = {}
        for n in self._psf_meas_columns:
            try:
                out[n] = float(cat[0][n])
            except Exception:
                out[n] = float("nan")
        return out

    def _append_psf_columns(
        self,
        per_band_cat: NDArray,
        band: str,
        psf_values: dict[str, float],
    ) -> NDArray:
        """Append per-cell PSF moments as constant per-row columns named
        ``{band}_psf_<plugin tail>``.  All columns are float64.
        """
        if not psf_values:
            return per_band_cat
        n = len(per_band_cat)
        new_names: list[str] = []
        new_arrays: list[NDArray] = []
        for raw_name, val in psf_values.items():
            short = raw_name
            for prefix in (
                "base_SdssShape_psf_",
                "base_SdssShape_",
                "ext_shapeHSM_HsmPsfMoments_",
                "ext_shapeHSM_HigherOrderMomentsPSF_",
            ):
                if short.startswith(prefix):
                    tag = {
                        "base_SdssShape_psf_": "sdss_psf",
                        "base_SdssShape_": "sdss",
                        "ext_shapeHSM_HsmPsfMoments_": "hsm",
                        "ext_shapeHSM_HigherOrderMomentsPSF_": "hsm_ho",
                    }[prefix]
                    short = f"{tag}_{short[len(prefix):]}"
                    break
            colname = f"{band}_psf_{short}"
            new_names.append(colname)
            new_arrays.append(np.full(n, float(val), dtype=np.float64))
        return np.asarray(
            rfn.append_fields(
                per_band_cat, new_names, new_arrays, usemask=False,
            )
        )

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)

        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        patch = int(butlerQC.quantum.dataId["patch"])

        coadd_handles = inputs["cellCoadd"]
        coadd_handles_dict = {h.dataId["band"]: h for h in coadd_handles}

        outputs = self.run(
            coadd_handles_dict=coadd_handles_dict,
            skyMap=inputs["skyMap"],
            tract=tract,
            patch=patch,
            mask=inputs.get("mask", None),
        )
        butlerQC.put(outputs, outputRefs)

    @staticmethod
    def _build_cell_block(cell, psf_array):
        """Build a single anacal block covering the full cell outer region.

        The block inner region is set with pad=50, matching the cell's
        inner/outer structure (250x250 outer, 150x150 inner, 50px on
        each side). Anacal only keeps detections whose centers fall
        within the block inner region [50, 200) x [50, 200).
        """
        bbox = cell.outer.bbox
        width = bbox.getWidth()
        height = bbox.getHeight()
        pixel_scale = float(cell.wcs.getPixelScale().asArcseconds())
        pad = 50
        bb = anacal.geometry.block(
            int(width // 2),   # xcen
            int(height // 2),  # ycen
            0, 0,              # xmin, ymin
            width, height,     # xmax, ymax
            pad, pad,          # xmin_in, ymin_in
            width - pad,       # xmax_in
            height - pad,      # ymax_in
            pixel_scale,
            0,                 # index
        )
        bb.psf_array = psf_array.copy()
        bb.xmsk = [True] * width
        bb.ymsk = [True] * height
        return [bb]

    @staticmethod
    def _extract_cell_mask(mask_array, mask_origin, cell_bbox):
        """Extract the mask for a cell's outer bbox from the stitched mask.

        Parameters
        ----------
        mask_array : NDArray
            Full stitched mask array.
        mask_origin : tuple of int
            (x0, y0) origin of the stitched mask in pixel coordinates.
        cell_bbox : lsst.geom.Box2I
            Outer bounding box of the cell.

        Returns
        -------
        NDArray
            Mask slice for the cell's outer region.
        """
        x0, y0 = mask_origin
        sx = cell_bbox.getMinX() - x0
        sy = cell_bbox.getMinY() - y0
        return mask_array[
            sy: sy + cell_bbox.getHeight(),
            sx: sx + cell_bbox.getWidth(),
        ].copy()

    def _prepare_cell(
        self,
        cell,
        *,
        band: str,
        seed: int,
        mag_zero: float,
        skyMap,
        tract: int,
        patch: int,
        detection: NDArray | None = None,
        mask_array: NDArray | None = None,
    ) -> dict:
        """Build the data dict for a single cell via prepare_data_cell."""
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        npix = self.config.anacal.npix
        blocks = self._build_cell_block(cell, np.zeros((npix, npix)))
        noise_correction = self.config.anacal.do_noise_bias_correction
        data = prepare_data_cell(
            cell=cell,
            band=band,
            seed=seed,
            mag_zero=mag_zero,
            npix=npix,
            do_noise_bias_correction=noise_correction,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            detection=detection,
            blocks=blocks,
            mask_array=mask_array,
        )
        # Update block PSF with the actual computed PSF
        data["blocks"][0].psf_array = data["psf_array"].copy()
        return data

    def _cell_mask(
        self,
        stitched_mask_array: NDArray | None,
        mask_origin: tuple[int, int] | None,
        cell,
    ) -> NDArray | None:
        if stitched_mask_array is None:
            return None
        return self._extract_cell_mask(
            stitched_mask_array, mask_origin, cell.outer.bbox,
        )

    def _detect(
        self,
        *,
        coadd_handles_dict: dict[str, Any],
        seed: int,
        skyMap,
        tract: int,
        patch: int,
        stitched_mask_array: NDArray | None = None,
        mask_origin: tuple[int, int] | None = None,
    ) -> tuple[dict, float]:
        """Detect on the i-band cell coadd.

        Returns ``(det_cats, mag_zero)`` where ``det_cats`` maps each
        cell_id with non-empty detections to its anacal detection
        catalog, and ``mag_zero`` is the i-band photometric zeropoint.
        """
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        band = "i"
        if band not in coadd_handles_dict:
            raise KeyError(
                f"band '{band}' not in {coadd_handles_dict.keys()}"
            )

        det_coadd = coadd_handles_dict[band].get()
        photoCalib = det_coadd.stitch().asExposure().getPhotoCalib()
        mag_zero = float(
            np.log10(photoCalib.getInstFluxAtZeroMagnitude()) / 0.4
        )
        det_cells = dict(det_coadd.cells)

        det_cats: dict = {}
        for cell_id, det_cell in det_cells.items():
            cell_mask = self._cell_mask(
                stitched_mask_array, mask_origin, det_cell,
            )
            try:
                data = self._prepare_cell(
                    det_cell,
                    band=band,
                    seed=seed,
                    mag_zero=mag_zero,
                    skyMap=skyMap,
                    tract=tract,
                    patch=patch,
                    mask_array=cell_mask,
                )
                cat = self.anacal.run(**data)
                del data
                if len(cat) > 0:
                    det_cats[cell_id] = cat
            except Exception as e:
                ix, iy = int(cell_id.x), int(cell_id.y)
                self.log.error(
                    "Detection failed tract=%d patch=%d cell=(%d, %d): %s",
                    tract, patch, ix, iy, e,
                )
        del det_coadd, det_cells

        if not det_cats:
            raise RuntimeError("No objects found in any cell")
        return det_cats, mag_zero

    def _force(
        self,
        *,
        detection_dict: dict,
        coadd_handles_dict: dict[str, Any],
        seed: int,
        mag_zero: float,
        skyMap,
        tract: int,
        patch: int,
        stitched_mask_array: NDArray | None = None,
        mask_origin: tuple[int, int] | None = None,
    ) -> dict:
        """Force-measure the detected cells across all bands.

        Returns a dict mapping cell_id to the band-merged forced
        measurement catalog for cells where every band succeeded.
        """
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        active_cell_ids = list(detection_dict.keys())
        cell_force_parts: dict[Any, list] = {
            cid: [] for cid in active_cell_ids
        }
        bands = list(coadd_handles_dict.keys())

        for band in bands:
            self.log.debug("Measuring band %s", band)
            band_coadd = coadd_handles_dict[band].get()
            for cell_id in active_cell_ids:
                cell = band_coadd.cells[cell_id]
                cell_mask = self._cell_mask(
                    stitched_mask_array, mask_origin, cell,
                )
                try:
                    data = self._prepare_cell(
                        cell,
                        band=band,
                        seed=seed,
                        mag_zero=mag_zero,
                        skyMap=skyMap,
                        tract=tract,
                        patch=patch,
                        detection=detection_dict[cell_id],
                        mask_array=cell_mask,
                    )
                    cat = rename_flux_to_photoz_format(
                        self.fpfs.run(**data), band,
                    )
                    if self.config.do_measure_psf:
                        psf_vals = self._measure_psf_for_cell(cell)
                        cat = self._append_psf_columns(
                            cat, band, psf_vals,
                        )
                    cell_force_parts[cell_id].append(cat)
                except Exception as e:
                    ix, iy = int(cell_id.x), int(cell_id.y)
                    self.log.error(
                        "Measurement failed tract=%d patch=%d "
                        "cell=(%d, %d) band=%s: %s",
                        tract, patch, ix, iy, band, e,
                    )
            del band_coadd

        nbands = len(bands)
        force_cats: dict = {}
        for cell_id, parts in cell_force_parts.items():
            if len(parts) != nbands:
                continue
            force_cats[cell_id] = rfn.merge_arrays(parts, flatten=True)
        return force_cats

    def run(
        self,
        *,
        coadd_handles_dict: dict[str, Any],
        skyMap,
        tract: int,
        patch: int,
        mask=None,
        **kwargs,
    ):
        """Run detection and forced measurement on cell-based coadds.

        Detection is performed using i-band only. Forced measurement
        processes one band at a time to minimize memory usage: each
        band's MultipleCellCoadd is loaded, measured across all cells,
        then released before loading the next band.

        Parameters
        ----------
        coadd_handles_dict : dict
            Mapping of band name to deferred MultipleCellCoadd handle.
        skyMap : BaseSkyMap
            Sky map used for processing.
        tract, patch : int
            Tract and patch identifiers.
        mask : MaskX or None
            Combined stitched mask from BuildCellSystematicsTask.
            If provided, per-cell masks are extracted by slicing.
        """
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)

        # Seed is band-independent under the default
        # SkyMapIdGeneratorConfig (n_bands=0), so any handle in the dict
        # gives the same catalog_id.  Use the first one to avoid hard-
        # coding "i".
        first_handle = next(iter(coadd_handles_dict.values()))
        idGenerator = self.config.idGenerator.apply(first_handle.dataId)
        seed = idGenerator.catalog_id

        if mask is not None:
            stitched_mask_array = mask.getArray()
            mask_origin = (mask.getX0(), mask.getY0())
        else:
            stitched_mask_array = None
            mask_origin = None

        det_cats, mag_zero = self._detect(
            coadd_handles_dict=coadd_handles_dict,
            seed=seed,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            stitched_mask_array=stitched_mask_array,
            mask_origin=mask_origin,
        )

        force_cats = self._force(
            detection_dict=det_cats,
            coadd_handles_dict=coadd_handles_dict,
            seed=seed,
            mag_zero=mag_zero,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            stitched_mask_array=stitched_mask_array,
            mask_origin=mask_origin,
        )

        cell_results = []
        for cell_id, force_cat in force_cats.items():
            final = rfn.merge_arrays(
                [select_detection_columns(det_cats[cell_id]), force_cat],
                flatten=True,
            )
            cell_results.append(final)

        if not cell_results:
            raise RuntimeError("No objects found in any cell")

        output = np.concatenate(cell_results)
        # Stable per-object IDs derived from the patch-level seed. Used
        # downstream by ``photoZPipe`` and any object-level joiners.
        object_ids = (
            np.int64(seed) * np.int64(1_000_000)
            + np.arange(len(output), dtype=np.int64)
        )
        output = rfn.append_fields(
            output, "object_id", object_ids, usemask=False,
        )
        if skyMap is not None:
            # Use skymap's patchInfo (not MultipleCellCoadd.inner_bbox)
            # for is_primary deduplication. The skymap patch inner bbox
            # (3000x3000) defines the non-overlapping tiling, while
            # MultipleCellCoadd.inner_bbox equals the patch outer bbox.
            tractInfo = skyMap[tract]
            patchInfo = tractInfo[patch]
            pixel_scale = float(
                tractInfo.getWcs().getPixelScale().asArcseconds()
            )
            set_isPrimary(output, skyMap, tractInfo, patchInfo, pixel_scale)
        return Struct(anacalCatalog=output)
