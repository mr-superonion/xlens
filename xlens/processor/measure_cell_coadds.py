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
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.meas.base import SkyMapIdGeneratorConfig
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

from ..processor.anacal import AnacalTask
from ..processor.fpfs import FpfsMeasurementTask
from ..utils.catalog import set_isPrimary
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
    output_catalog = cT.Output(
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

    def _detect_cell(self, data: dict) -> np.ndarray:
        """Run detection on a prepared cell data dict."""
        return self.anacal.run(**data)

    def _measure_one_band(
        self,
        *,
        cell,
        band: str,
        detection: NDArray,
        seed: int,
        mag_zero: float,
        skyMap,
        tract: int,
        patch: int,
        mask_array: NDArray | None = None,
    ) -> np.ndarray:
        """Run forced measurement on one band for a single cell."""
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        data = self._prepare_cell(
            cell,
            band=band,
            seed=seed,
            mag_zero=mag_zero,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            detection=detection,
            mask_array=mask_array,
        )

        colnames = [
            "flux_gauss0", "dflux_gauss0_dg1", "dflux_gauss0_dg2",
            "flux_gauss2", "dflux_gauss2_dg1", "dflux_gauss2_dg2",
            "flux_gauss4", "dflux_gauss4_dg1", "dflux_gauss4_dg2",
            "flux_gauss0_err", "flux_gauss2_err", "flux_gauss4_err",
        ]
        out = []
        out.append(
            rfn.repack_fields(
                self.anacal.run(**data)[colnames]
            )
        )
        out.append(self.fpfs.run(**data))
        res = rfn.merge_arrays(out, flatten=True)
        map_dict = {name: f"{band}_{name}" for name in colnames}
        return rfn.rename_fields(res, map_dict)

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

        det_band = "i"
        if det_band not in coadd_handles_dict:
            raise KeyError(
                f"band '{det_band}' not in {coadd_handles_dict.keys()}"
            )

        idGenerator = self.config.idGenerator.apply(
            coadd_handles_dict[det_band].dataId
        )
        seed = idGenerator.catalog_id

        # Prepare stitched mask for per-cell extraction
        if mask is not None:
            stitched_mask_array = mask.getArray()
            mask_origin = (mask.getX0(), mask.getY0())
        else:
            stitched_mask_array = None
            mask_origin = None

        # --- Phase 1: Detection (i-band only) ---
        det_coadd = coadd_handles_dict[det_band].get()
        photoCalib = det_coadd.stitch().asExposure().getPhotoCalib()
        mag_zero = float(
            np.log10(photoCalib.getInstFluxAtZeroMagnitude()) / 0.4
        )
        # Copy cell references before releasing the coadd
        det_cells = dict(det_coadd.cells)

        # Per-cell detection catalogs keyed by cell_id
        det_cats: dict = {}
        for cell_id, det_cell in det_cells.items():
            # Extract per-cell mask from stitched mask
            cell_mask = None
            if stitched_mask_array is not None:
                cell_mask = self._extract_cell_mask(
                    stitched_mask_array, mask_origin,
                    det_cell.outer.bbox,
                )
            try:
                det_data = self._prepare_cell(
                    det_cell,
                    band=det_band,
                    seed=seed,
                    mag_zero=mag_zero,
                    skyMap=skyMap,
                    tract=tract,
                    patch=patch,
                    mask_array=cell_mask,
                )
                det_cat = self._detect_cell(det_data)
                del det_data
                if len(det_cat) > 0:
                    det_cats[cell_id] = det_cat
            except Exception as e:
                ix, iy = int(cell_id.x), int(cell_id.y)
                self.log.error(
                    "Detection failed tract=%d patch=%d cell=(%d, %d): %s",
                    tract, patch, ix, iy, e,
                )
        del det_coadd, det_cells
        if not det_cats:
            raise RuntimeError("No objects found in any cell")

        # Cell IDs with detections
        active_cell_ids = list(det_cats.keys())

        # --- Phase 2: Forced measurement (one band at a time) ---
        cell_force_parts: dict[Any, list] = {
            cid: [] for cid in active_cell_ids
        }

        bands = list(coadd_handles_dict.keys())
        for band in bands:
            self.log.debug("Measuring band %s", band)
            band_coadd = coadd_handles_dict[band].get()

            for cell_id in active_cell_ids:
                cell = band_coadd.cells[cell_id]
                # Extract per-cell mask
                cell_mask = None
                if stitched_mask_array is not None:
                    cell_mask = self._extract_cell_mask(
                        stitched_mask_array, mask_origin,
                        cell.outer.bbox,
                    )
                try:
                    res = self._measure_one_band(
                        cell=cell,
                        band=band,
                        detection=det_cats[cell_id],
                        seed=seed,
                        mag_zero=mag_zero,
                        skyMap=skyMap,
                        tract=tract,
                        patch=patch,
                        mask_array=cell_mask,
                    )
                    cell_force_parts[cell_id].append(res)
                except Exception as e:
                    ix, iy = int(cell_id.x), int(cell_id.y)
                    self.log.error(
                        "Measurement failed tract=%d patch=%d "
                        "cell=(%d, %d) band=%s: %s",
                        tract, patch, ix, iy, band, e,
                    )

            del band_coadd

        # --- Phase 3: Merge results ---
        cell_results = []
        for cell_id in active_cell_ids:
            parts = cell_force_parts[cell_id]
            if len(parts) != len(bands):
                continue
            force_cat = rfn.merge_arrays(parts, flatten=True)
            final = rfn.merge_arrays(
                [det_cats[cell_id], force_cat], flatten=True,
            )
            cell_results.append(final)

        if not cell_results:
            raise RuntimeError("No objects found in any cell")

        output = np.concatenate(cell_results)
        if skyMap is not None:
            tractInfo = skyMap[tract]
            patchInfo = tractInfo[patch]
            pixel_scale = float(
                tractInfo.getWcs().getPixelScale().asArcseconds()
            )
            set_isPrimary(output, skyMap, tractInfo, patchInfo, pixel_scale)
        return Struct(output_catalog=output)
