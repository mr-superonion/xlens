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

__all__ = [
    "MeasureCellCoaddsPipeConfig",
    "MeasureCellCoaddsPipe",
    "MeasureCellCoaddsPipeConnections",
]

import logging
from typing import Any

import anacal
import lsst.meas.extensions.shapeHSM  # noqa: F401  (registers HSM plugins)
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.meas.base import (
    SkyMapIdGeneratorConfig,
)
from lsst.pex.config import ConfigurableField, Field, FieldValidationError, ListField
from lsst.pipe.base import (
    NoWorkFound,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..utils.catalog import set_isPrimary
from ..utils.columns import (
    rename_flux_to_photoz_format,
    select_band_gauss_fluxes,
    select_detection_columns,
)
from ..utils.image import prepare_data_one_cell
from .anacal import AnacalTask
from .fpfs import FpfsMeasurementTask


class MeasureCellCoaddsPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={"inputName": "deep_coadd_cell"},
):
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    cellCoadd = cT.Input(
        doc="Input cell-based coadd image",
        name="{inputName}_predetection",
        storageClass="MultipleCellCoadd",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        deferLoad=True,
    )
    mask = cT.Input(
        doc="Combined mask from cell-based systematics.",
        name="{inputName}_systematics_mask",
        storageClass="Mask",
        dimensions=("skymap", "tract", "patch"),
        minimum=0,
        multiple=False,
    )
    anacalCatalog = cT.Output(
        doc="anacal catalog",
        name="{inputName}_anacal_catalog",
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
    do_measure_flux_gauss = Field[bool](
        doc=(
            "If True, also run AnaCal forced measurement during the "
            "force stage to extract per-band Gaussian fluxes and merge "
            "them into the output catalog."
        ),
        default=False,
    )
    bands = ListField[str](
        doc=(
            "Bands required to be present in the input cell coadd dict. "
            "The task raises if the set of bands actually delivered by "
            "the butler does not match this list (no partial-band runs)."
        ),
        default=["g", "r", "i", "z"],
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
            config=config,
            log=log,
            initInputs=initInputs,
            **kwargs,
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
    def _build_cell_block(cell):
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
            int(width // 2),  # xcen
            int(height // 2),  # ycen
            0,
            0,  # xmin, ymin
            width,
            height,  # xmax, ymax
            pad,
            pad,  # xmin_in, ymin_in
            width - pad,  # xmax_in
            height - pad,  # ymax_in
            pixel_scale,
            0,  # index
        )
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
            sy : sy + cell_bbox.getHeight(),
            sx : sx + cell_bbox.getWidth(),
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
        """Build the data dict for a single cell via prepare_data_one_cell."""
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        npix = self.config.anacal.npix
        blocks = self._build_cell_block(cell)
        noise_correction = self.config.anacal.do_noise_bias_correction
        data = prepare_data_one_cell(
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
            stitched_mask_array,
            mask_origin,
            cell.outer.bbox,
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
    ) -> dict:
        """Detect on the i-band cell coadd.

        Returns ``det_cats``, mapping each cell_id with non-empty
        detections to its anacal detection catalog.  The i-band
        photometric zeropoint is computed locally for use in detection
        but is not returned; ``_force`` re-derives the per-band
        zeropoint from each band's coadd.
        """
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        band = "i"
        if band not in coadd_handles_dict:
            raise KeyError(f"band '{band}' not in {coadd_handles_dict.keys()}")

        det_coadd = coadd_handles_dict[band].get()
        mag_zero = self._coadd_mag_zero(det_coadd)
        det_cells = dict(det_coadd.cells)

        det_cats: dict = {}
        for cell_id, det_cell in det_cells.items():
            cell_mask = self._cell_mask(
                stitched_mask_array,
                mask_origin,
                det_cell,
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
                    tract,
                    patch,
                    ix,
                    iy,
                    e,
                )
        del det_coadd, det_cells

        if not det_cats:
            # Edge-of-tract patches whose every cell fails noise estimation
            # end up with zero detections. Raise NoWorkFound so bps marks
            # this quantum as SKIPPED rather than FAILED; downstream
            # photoZ is auto-pruned by the missing-input rule and the
            # tract-level mergePatches still runs on the surviving
            # sibling patches via its srcList multiple-input connection.
            raise NoWorkFound(
                f"No objects detected in any cell "
                f"(tract={tract}, patch={patch}); skipping this patch."
            )
        return det_cats

    def _coadd_mag_zero(self, mca) -> float:
        """Photometric zeropoint of a ``MultipleCellCoadd``."""
        photoCalib = mca.stitch().asExposure().getPhotoCalib()
        return float(np.log10(photoCalib.getInstFluxAtZeroMagnitude()) / 0.4)

    def _force(
        self,
        *,
        detection_dict: dict,
        coadd_handles_dict: dict[str, Any],
        seed: int,
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
        cell_force_parts: dict[Any, list] = {cid: [] for cid in active_cell_ids}
        bands = list(coadd_handles_dict.keys())

        for band in bands:
            self.log.debug("Measuring band %s", band)
            band_coadd = coadd_handles_dict[band].get()
            mag_zero = self._coadd_mag_zero(band_coadd)
            for cell_id in active_cell_ids:
                cell = band_coadd.cells[cell_id]
                cell_mask = self._cell_mask(
                    stitched_mask_array,
                    mask_origin,
                    cell,
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
                        self.fpfs.run(**data),
                        band,
                    )
                    if self.config.do_measure_flux_gauss:
                        gauss_cat = select_band_gauss_fluxes(
                            self.anacal.run(**data),
                            band,
                        )
                        cat = np.asarray(
                            rfn.merge_arrays(
                                [cat, gauss_cat],
                                flatten=True,
                            )
                        )
                    cell_force_parts[cell_id].append(cat)
                except Exception as e:
                    ix, iy = int(cell_id.x), int(cell_id.y)
                    self.log.error(
                        "Measurement failed tract=%d patch=%d " "cell=(%d, %d) band=%s: %s",
                        tract,
                        patch,
                        ix,
                        iy,
                        band,
                        e,
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

        expected = set(self.config.bands)
        provided = set(coadd_handles_dict.keys())
        if provided != expected:
            raise RuntimeError(
                f"band mismatch for tract={tract} patch={patch}: "
                f"expected {sorted(expected)}, "
                f"got {sorted(provided)} "
                f"(missing={sorted(expected - provided)}, "
                f"extra={sorted(provided - expected)})"
            )

        first_handle = next(iter(coadd_handles_dict.values()))
        idGenerator = self.config.idGenerator.apply(first_handle.dataId)
        seed = idGenerator.catalog_id

        if mask is not None:
            stitched_mask_array = mask.getArray()
            mask_origin = (mask.getX0(), mask.getY0())
        else:
            stitched_mask_array = None
            mask_origin = None

        det_cats = self._detect(
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
            # Same edge-of-tract path as in `_detect` above, but here
            # forced measurement is what produced the empty result.
            raise NoWorkFound(
                f"No measurements produced in any cell "
                f"(tract={tract}, patch={patch}); skipping this patch."
            )

        output = np.concatenate(cell_results)
        # Stable per-object IDs derived from the patch-level seed. Used
        # downstream by ``photoZPipe`` and any object-level joiners.
        object_ids = np.int64(seed) * np.int64(1_000_000) + np.arange(len(output), dtype=np.int64)
        output = rfn.append_fields(
            output,
            "object_id",
            object_ids,
            usemask=False,
        )
        if skyMap is not None:
            # Use skymap's patchInfo (not MultipleCellCoadd.inner_bbox)
            # for is_primary deduplication. The skymap patch inner bbox
            # (3000x3000) defines the non-overlapping tiling, while
            # MultipleCellCoadd.inner_bbox equals the patch outer bbox.
            tractInfo = skyMap[tract]
            patchInfo = tractInfo[patch]
            pixel_scale = float(tractInfo.getWcs().getPixelScale().asArcseconds())
            set_isPrimary(output, skyMap, tractInfo, patchInfo, pixel_scale)
        return Struct(anacalCatalog=output)
