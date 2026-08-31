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
import lsst.pipe.base.connectionTypes as cT
import typing

import numpy as np
from lsst.images._geom import NoOverlapError
from lsst.pex.config import Field, FieldValidationError, ListField
from lsst.pipe.base import (
    NoWorkFound,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..utils.columns import select_detection_columns
import lsst.geom as lsst_geom
from lsst.afw.image import MaskX

from ..utils.image import (
    rle_table_to_mask,
    make_psf_stamp_exposure,
    prepare_data_one_cell,
    prepare_data_one_cell_multiband,
)
from .measure_base import AnacalMeasureTaskBase, MeasureBandsConfigBase


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
        # Native cell coadd. Declaring MultipleCellCoadd would make
        # butler run DM's CellCoadd -> MultipleCellCoadd converter
        # (lsst.images.cells._coadd.to_legacy_cell_coadd), which raises
        # "requires its bounding box to lie on the cell grid" for
        # patches whose bbox is not a whole number of cells -- 322 of
        # 81960 patches lost in the first DP2 measurement run.
        storageClass="CellCoadd",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        deferLoad=True,
    )
    mask = cT.Input(
        doc=(
            "Combined mask from cell-based systematics, run-length "
            "encoded with a value column (decode with "
            "xlens.utils.image.rle_table_to_mask; bit 0 = masked/cut, "
            "bit 1 = discontinuity, stamped per source as "
            "n_mask_discontinuity, never cut; "
            "origin is the patch outer bbox)."
        ),
        name="{inputName}_systematics_mask_rle",
        storageClass="ArrowAstropy",
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
    MeasureBandsConfigBase,
    PipelineTaskConfig,
    pipelineConnections=MeasureCellCoaddsPipeConnections,
):
    bands = ListField[str](
        doc=(
            "PHYSICAL bands (butler ``band`` dimension) required in the input "
            "cell coadd dict. The task raises if the set of bands actually "
            "delivered by the butler does not match this list."
        ),
        default=["g", "r", "i", "z"],
    )

    cell_border = Field[int](
        doc=(
            "Border in pixels grown around each cell to build its outer "
            "stamp, for skymaps whose cells have none. DP1's cell coadds "
            "stored a 50 px border (250x250 outer around a 150x150 "
            "inner); DP2's lsst_cells_v2 cells tile edge to edge, so the "
            "outer region has to be cut from the stitched patch here -- "
            "otherwise the outer IS the inner and anacal's acceptance "
            "region collapses to the middle ninth of every cell. Same "
            "role as `border` in lsst.drp.tasks.metadetection_shear, "
            "which defaults to 50 for the same reason. 0 disables the "
            "dilation and uses each cell as stored."
        ),
        default=50,
    )

    def validate(self):
        super().validate()
        missing = [b for b in self.detection_bands if b not in self.bands]
        if missing:
            raise FieldValidationError(
                self.__class__.detection_bands,
                self,
                f"detection_bands must be a subset of bands; missing {missing}.",
            )

    def setDefaults(self):
        super().setDefaults()
        self.anacal.bound = 5


class MeasureCellCoaddsPipe(AnacalMeasureTaskBase):
    """Detect and measure sources on cell-based coadds.

    Each SingleCellCoadd has a 250x250 outer region and a 150x150 inner
    region with 50px padding on all sides. Detection and measurement are
    performed on the full outer region so that objects near inner-region
    boundaries have complete pixel context. The anacal cell inner region
    (pad=50) keeps only objects whose centers fall within the 150x150
    inner region, preventing double-counting across neighboring cells.

    The noise realization stored in each cell coadd is passed directly
    to anacal for noise bias correction. The noise image is rotated by
    90 degrees inside ``prepare_data`` to remove anisotropy.

    The per-cell loops run through a thread pool of ``num_workers``
    threads (default 1, a plain serial loop), sharing the dispatch point
    with the per-cell loops of :class:`MeasureCoaddsPipe`.
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
        self._make_measure_subtasks()

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)

        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        patch = int(butlerQC.quantum.dataId["patch"])

        coadd_handles = inputs["cellCoadd"]
        coadd_handles_dict = {h.dataId["band"]: h for h in coadd_handles}

        # The combined bitmask arrives run-length encoded; decode to
        # pixels once (preserving the bit values 0..3), restoring the
        # stitched origin the cell chain relies on.
        if inputs.get("mask", None) is not None:
            # Shape and origin come from the patch's outer bbox, which
            # is what the mask covers by construction -- never from the
            # table, which carries no geometry (butler.get strips
            # table.meta, so anything stored there would be lost).
            bbox = inputs["skyMap"][tract][patch].getOuterBBox()
            arr = rle_table_to_mask(
                inputs["mask"], (bbox.getHeight(), bbox.getWidth())
            )
            msk = MaskX(width=arr.shape[1], height=arr.shape[0])
            msk.getArray()[:, :] = arr.astype(
                msk.getArray().dtype, copy=False
            )
            msk.setXY0(lsst_geom.Point2I(bbox.getMinX(), bbox.getMinY()))
            inputs["mask"] = msk

        outputs = self.run(
            coadd_handles_dict=coadd_handles_dict,
            skyMap=inputs["skyMap"],
            tract=tract,
            patch=patch,
            mask=inputs.get("mask", None),
        )
        butlerQC.put(outputs, outputRefs)

    @staticmethod
    def _get_coadd(handle):
        """Load one band's cell coadd, skipping patches butler cannot convert.

        This task declares MultipleCellCoadd, so DP2's CellCoadd goes
        through the storage-class converter, which raises for patches
        whose bbox is not a whole number of cells:

            ValueError: MultipleCellCoadd requires its bounding box to
                        lie on the cell grid.

        That is a property of the patch, not a fault, and it took out 36
        of ~8700 quanta (~0.4%) as hard failures in the first DP2
        systematics run. NoWorkFound makes the executor skip the quantum
        instead. buildCellSystematics avoids it entirely by reading
        CellCoadd natively; doing the same here needs the per-cell image
        access reworked, so this is the interim guard.
        """
        try:
            return handle.get()
        except ValueError as err:
            if "cell grid" in str(err):
                raise NoWorkFound(
                    "cell coadd cannot be converted to MultipleCellCoadd "
                    "(bbox not on the cell grid); skipping: %s" % err
                ) from err
            raise

    class _Plane:
        """Minimal ``.array`` holder, matching what the cell reader wants."""

        __slots__ = ("array",)

        def __init__(self, array):
            self.array = array

    class _Region:
        """``.bbox`` plus optional planes -- stands in for inner/outer."""

        def __init__(self, bbox):
            self.bbox = bbox

    class _CellId(typing.NamedTuple):
        """Cell index exposing ``.x``/``.y``, as the legacy keys do."""

        x: int
        y: int

    class _NativeCellView:
        """``SingleCellCoadd``-like view cut from a native ``CellCoadd``.

        The native coadd stores every plane as one contiguous full-patch
        array plus a cell grid, so a cell -- with or without a border --
        is a SLICE, not a stitch. ``inner`` is the cell's own footprint;
        ``outer`` is that box grown by ``border``.
        """

        __slots__ = ("inner", "outer", "wcs", "psf_image")

        def __init__(self, exposure, noise_planes, psf_image, wcs,
                     inner_bbox, outer_bbox):
            sub = exposure[outer_bbox]
            outer = MeasureCellCoaddsPipe._Region(outer_bbox)
            outer.image = sub.image
            outer.variance = sub.variance
            outer.mask = sub.mask
            x0 = outer_bbox.getMinX() - exposure.getBBox().getMinX()
            y0 = outer_bbox.getMinY() - exposure.getBBox().getMinY()
            h, w = outer_bbox.getHeight(), outer_bbox.getWidth()
            outer.noise_realizations = [
                MeasureCellCoaddsPipe._Plane(n[y0:y0 + h, x0:x0 + w])
                for n in noise_planes
            ]
            self.inner = MeasureCellCoaddsPipe._Region(inner_bbox)
            self.outer = outer
            self.wcs = wcs
            self.psf_image = psf_image

    def _native_cells(self, coadd, bounds=None) -> dict:
        """``{cell_id: view}`` for a native ``CellCoadd``.

        Cells are enumerated from the PSF's ``CellGridBounds``, NOT from
        the image grid. The image grid is pure geometry -- it yields
        every cell position the patch could have -- while the PSF is a
        separate structure that need not cover all of them: it carries a
        ``missing`` set, and DM's own docstring warns its ``grid`` "is
        usually (but is not guaranteed to be) the grid for a full patch,
        even when the PSF only covers a subimage".

        Walking the image grid and indexing ``coadd.psf[cid]`` therefore
        raised ``BoundsError`` on the first absent cell, which killed
        the whole patch: 1470 of 4259 deep-field quanta died that way,
        e.g. band r of tract 4636 patch 0 is missing 128 of 484 cells.

        ``bounds`` lets the caller pass an already-intersected
        ``CellGridBounds`` so every band detects on the SAME cells --
        a cell present in i but absent in r is unusable for a
        multi-band detection, and silently using a different cell set
        per band would make the bands' catalogs incommensurate.
        """
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)

        border = int(self.config.cell_border)
        exposure = coadd.to_legacy()
        ebox = exposure.getBBox()
        noise = [np.asarray(n.array) for n in
                 (getattr(coadd, "noise_realizations", None) or [])]
        wcs = exposure.getWcs()
        grid = coadd.grid
        size = grid.grid_size
        if bounds is None:
            bounds = coadd.psf.bounds
        out = {}
        n_grid = size.i * size.j
        for cid in bounds.cell_indices():
                b = grid.bbox_of(cid)
                i, j = cid.i, cid.j
                inner = lsst_geom.Box2I(
                    lsst_geom.Point2I(b.x.start, b.y.start),
                    lsst_geom.Point2I(b.x.stop - 1, b.y.stop - 1))
                outer = lsst_geom.Box2I(inner)
                if border > 0:
                    outer.grow(border)
                    if not ebox.contains(outer):
                        # patch-border cell: the neighbouring patch
                        # covers this sky, as in _dilated_cells
                        continue
                # the native PSF is an lsst.images.Image; downstream
                # (psf HSM moments) wants the afw one, which has getBBox
                psf = coadd.psf[cid]
                if hasattr(psf, "to_legacy"):
                    psf = psf.to_legacy()
                out[self._CellId(x=j, y=i)] = self._NativeCellView(
                    exposure, noise, psf, wcs, inner, outer)
        self.log.info(
            "native cell coadd: %d of %d grid cells usable "
            "(psf covers %d, border=%d)",
            len(out), n_grid, sum(1 for _ in bounds.cell_indices()), border)
        return out

    def _dilated_cells(self, coadd, bounds=None) -> dict:
        """``{cell_id: view}`` with outer regions grown by cell_border.

        Thin wrapper over :meth:`_native_cells`. The MultipleCellCoadd
        branch that used to live here was unreachable: the cellCoadd
        connection declares storageClass="CellCoadd", so butler always
        delivers a CellCoadd -- converting a stored MultipleCellCoadd
        via CellCoadd.from_legacy_cell_coadd if it has to -- and the
        `hasattr(mca, "cells")` test was therefore always False.
        """
        return self._native_cells(coadd, bounds=bounds)

    @staticmethod
    def _build_anacal_cell(cell):
        """Build a single anacal cell covering the full cell outer region.

        The cell inner region is set with pad=50, matching the cell's
        inner/outer structure (250x250 outer, 150x150 inner, 50px on
        each side). Anacal only keeps detections whose centers fall
        within the cell inner region [50, 200) x [50, 200).
        """
        bbox = cell.outer.bbox
        width = bbox.getWidth()
        height = bbox.getHeight()
        pixel_scale = float(cell.wcs.getPixelScale().asArcseconds())
        # Derived, NOT hardcoded: it must match the outer/inner geometry
        # actually in hand. A fixed 50 silently shrank the acceptance
        # region to [50,100) of a 150 px cell on DP2, throwing away ~89%
        # of the detections.
        pad = (width - cell.inner.bbox.getWidth()) // 2
        bb = anacal.geometry.cell(
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
        cells = self._build_anacal_cell(cell)
        noise_correction = self.config.anacal.do_noise_bias_correction
        data = prepare_data_one_cell(
            cell=cell,
            band=band,
            survey=self.config.survey,
            seed=seed,
            mag_zero=mag_zero,
            npix=npix,
            do_noise_bias_correction=noise_correction,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            detection=detection,
            cells=cells,
            mask_array=mask_array,
        )
        # Update cell PSF with the actual computed PSF
        data["cells"][0].psf_array = data["psf_array"].copy()
        return data

    def _prepare_cell_multiband(
        self,
        lsst_cells: dict,
        *,
        bands: list[str],
        seed: int,
        mag_zeros: dict,
        skyMap,
        tract: int,
        patch: int,
        mask_array: NDArray | None = None,
    ) -> dict:
        """Build the data dict for one cell across several bands.

        ``lsst_cells`` maps band to that band's ``SingleCellCoadd`` for
        the same cell id.  They share a pixel grid, so one AnaCal cell
        covers them all.
        """
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        npix = self.config.anacal.npix
        cells = self._build_anacal_cell(lsst_cells[bands[0]])
        noise_correction = self.config.anacal.do_noise_bias_correction
        data = prepare_data_one_cell_multiband(
            lsst_cells=lsst_cells,
            bands=bands,
            survey=self.config.survey,
            seed=seed,
            mag_zeros=mag_zeros,
            npix=npix,
            do_noise_bias_correction=noise_correction,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            cells=cells,
            mask_array=mask_array,
        )
        # One PSF stamp per band, in the order the images were stacked.
        data["cells"][0].psf_array = data["psf_array"].copy()
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
        """Detect on the cell coadd of ``config.detection_bands``.

        Returns ``det_cats``, mapping each cell_id with non-empty
        detections to its anacal detection catalog.  The detection bands'
        photometric zeropoints are computed locally for use in detection
        but are not returned; ``_force`` re-derives the per-band
        zeropoint from each band's coadd.
        """
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        bands = list(self.config.detection_bands)
        missing = [b for b in bands if b not in coadd_handles_dict]
        if missing:
            raise KeyError(
                f"detection band(s) {missing} not in "
                f"{list(coadd_handles_dict.keys())}"
            )
        if len(bands) > 1:
            self.log.info("Detecting on the coadd of bands %s", bands)

        det_coadds = {b: self._get_coadd(coadd_handles_dict[b]) for b in bands}
        mag_zeros = {b: self._coadd_mag_zero(c) for b, c in det_coadds.items()}
        # Detection combines the bands, so a cell is usable only where
        # EVERY detection band has a PSF. Intersecting the bands'
        # CellGridBounds up front gives one cell set for all of them;
        # per-band sets would make the stacked moments incommensurate.
        det_bounds = None
        for c in det_coadds.values():
            pb = getattr(getattr(c, "psf", None), "bounds", None)
            if pb is None:            # legacy MultipleCellCoadd path
                det_bounds = None
                break
            if det_bounds is None:
                det_bounds = pb
                continue
            try:
                det_bounds = det_bounds.intersection(pb)
            except NoOverlapError as exc:
                # The bands' PSFs cover DISJOINT parts of the patch, so
                # there is no cell where all of them can be evaluated
                # and no multi-band detection is possible here. DM
                # raises rather than returning an empty bounds, so the
                # empty case has to be caught. Rare but real: 2 of 4259
                # deep-field patches, e.g. tract 5280 patch 50, where
                # one band covers 16350:18150 and another 14850:15750.
                raise NoWorkFound(
                    f"detection bands {bands} have disjoint PSF coverage "
                    f"for tract={tract} patch={patch}: {exc}"
                ) from exc
        if det_bounds is not None:
            n_common = sum(1 for _ in det_bounds.cell_indices())
            per_band = {b: sum(1 for _ in c.psf.bounds.cell_indices())
                        for b, c in det_coadds.items()}
            if any(v != n_common for v in per_band.values()):
                self.log.info(
                    "detection cells: %d common to %s (per band: %s)",
                    n_common, bands,
                    ", ".join("%s=%d" % kv for kv in sorted(per_band.items())),
                )
        det_cells = {b: self._dilated_cells(c, bounds=det_bounds)
                     for b, c in det_coadds.items()}

        def _detect_one(item):
            cell_id, det_cell = item
            cell_mask = self._cell_mask(
                stitched_mask_array,
                mask_origin,
                det_cell,
            )
            try:
                if len(bands) == 1:
                    data = self._prepare_cell(
                        det_cell,
                        band=bands[0],
                        seed=seed,
                        mag_zero=mag_zeros[bands[0]],
                        skyMap=skyMap,
                        tract=tract,
                        patch=patch,
                        mask_array=cell_mask,
                    )
                else:
                    data = self._prepare_cell_multiband(
                        {b: det_cells[b][cell_id] for b in bands},
                        bands=bands,
                        seed=seed,
                        mag_zeros=mag_zeros,
                        skyMap=skyMap,
                        tract=tract,
                        patch=patch,
                        mask_array=cell_mask,
                    )
                return self._run_anacal(**data)
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
                return None

        items = list(det_cells[bands[0]].items())
        det_cats: dict = {}
        for (cell_id, _), cat in zip(
            items, self._map_parallel(_detect_one, items)
        ):
            if cat is not None and len(cat) > 0:
                det_cats[cell_id] = cat
        del det_coadds

        self.log.info(
            "DETECT tract=%d patch=%d: %d cells scanned, %d cells with "
            "detections, %d detections total",
            tract, patch, len(items), len(det_cats),
            sum(len(c) for c in det_cats.values()),
        )
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
        """Photometric zeropoint, from either coadd flavour."""
        if hasattr(mca, "stitch"):
            photoCalib = mca.stitch().asExposure().getPhotoCalib()
        else:
            photoCalib = mca.to_legacy().getPhotoCalib()
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
            band_coadd = self._get_coadd(coadd_handles_dict[band])
            mag_zero = self._coadd_mag_zero(band_coadd)
            # Per-cell visit count for THIS band, straight from the
            # coadd's own provenance -- DP2 persists no nImage, and each
            # band has its own visit set (u can be 1 visit where i is 9).
            n_image_cells = self.n_image_per_cell(band_coadd)
            # Built ONCE per band: the views are cheap but the dict is
            # rebuilt for every cell if this sits inside _force_one.
            band_cells = self._dilated_cells(band_coadd)

            def _force_one(cell_id, band_cells=band_cells):
                cell = band_cells[cell_id]
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
                    cat = self._run_fpfs(**data)
                    cat = self._append_gauss_fluxes(
                        cat, data=data, band=band,
                    )
                    if self.config.doPsfHsmMoments:
                        # HSM on the COADD PSF model stamp. The
                        # provenance table's psf_shape_* are per-visit
                        # ADAPTIVE moments; adaptive moments are not
                        # linear in the profile, so a weighted mean of
                        # them is not the coadd PSF's moments -- and the
                        # per-visit spread within one cell reaches 167%.
                        # Measure the actual coadd PSF instead. (The
                        # synthetic ExposureF is released as this call
                        # exits.)
                        cat = self._append_psf_hsm_moments(
                            cat,
                            band=band,
                            hsm_exposure=make_psf_stamp_exposure(
                                cell.psf_image
                            ),
                            # From the cell's own WCS: the stamp
                            # exposure has none, and this is the grid
                            # the moments were measured on.
                            pixel_scale=float(
                                cell.wcs.getPixelScale().asArcseconds()
                            ),
                        )
                    if n_image_cells is not None:
                        n_vis = n_image_cells.get(
                            (int(cell_id.x), int(cell_id.y)), 0
                        )
                        cat = self.attach_n_inputs_column(
                            cat, np.full(len(cat), n_vis, dtype=np.int32),
                            band,
                        )
                    return cat
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
                    return None

            for cell_id, cat in zip(
                active_cell_ids,
                self._map_parallel(_force_one, active_cell_ids),
            ):
                if cat is not None:
                    cell_force_parts[cell_id].append(cat)
            # Release this band's coadd before loading the next one
            # (rebinding, not `del`: the closure above still names it).
            band_coadd = None

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
        detection: NDArray | None = None,
        **kwargs,
    ):
        """Run detection and forced measurement on cell-based coadds.

        When an external ``detection`` catalog is given, the internal
        detection step is skipped and the catalog is partitioned into
        the same per-cell groups internal detection would produce (cell
        inner regions tile the patch), so forced measurement runs -- and
        threads -- identically either way.  The output preserves the
        input row order; rows whose cell failed in any band are dropped.
        That catalog must carry ``ra``/``dec`` and ``wsel``/
        ``dwsel_dg1``/``dwsel_dg2``; pixel positions are derived from
        the sky coordinates and any it carries are ignored.  See
        :meth:`_ingest_external_detection` for why.

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
            Combined stitched anacal bitmask from
            BuildCellSystematicsTask (bit 0 = masked/cut, bit 1 =
            discontinuity, stamped per source as
            n_mask_discontinuity). If provided, per-cell masks are
            extracted by slicing.
        """
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)

        expected = set(self.config.bands)
        provided = set(coadd_handles_dict.keys())
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

        first_handle = next(iter(coadd_handles_dict.values()))
        seed = self._seed_from_handle(first_handle)

        # The mask already carries the anacal uint8 bit convention:
        # bit 0 = masked (cut), bit 1 = discontinuity (kept but stamped
        # into n_mask_discontinuity per source).
        if mask is not None:
            stitched_mask_array = mask.getArray().astype(np.uint8)
            mask_origin = (mask.getX0(), mask.getY0())
        else:
            stitched_mask_array = None
            mask_origin = None
            # Mask building lives entirely in BuildCellSystematicsTask now;
            # without its output, measurement runs unmasked (saturated
            # pixels, streaks, bright-star halos included).
            self.log.warning(
                "No systematics mask for tract=%d patch=%d; measuring "
                "with NO pixel masking. Run BuildCellSystematicsTask "
                "upstream unless this is intentional.",
                tract,
                patch,
            )

        order: dict | None = None
        if detection is not None:
            # External catalog: partition into the SAME per-cell groups
            # internal detection would produce (cell inner regions tile
            # the patch), so _force sees one interface and its cell loop
            # threads either way.  Input row order restored below.
            first_band = next(iter(coadd_handles_dict))
            mca = coadd_handles_dict[first_band].get()
            pixel_scale = float(
                skyMap[tract].getWcs().getPixelScale().asArcseconds()
            )
            regions = []
            for cell_id, cell in dict(mca.cells).items():
                ib = cell.inner.bbox
                regions.append(
                    (cell_id, ib.getBeginX(), ib.getBeginY(),
                     ib.getEndX(), ib.getEndY())
                )
            del mca
            # Pixel positions come from ra/dec through the tract WCS
            # (cell inner bboxes are in that frame); whatever the
            # catalog carried is overwritten.  This runs BEFORE the mask
            # stamping below, which samples at x1/x2.
            det_use = self._ingest_external_detection(
                detection, skyMap[tract].getWcs(), pixel_scale,
            )
            if stitched_mask_array is not None:
                # Stamp n_mask_base / n_mask_discontinuity from the
                # systematics mask (same C++ smoothing/sampling internal
                # detections get).  The n_mask_base cut itself happens
                # in C++ (ForceTask / process_image) via the fpfs/anacal
                # n_mask_base_max configs -- Python only stamps and
                # partitions.
                det_use = self._stamp_external_mask_fractions(
                    det_use,
                    stitched_mask_array,
                    mask_origin,
                    pixel_scale,
                    float(self.config.anacal.sigma_arcsec),
                )
            # Basic geometric selection only: rows in no existing cell's
            # inner region (outside the coadd, patch border, or holes)
            # are dropped by the partition.
            det_cats, order = self._partition_external_detection(
                det_use, regions, pixel_scale,
            )
            if not det_cats:
                raise NoWorkFound(
                    f"External detection catalog is empty "
                    f"(tract={tract}, patch={patch}); skipping this patch."
                )
        else:
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

        self.log.info(
            "FORCE  tract=%d patch=%d: %d cells detected -> %d cells forced, "
            "%d rows",
            tract, patch, len(det_cats), len(force_cats),
            sum(len(c) for c in force_cats.values()),
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
        if order is not None:
            # External catalogs are row-aligned to their producer; undo
            # the per-cell grouping.
            output = self._restore_input_order(
                output, list(force_cats.keys()), order,
            )
        output = self._finalize_catalog(
            output, seed=seed, skyMap=skyMap, tract=tract, patch=patch,
        )
        return Struct(anacalCatalog=output)
