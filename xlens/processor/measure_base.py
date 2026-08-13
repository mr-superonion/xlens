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

"""Shared config and task base classes for the AnaCal measurement tasks.

``MeasureCoaddsPipe``, ``MeasureCellCoaddsPipe`` and ``AnacalDetectPipe``
share their anacal/fpfs subtask wiring, validation, seed derivation, the
per-band extra columns (Gaussian fluxes, PSF HSM moments) and catalog
finalization.  That logic lives here ONCE, so a fix lands in every task
at the same time -- the ``.fields`` validation crash existed in three
copies precisely because this wiring used to be triplicated.
"""

__all__ = [
    "AnacalMeasureConfigBase",
    "MeasureBandsConfigBase",
    "AnacalMeasureTaskBase",
]

from concurrent.futures import ThreadPoolExecutor

import anacal
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import numpy as np
from lsst.meas.base import SingleFrameMeasurementTask, SkyMapIdGeneratorConfig
from lsst.pex.config import (
    Config,
    ConfigurableField,
    Field,
    FieldValidationError,
    ListField,
)
from lsst.pipe.base import PipelineTask
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..catalog.utils import add_magnitude_columns
from ..utils.catalog import set_isPrimary
from ..utils.columns import select_band_gauss_fluxes
from ..utils.constants import MAG_ZERO_AB
from ..utils.image import (
    broadcast_psf_hsm_moments,
    build_psf_hsm_context,
    default_psf_hsm_plugin_config,
    measure_psf_hsm_moments,
)
from ..utils.image.hsm import make_psf_stamp_exposure, psf_array_to_image
from .anacal import AnacalTask
from .fpfs import FpfsMeasurementTask


class AnacalMeasureConfigBase(Config):
    """Config fields and validation shared by every AnaCal measurement task."""

    anacal = ConfigurableField(
        target=AnacalTask,
        doc="AnaCal Task for the detection stage (see detection_bands)",
    )
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def _fpfs_required(self) -> bool:
        """Whether the fpfs subtask will actually run (hook for subclasses)."""
        return True

    def validate(self):
        super().validate()
        # Single mask-cut knob for the whole task: set
        # ``config.anacal.mask_value_max`` and the fpfs subtask always
        # measures with the SAME threshold -- the two stages skipping
        # different sources would produce an inconsistent catalog.
        # Guarded so a re-validation of an already-frozen config (where
        # the two necessarily agree) is a no-op instead of a
        # "Cannot modify a frozen Config" error.
        if self.fpfs.mask_value_max != self.anacal.mask_value_max:
            self.fpfs.mask_value_max = self.anacal.mask_value_max
        if self._fpfs_required() and self.fpfs.sigma_shapelets1 < 0.0:
            raise FieldValidationError(
                self.fpfs.__class__.sigma_shapelets1,
                self,
                "sigma_shapelets1 in a wrong range",
            )


class MeasureBandsConfigBase(AnacalMeasureConfigBase):
    """Adds the fields shared by the two detect-then-force measurement tasks.

    The per-task superset check on ``detection_bands`` (against ``sim_bands``
    or ``bands``) stays in the subclasses: the reference list, the blamed
    field and the message genuinely differ.
    """

    do_measure_flux_gauss = Field[bool](
        doc=(
            "If True, also run AnaCal forced measurement during the "
            "force stage to extract per-band Gaussian fluxes and merge "
            "them into the output catalog."
        ),
        default=False,
    )
    detection_bands = ListField[str](
        doc=(
            "PHYSICAL bands combined to form the detection image. AnaCal "
            "removes each band's own PSF first and then averages the bands "
            "with inverse-variance weights, so the bands need not be "
            "PSF-matched. Forced measurement in the second stage is "
            "unaffected and still runs band by band. A single entry "
            "reproduces the previous single-band behaviour exactly."
        ),
        default=["i"],
    )
    survey = Field[str](
        doc=(
            "Survey name for the survey-prefixed output columns "
            "``{survey}_{band}_...`` and the survey-aware noise seed."
        ),
        default="lsst",
    )
    doPsfHsmMoments = Field[bool](
        doc=(
            "If True, run lsst.meas.extensions.shapeHSM.HsmPsfMomentsPlugin "
            "+ HigherOrderMomentsPSFPlugin (the same plugins DRP uses) once "
            "per (exposure or cell, band) on the PSF model, and broadcast "
            "the resulting PSF moments to every source. Adds "
            "{band}_ext_shapeHSM_HsmPsfMoments_{xx,yy,xy,flag,...} and "
            "{band}_ext_shapeHSM_HigherOrderMomentsPSF_{pq,flag} columns "
            "to the per-source anacal catalog."
        ),
        default=False,
    )
    psfHsmMeasurement = ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc=(
            "DRP-style single-frame measurement subtask used to evaluate "
            "the PSF model HSM moments. Only used when doPsfHsmMoments is "
            "True."
        ),
    )
    num_workers = Field[int](
        doc="Worker threads for the per-cell loops. 1 means a plain "
            "serial Python loop. Values >1 use a thread pool: AnaCal "
            "releases the GIL around its C++ compute (detection, "
            "measurement and the native per-source PSF draws), so the "
            "cell loops scale -- ~3x at 8 workers on a patch coadd. "
            "Results are independent of this value.",
        default=1,
    )

    def validate(self):
        super().validate()
        if self.num_workers < 1:
            raise FieldValidationError(
                self.__class__.num_workers,
                self,
                "num_workers must be >= 1.",
            )
        if len(self.detection_bands) == 0:
            raise FieldValidationError(
                self.__class__.detection_bands,
                self,
                "detection_bands must name at least one band.",
            )
        if len(set(self.detection_bands)) != len(self.detection_bands):
            raise FieldValidationError(
                self.__class__.detection_bands,
                self,
                f"detection_bands has duplicates: {list(self.detection_bands)}",
            )

    def setDefaults(self):
        super().setDefaults()
        self.anacal.force_size = True
        self.anacal.force_center = True
        # Shared DRP-equivalent plugin wiring, so column names match across
        # both measurement flavours. HigherOrderMomentsPSF defaults to
        # (min_order=3, max_order=4); to widen, set
        # cfg.psfHsmMeasurement.plugins[
        #     "ext_shapeHSM_HigherOrderMomentsPSF"].max_order = N.
        default_psf_hsm_plugin_config(self.psfHsmMeasurement)


class AnacalMeasureTaskBase(PipelineTask):
    """Base ``PipelineTask`` with the helpers shared by the AnaCal
    measurement tasks.

    Subclasses use a config deriving from :class:`MeasureBandsConfigBase`
    (except :meth:`_seed_from_handle` and :meth:`_normalize_noise_corr`,
    which only need ``idGenerator``).  Not runnable on its own: it defines
    no ``ConfigClass`` or connections.
    """

    def _map_parallel(self, fn, items: list) -> list:
        """Apply ``fn`` to ``items`` in order; threaded if num_workers > 1.

        The work-unit loops of the measurement tasks (AnaCal cells over a
        patch coadd, DM coadd cells for cell coadds) all run through
        here, so the
        ``num_workers`` config is the single dispatch point for threaded
        (and future GPU) AnaCal backends.
        """
        if self.config.num_workers <= 1 or len(items) <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(
            max_workers=self.config.num_workers
        ) as pool:
            return list(pool.map(fn, items))

    def _partition_external_detection(
        self,
        detection: NDArray,
        regions: list,
        pixel_scale: float,
    ) -> tuple[dict, dict]:
        """Split an external detection catalog into per-region groups.

        ``regions`` is a list of ``(key, x0, y0, x1, y1)`` half-open
        rectangles in GLOBAL pixel units -- the same inner regions
        (cells for patch coadds, cells for cell coadds) that internal
        detection uses, so ``_force`` sees the identical per-group
        interface either way.  A row belongs to the region whose
        rectangle contains it; rows contained by NO region -- outside
        the tiling, or over a hole in a region list that omits cells
        with no input data -- are DROPPED (no coadd data at that
        position means no measurement), mirroring internal detection,
        which can only ever find sources inside existing regions.

        Returns ``(det_cats, order)``: per-key sub-catalogs (non-empty
        only) and each key's original row indices, so callers can
        restore the input row order after per-group results are merged.
        """
        px = np.asarray(detection["x1_det"], float) / pixel_scale
        py = np.asarray(detection["x2_det"], float) / pixel_scale
        det_cats: dict = {}
        order: dict = {}
        assigned = np.zeros(len(px), dtype=bool)
        for key, x0, y0, x1, y1 in regions:
            sel = (
                ~assigned
                & (px >= x0) & (px < x1)
                & (py >= y0) & (py < y1)
            )
            if not np.any(sel):
                continue
            assigned |= sel
            idx = np.flatnonzero(sel)
            det_cats[key] = detection[idx]
            order[key] = idx
        n_left = int((~assigned).sum())
        if n_left:
            self.log.info(
                "%d external detections fall in no region "
                "(outside the tiling or over a missing cell); dropped.",
                n_left,
            )
        return det_cats, order

    def _stamp_external_mask_value(
        self,
        detection: NDArray,
        mask_array: NDArray,
        mask_origin: tuple,
        pixel_scale: float,
        sigma_arcsec: float,
    ) -> NDArray:
        """Return a copy of ``detection`` with ``mask_value`` stamped.

        Delegates to the C++ ``anacal.mask.add_pixel_mask_column`` --
        the same smoothing and sampling internal detections get
        (mask_value = int(1000 * Gaussian-smoothed 0/1 mask) at the
        source centre, sigma = sigma_arcsec * sqrt(2) * 1.5).  The C++
        samples at the model centre in the MASK's pixel frame, so the
        positions are shifted by the mask origin for the call; the
        returned catalog keeps the original coordinates.  Positions
        outside the mask keep their input mask_value.
        """
        shifted = detection.copy()
        dx = float(mask_origin[0]) * pixel_scale
        dy = float(mask_origin[1]) * pixel_scale
        shifted["x1"] = shifted["x1"] - dx
        shifted["x2"] = shifted["x2"] - dy
        stamped = anacal.mask.add_pixel_mask_column(
            shifted,
            (np.asarray(mask_array) > 0).astype(np.int16),
            float(sigma_arcsec) * np.sqrt(2.0) * 1.5,
            pixel_scale,
        )
        out = detection.copy()
        out["mask_value"] = stamped["mask_value"]
        return out

    def _restore_input_order(
        self,
        catalog: NDArray,
        force_keys: list,
        order: dict,
    ) -> NDArray:
        """Reorder per-group concatenated rows back to input row order."""
        orig = np.concatenate([order[key] for key in force_keys])
        assert len(orig) == len(catalog)
        return catalog[np.argsort(orig, kind="stable")]

    def _make_measure_subtasks(self) -> None:
        """anacal + fpfs subtasks, plus the PSF-HSM subtask when enabled."""
        self.makeSubtask("anacal")
        self.makeSubtask("fpfs")
        if self.config.doPsfHsmMoments:
            # Schema is shared across every measurement; the plugins
            # register their fields on construction.
            schema = afwTable.SourceTable.makeMinimalSchema()
            self.makeSubtask(
                "psfHsmMeasurement",
                schema=schema,
                algMetadata=dafBase.PropertyList(),
            )
            self._psfHsmCtx = build_psf_hsm_context(
                schema, self.config.psfHsmMeasurement,
            )

    def _seed_from_handle(self, handle) -> int:
        """Patch-level seed from a butler handle's dataId."""
        return self.config.idGenerator.apply(handle.dataId).catalog_id

    def _normalize_noise_corr(self, noise_corr: NDArray | None) -> NDArray | None:
        """Normalize a noise-correlation image to 1 at its centre pixel.

        Returns ``None`` for missing input or a non-positive peak; raises if
        the maximum is not at the centre pixel (the kernel would then be
        applied off-centre).
        """
        if noise_corr is None:
            return None
        variance = float(np.amax(noise_corr))
        if variance <= 0:
            return None
        noise_corr = noise_corr / variance
        ny, nx = noise_corr.shape
        if not np.isclose(noise_corr[ny // 2, nx // 2], 1.0):
            raise RuntimeError(
                "Noise correlation is not normalized to 1 at the center pixel."
            )
        return noise_corr

    def _append_gauss_fluxes(
        self,
        cat: NDArray,
        *,
        data: dict,
        band: str,
    ) -> NDArray:
        """Merge the per-band Gaussian fluxes into a forced catalog.

        No-op unless ``do_measure_flux_gauss`` is set.  ``data`` is the
        prepared measurement dict for this band, reused for the AnaCal
        forced run that produces the fluxes.
        """
        if not self.config.do_measure_flux_gauss:
            return cat
        gauss_cat = select_band_gauss_fluxes(
            self.anacal.run(**data),
            band,
            survey=self.config.survey,
        )
        return np.asarray(rfn.merge_arrays([cat, gauss_cat], flatten=True))

    def _psf_hsm_moments_per_cell(self, cells, band: str) -> dict:
        """PSF HSM moments for every cell, measured SERIALLY.

        The HSM plugins are Python + pybind11 without a GIL release, so
        running them inside the ``_map_parallel`` cell loop serialises
        the threads that AnaCal's GIL-free C++ was overlapping. They also
        do not depend on the sources -- only on the cell's PSF stamp --
        so they belong outside the parallel section entirely.

        Returns ``{cell index: moments dict}``; empty when the option is
        off, so the caller can treat "no entry" as "nothing to attach".
        """
        if not self.config.doPsfHsmMoments:
            return {}
        out = {}
        for bb in cells:
            psf = getattr(bb, "psf_image", None)
            if psf is None:
                arr = getattr(bb, "psf_array", None)
                if arr is None:
                    continue
                arr = np.asarray(arr)
                if arr.ndim != 2:      # multiband stack: this band's slice
                    continue
                psf = psf_array_to_image(arr)
            try:
                out[bb.index] = measure_psf_hsm_moments(
                    self._psfHsmCtx, self.psfHsmMeasurement,
                    make_psf_stamp_exposure(psf),
                )
            except Exception as exc:
                self.log.warning(
                    "PSF HSM failed for cell %s band %s: %s",
                    bb.index, band, exc,
                )
        return out

    def _attach_psf_hsm_moments(
        self,
        cat: NDArray,
        *,
        band: str,
        moments: dict | None,
    ) -> NDArray:
        """Broadcast one cell's pre-measured PSF moments onto its rows."""
        if not self.config.doPsfHsmMoments or not moments:
            return cat
        psf_cols = broadcast_psf_hsm_moments(
            moments, band, n=len(cat), survey=self.config.survey,
        )
        return np.asarray(rfn.merge_arrays([cat, psf_cols], flatten=True))

    def _append_psf_hsm_moments(
        self,
        cat: NDArray,
        *,
        band: str,
        hsm_exposure,
    ) -> NDArray:
        """Merge the broadcast PSF HSM moments into a forced catalog.

        No-op unless ``doPsfHsmMoments`` is set.  ``hsm_exposure`` is what
        the PSF HSM plugins measure on -- the band exposure for patch
        coadds, the synthetic PSF stamp exposure for cell coadds.  One
        measurement per (exposure or cell, band), broadcast across all
        sources of this band.
        """
        if not self.config.doPsfHsmMoments or hsm_exposure is None:
            return cat
        psf_moments = measure_psf_hsm_moments(
            self._psfHsmCtx, self.psfHsmMeasurement, hsm_exposure,
        )
        psf_cols = broadcast_psf_hsm_moments(
            psf_moments, band, n=len(cat),
            survey=self.config.survey,
        )
        return np.asarray(rfn.merge_arrays([cat, psf_cols], flatten=True))

    def _finalize_catalog(
        self,
        catalog: NDArray,
        *,
        seed: int,
        skyMap,
        tract: int,
        patch: int,
    ) -> NDArray:
        """Magnitude columns, stable object ids, and is_primary flags."""
        # Per-band AB magnitude + shear response for each published flux
        # family (fluxes are on the fixed MAG_ZERO_AB zeropoint here).
        catalog = np.asarray(add_magnitude_columns(catalog, MAG_ZERO_AB))
        # Stable per-object IDs derived from the patch-level seed. Used
        # downstream by ``photoZPipe`` and any object-level joiners.
        object_ids = np.int64(seed) * np.int64(1_000_000) + np.arange(
            len(catalog), dtype=np.int64
        )
        catalog = rfn.append_fields(
            catalog,
            "object_id",
            object_ids,
            usemask=False,
        )
        if skyMap is not None:
            # Use skymap's patchInfo for is_primary: the skymap patch inner
            # bbox defines the non-overlapping tiling (not the exposure or
            # cell-coadd bbox).
            tractInfo = skyMap[tract]
            patchInfo = tractInfo[patch]
            pixel_scale = float(
                tractInfo.getWcs().getPixelScale().asArcseconds()
            )
            set_isPrimary(catalog, skyMap, tractInfo, patchInfo, pixel_scale)
        return catalog
