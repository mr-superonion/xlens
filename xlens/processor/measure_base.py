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
    n_mask_base_max = Field[float](
        doc=(
            "Mask-cut threshold on ``n_mask_base``, the Gaussian-"
            "weighted masked fraction in [0, 1]. ONE knob drives the "
            "whole cut:\n\n"
            "  * the anacal and fpfs subtasks receive it as a run() "
            "argument and skip those sources in C++, so no time is "
            "spent measuring rows that are about to be discarded "
            "(detection and forced measurement therefore cannot skip "
            "different sources);\n"
            "  * the finalized catalog then DROPS the rows outright, "
            "so the per-patch output carries no masked sources at all.\n\n"
            "The default 1.0 removes exactly the unusable rows: a "
            "source whose kernel footprint is entirely masked, and one "
            "carrying the psf_invalid sentinel (also 1.0, written when "
            "a band has no valid PSF model there). Lower it -- 0.035 "
            "is the value the analysis cuts on -- to drop partially "
            "masked sources during measurement instead of downstream."
        ),
        default=1.0,
    )

    def _fpfs_required(self) -> bool:
        """Whether the fpfs subtask will actually run (hook for subclasses)."""
        return True

    def validate(self):
        super().validate()
        if not 0.0 < self.n_mask_base_max <= 1.0:
            raise FieldValidationError(
                self.__class__.n_mask_base_max,
                self,
                "n_mask_base_max must be in (0, 1]: n_mask_base is a "
                f"masked FRACTION, got {self.n_mask_base_max}.",
            )
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

    n_image_min = Field[int](
        doc=(
            "Coverage cut from the coadd nImage (per-band visit-count "
            "map, dataset type {inputName}_n_image): pixels with fewer "
            "than this many contributing visits are masked (bit 0) "
            "before measurement. 0 disables the cut; the connection is "
            "optional, so a repo without nImage (e.g. DP2, which does "
            "not register it) simply runs without it.\n\n"
            "Low-coverage pixels are where the coadd's outlier "
            "rejection has the least leverage -- a contaminated frame "
            "is 1/4 of a 4-visit stack -- so artefacts (ghosts, "
            "satellite glints) survive there. HSC's full-depth "
            "criterion is >= 4 visits in g,r and >= 6 in i,z,y."
        ),
        default=0,
    )
    n_image_bands = ListField[str](
        doc=(
            "Bands whose nImage is used for the n_image_min cut; empty "
            "means every band supplied. A pixel is masked when ANY of "
            "these bands is below the threshold."
        ),
        default=[],
    )

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
            "{band}_ext_shapeHSM_HsmPsfMoments_{xx,yy,xy,flag,...} "
            "(second moments in ARCSEC**2, converted from the plugin's "
            "pixel**2 using the coadd WCS) and "
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

    def n_image_handles_dict(self, inputs: dict) -> dict:
        """``{band: handle}`` for the optional nImage inputs.

        Empty when the connection is absent from the repo (DP2 does not
        register ``deep_coadd_n_image``) or when the quantum simply had
        none, so callers can treat "no coverage map" as "no cut".
        """
        handles = inputs.get("nImage", None) or []
        return {h.dataId["band"]: h for h in handles}

    def n_inputs_at(self, n_image, x_pix, y_pix, *, sigma, scale):
        """Gaussian-weighted mean coverage at each source position.

        Delegates to ``anacal.mask.gaussian_average_at_sources``: the
        same kernel that stamps ``n_mask_base`` at detection, but
        normalised (sum(K*n)/sum(K)), so the value is the average number
        of inputs over the pixels the source actually spans rather than
        the single pixel its centre lands on -- a source straddling a
        chip-gap edge reports the blend, which is what matters for a
        measurement that weights those pixels together. Sources outside
        the image get 0.
        """
        return np.asarray(
            anacal.mask.gaussian_average_at_sources(
                np.ascontiguousarray(n_image, dtype=np.float32),
                np.ascontiguousarray(x_pix, dtype=np.float64),
                np.ascontiguousarray(y_pix, dtype=np.float64),
                float(sigma),
                float(scale),
            )
        )

    def n_image_per_cell(self, cell_coadd):
        """``{(cell_i, cell_j): n_visits}`` from a cell coadd's provenance.

        The modern ``CellCoadd`` records a ``provenance.contributions``
        table -- one row per (cell, visit, detector) that went into the
        coadd -- so the visit count per cell is exact, with no
        exposure-time arithmetic and no extra dataset. Rows are counted
        by DISTINCT visit: a visit straddling a detector boundary
        contributes two rows but is still one epoch, which is the
        nImage convention.

        Returns None for legacy coadds without provenance (the column is
        then simply absent).
        """
        prov = getattr(cell_coadd, "provenance", None)
        con = getattr(prov, "contributions", None) if prov is not None else None
        if con is None or len(con) == 0:
            return None
        try:
            ci = np.asarray(con["cell_i"], dtype=int)
            cj = np.asarray(con["cell_j"], dtype=int)
            visit = np.asarray(con["visit"])
        except (KeyError, TypeError):
            self.log.warning("coadd provenance has no cell/visit columns")
            return None
        seen: dict = {}
        for i, j, v in zip(ci, cj, visit):
            seen.setdefault((int(i), int(j)), set()).add(v)
        return {k: len(v) for k, v in seen.items()}

    def attach_n_inputs_column(self, cat, values, band: str):
        """Merge a per-source ``{survey}_{band}_n_inputs`` column."""
        assert isinstance(self.config, MeasureBandsConfigBase)
        prefix = (
            "%s_%s_" % (self.config.survey, band)
            if self.config.survey is not None else "%s_" % band
        )
        col = np.zeros(
            len(cat), dtype=[("%sn_inputs" % prefix, np.float32)]
        )
        col["%sn_inputs" % prefix] = values
        return np.asarray(rfn.merge_arrays([cat, col], flatten=True))

    def apply_n_image_cut(self, mask_array, n_image_handles: dict):
        """OR low-coverage pixels into bit 0 of ``mask_array``.

        Returns the mask (possibly newly allocated when ``mask_array``
        was None) or None when no cut applies -- either the threshold
        is off or no nImage was supplied.
        """
        assert isinstance(self.config, MeasureBandsConfigBase)
        if self.config.n_image_min <= 0 or not n_image_handles:
            return mask_array
        want = list(self.config.n_image_bands) or list(n_image_handles)
        low = None
        for band in want:
            handle = n_image_handles.get(band)
            if handle is None:
                self.log.warning("no nImage for band %s; not cut", band)
                continue
            nim = np.asarray(handle.get().array)
            below = nim < self.config.n_image_min
            low = below if low is None else (low | below)
        if low is None:
            return mask_array
        self.log.info(
            "nImage cut (< %d visits in %s): masking %.2f%% of the patch",
            self.config.n_image_min, ",".join(want), 100.0 * low.mean(),
        )
        if mask_array is None:
            return low.astype(np.uint8)
        mask_array = np.asarray(mask_array, dtype=np.uint8).copy()
        mask_array[low] |= 1
        return mask_array

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

    def _ingest_external_detection(
        self,
        detection: NDArray,
        wcs,
        pixel_scale: float,
    ) -> NDArray:
        """Return ``detection`` with its pixel positions rebuilt from sky.

        THE CONTRACT for an external detection catalog is sky
        coordinates plus a selection weight:

        - ``ra``, ``dec`` (degrees) -- the positions.  They are the only
          frame-independent identity a catalog can carry: ``x1``/``x2``
          are pixels of ONE exposure (scaled by that exposure's pixel
          scale, offset by its bbox origin), so a catalog forced onto
          several bands, patches or surveys cannot carry meaningful
          ones.  Whatever pixel columns arrive are therefore OVERWRITTEN
          here, never trusted.
        - ``wsel``, ``dwsel_dg1``, ``dwsel_dg2`` -- the selection weight
          and its shear derivatives.  Nothing downstream recomputes
          them: the merged output takes them straight from this catalog,
          and ``_select_rows`` drops every row with ``wsel <= 1e-5``.
          There is deliberately NO default, because both plausible ones
          are wrong.  Zero silently empties the output (a catalog that
          measured perfectly well disappears in finalization), and 1.0
          asserts a shear-INDEPENDENT selection, which almost never
          holds: a detection made in another band or another survey
          still looks at the same sheared sky, so its selection responds
          to shear just as this band's would.  Only a selection made on
          PRE-LENSED properties -- truth-catalog quantities in a
          simulation -- may set ``wsel = 1, dwsel_dg = 0``.  The caller
          has to state which case it is.

        Everything else (fluxes, moments, mask fractions) is filled by
        measurement downstream.
        """
        names = set(detection.dtype.names or ())
        missing = [
            c for c in ("ra", "dec", "wsel", "dwsel_dg1", "dwsel_dg2")
            if c not in names
        ]
        if missing:
            raise ValueError(
                f"External detection catalog is missing required "
                f"column(s) {missing}. It must carry sky positions "
                "(ra, dec) and the selection weight with its shear "
                "derivatives (wsel, dwsel_dg1, dwsel_dg2); pixel "
                "positions are derived here and never read from the "
                "input."
            )
        ra = np.asarray(detection["ra"], dtype=np.float64)
        dec = np.asarray(detection["dec"], dtype=np.float64)
        if not np.any(ra != 0.0) and not np.any(dec != 0.0):
            raise ValueError(
                "External detection catalog has ra = dec = 0 for every "
                "row. Sky positions are required: pixel positions are "
                "derived from them, and is_primary is decided by which "
                "tract contains them."
            )
        # The position columns are OUTPUTS of this step, so a catalog
        # that never had them (a survey catalog carries sky coordinates,
        # not another instrument's pixels) is as valid an input as one
        # whose values we are about to discard.
        out = detection.copy()
        absent = [
            c for c in ("x1", "x2", "x1_det", "x2_det")
            if c not in names
        ]
        if absent:
            out = rfn.append_fields(
                out,
                absent,
                [np.zeros(len(out), dtype=np.float64) for _ in absent],
                usemask=False,
            )
        px, py = wcs.skyToPixelArray(ra, dec, degrees=True)
        # x1/x2 are pixels scaled to arcsec in the exposure's own frame,
        # which is what anacal's catalog and the region partition below
        # both expect.  x1_det/x2_det are the detection positions; for
        # an external catalog they are the same points, and forced
        # measurement is what may move x1/x2 away from them.
        out["x1"] = px * pixel_scale
        out["x2"] = py * pixel_scale
        out["x1_det"] = out["x1"]
        out["x2_det"] = out["x2"]
        n_bad = int(np.sum(~np.isfinite(px) | ~np.isfinite(py)))
        if n_bad:
            self.log.warning(
                "%d external detections have sky positions the WCS "
                "cannot map to pixels; they fall in no region and are "
                "dropped by the partition.",
                n_bad,
            )
        return out

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

    def _stamp_external_mask_fractions(
        self,
        detection: NDArray,
        mask_array: NDArray,
        mask_origin: tuple,
        pixel_scale: float,
        sigma_arcsec: float,
    ) -> NDArray:
        """Return a copy of ``detection`` with the mask fractions stamped.

        Delegates to the C++ ``anacal.mask.add_mask_fraction_columns``
        -- the same smoothing and sampling internal detections get:
        ``n_mask_base`` (bit 0) and ``n_mask_discontinuity`` (bit 1)
        are the Gaussian-weighted MEAN of that bit over the kernel,
        i.e. a masked fraction in [0, 1], with
        sigma = sigma_arcsec * sqrt(2) * 1.5.  The C++ samples at the
        model centre in the MASK's pixel frame, so the positions are
        shifted by the mask origin for the call; the returned catalog
        keeps the original coordinates.  Positions outside the mask
        keep their input fractions.
        """
        shifted = detection.copy()
        dx = float(mask_origin[0]) * pixel_scale
        dy = float(mask_origin[1]) * pixel_scale
        shifted["x1"] = shifted["x1"] - dx
        shifted["x2"] = shifted["x2"] - dy
        stamped = anacal.mask.add_mask_fraction_columns(
            shifted,
            np.ascontiguousarray(mask_array, dtype=np.uint8),
            float(sigma_arcsec) * np.sqrt(2.0) * 1.5,
            pixel_scale,
        )
        out = detection.copy()
        out["n_mask_base"] = stamped["n_mask_base"]
        out["n_mask_discontinuity"] = stamped["n_mask_discontinuity"]
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

    def _run_anacal(self, **data):
        """Detection + detection-band measurement, with the mask cut."""
        return self.anacal.run(
            n_mask_base_max=self.config.n_mask_base_max, **data
        )

    def _run_fpfs(self, **data):
        """Forced FPFS measurement, with the mask cut.

        The threshold has to reach BOTH AnaCal entry points -- the
        detector (``anacal.task.Task.process_image``) and the forced
        measurement (``anacal.fpfs.process_image`` -> ``ForceTask``) --
        because they are separate C++ calls and the second never sees
        the first's configuration.  A source over the threshold is kept
        as a zero-filled row rather than dropped there, so if only
        detection cut, forced measurement would still write real
        per-band values for it and the two halves of one catalog would
        disagree.  Neither subtask carries a field of its own: both are
        handed ``config.n_mask_base_max`` here, so no caller can set
        the two stages differently, and :meth:`_finalize_catalog` drops
        the rows on that same value.
        """
        return self.fpfs.run(
            n_mask_base_max=self.config.n_mask_base_max, **data
        )

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
            self._run_anacal(**data),
            band,
            survey=self.config.survey,
        )
        return np.asarray(rfn.merge_arrays([cat, gauss_cat], flatten=True))

    def _psf_hsm_moments_per_cell(
        self, cells, band: str, *, pixel_scale: float,
    ) -> dict:
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
                    pixel_scale=pixel_scale,
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
        pixel_scale: float,
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
            pixel_scale=pixel_scale,
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
        """Magnitude columns, stable object ids, is_primary, row cut."""
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
        return self._select_rows(catalog)

    def _select_rows(self, catalog: NDArray) -> NDArray:
        """Drop the rows the published catalog should never carry.

        Applied ONCE, here, on the finished catalog: a single boolean
        mask and a single fancy-index copy.  Doing it per cell instead
        would break the row alignment forced measurement depends on
        (the C++ keeps skipped sources as zero-filled rows precisely to
        stay aligned with the input positions), and doing it in the
        merge stage would mean writing, storing and re-reading rows
        that are discarded anyway.

        - ``is_primary``: the patch inner region owns the source, so
          the tract-level concatenation does not double-count sources
          in patch overlaps.  Defaults to True in the AnaCal schema, so
          a run without a skymap keeps everything.
        - ``wsel > 1e-5``: zero selection weight.  Both AnaCal weights
          are fail-closed (zero until detection and measurement give
          them a value), so this drops sources the measurement never
          completed -- including everything the detection-stage mask
          cut skipped.
        - ``n_mask_base < n_mask_base_max``: masked sources.  At the
          default 1.0 this removes the two unusable cases that share
          that value -- a fully masked footprint, and the psf_invalid
          sentinel written when some band has no valid PSF model.
          The sentinel is set during FORCED measurement, after ``wsel``
          is fixed, so this is the only term that catches it.
        """
        names = catalog.dtype.names or ()
        # TEMPORARY DEBUG: dump the pre-cut catalog when asked.
        import os as _os
        _dump = _os.environ.get("XLENS_DUMP_PRECUT")
        if _dump:
            np.save(_dump, catalog)
            self.log.info("pre-cut catalog dumped to %s (%d rows)",
                          _dump, len(catalog))
        keep = np.ones(len(catalog), dtype=bool)
        if "is_primary" in names:
            keep &= np.asarray(catalog["is_primary"], dtype=bool)
        if "wsel" in names:
            keep &= np.asarray(catalog["wsel"]) > 1e-5
        if "n_mask_base" in names:
            keep &= (
                np.asarray(catalog["n_mask_base"])
                < self.config.n_mask_base_max
            )
        if keep.all():
            # Nothing to drop: skip the copy entirely.
            return catalog
        # Per-cut counts, not just the total: the three cuts fire at
        # different STAGES (is_primary is geometry, wsel comes from
        # detection or from the external catalog, n_mask_base is
        # masking), so a bare total gives no idea which one is
        # responsible. Without this a source that entered with
        # wsel = 1.0 and then vanished looks like a detection failure,
        # which is exactly the wrong place to look.
        parts = []
        for _f, _m in (
            ("is_primary", np.asarray(catalog["is_primary"], dtype=bool)
             if "is_primary" in names else None),
            ("wsel<=1e-5", ~(np.asarray(catalog["wsel"]) > 1e-5)
             if "wsel" in names else None),
            ("n_mask_base", ~(np.asarray(catalog["n_mask_base"])
                              < self.config.n_mask_base_max)
             if "n_mask_base" in names else None),
        ):
            if _m is None:
                continue
            _drop = int((~_m).sum()) if _f == "is_primary" else int(_m.sum())
            parts.append("%s drops %d" % (_f, _drop))
        self.log.info(
            "Row cut: keeping %d of %d sources (%s; n_mask_base_max=%g).",
            int(keep.sum()), len(catalog), ", ".join(parts),
            self.config.n_mask_base_max,
        )
        return catalog[keep]
