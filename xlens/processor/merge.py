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

"""Per-tract merge of FPFS shape catalogs.

This pipeline task takes the per-patch ``*_coadd_anacal_*`` catalogs of a
single tract, stacks them into one tract-level catalog, combines the
multi-band shapelet moments into a single set of ``fpfs1_*`` columns
(updating ``fpfs1_e1``/``fpfs1_e2`` and their first-order shear
derivatives), then applies the local WCS distortion correction to the
combined ellipticity using the linearized tract WCS at each detection
position.

The connection schema mirrors :mod:`xlens.analysis.cluster` — inputs are
templated by ``inputName``  so the same task can merge catalogs at any of the
standard pipeline stages.
"""

__all__ = [
    "MergePipeConfig",
    "MergePipe",
    "MergePipeConnections",
]

import logging
from typing import Any

import lsst.geom as lgeom
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from astropy.table import Table, vstack
from lsst.pex.config import Field, ListField
from lsst.pipe.base import (
    NoWorkFound,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter

from ..utils.constants import FPFS_C0
from ..wcs import extract_perturbation_dm_wcs, sky_to_pixel


class MergePipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract"),
    defaultTemplates={
        "inputName": "deep_coadd_cell",
    },
):
    """Butler connections for :class:`MergePipe`."""

    skymap = cT.Input(
        doc="SkyMap used to look up the tract WCS.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    srcList = cT.Input(
        doc="Per-patch anacal catalogs to merge for one tract.",
        name="{inputName}_anacal_catalog",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch"),
        multiple=True,
        deferLoad=True,
    )
    pzList = cT.Input(
        doc=(
            "Optional per-patch photo-z point catalogs (output of "
            ":class:`xlens.processor.photoz.photoZPipe`). When provided, "
            "the photo-z columns are joined onto each patch's catalog "
            "before stacking."
        ),
        name="{inputName}_anacal_fzb_point",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch"),
        multiple=True,
        deferLoad=True,
        minimum=0,
    )
    mergedCatalog = cT.Output(
        doc=("Tract-level merged anacal catalog " "(band-combined + WCS-corrected)."),
        name="{inputName}_anacal_merged",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
    )


class MergePipeConfig(
    PipelineTaskConfig,
    pipelineConnections=MergePipeConnections,
):
    """Configuration for :class:`MergePipe`."""

    bands = ListField[str](
        doc="Survey-prefixed bands whose ``{band}_fpfs1_*`` moments are combined.",
        default=["lsst_g", "lsst_r", "lsst_i", "lsst_z"],
    )
    band_weights = ListField[float](
        doc=(
            "Optional fixed band weights matching ``bands`` (need not be "
            "normalised). When empty, weights are derived from the median "
            "1/(``{band}_flux_fpfs1_err``)**2 of the stacked catalog.\n\n"
            "PREFER fixed weights for production: a derived weight "
            "makes the estimator differ from tract to tract, and any "
            "weight built from the data carries a shear dependence "
            "that the response does not account for. Derive them once "
            "with the HSC compute script and pass the numbers here."
        ),
        default=[],
    )
    fpfs_c0 = Field[float](
        doc=(
            "C0 normalisation in ``e = m22c / (m00 + c0)``, on the fixed AB "
            "nanojansky zeropoint (MAG_ZERO_AB) the measurement normalizes "
            "onto -- used as-is, no rescaling."
        ),
        default=FPFS_C0,
    )
    fpfs_prefix = Field[str](
        doc="Per-band moment column prefix.",
        default="fpfs1_",
    )
    do_wcs_correction = Field[bool](
        doc=(
            "Apply the local WCS distortion correction to the "
            "band-combined ``fpfs1_e1``/``fpfs1_e2``. Set to ``False`` "
            "for simulated catalogs whose WCS is already free of shear "
            "and rotation; the band-combined ellipticity is then left "
            "untouched."
        ),
        default=True,
    )
    flipu_wcs = Field[bool](
        doc=(
            "Convert from GalSim u=West to LSST u=East at the end of "
            "the merge: negate spin-2 ``e2`` and every ``*_dg2`` "
            "derivative (``dwsel_dg2``, ``fpfs1_de1_dg2``, "
            "``fpfs1_de2_dg1``, ``fpfs1_dm00_dg2``, "
            "``fpfs1_dm20_dg2``, per-band ``{b}_dflux_fpfs1_dg2``) and "
            "swap the photo-z ``_2p`` and ``_2m`` distortion columns. "
            "PSF HSM columns are also flipped where appropriate "
            "(``{b}_ext_shapeHSM_HsmPsfMoments_xy`` and any "
            "``{b}_ext_shapeHSM_HigherOrderMomentsPSF_<pq>`` whose "
            "leading x-power ``p`` is odd, e.g. ``_12, _30, _13, _31``)."
        ),
        default=False,
    )

    def validate(self):
        super().validate()
        if len(self.band_weights) != 0 and len(self.band_weights) != len(self.bands):
            raise ValueError("band_weights, when set, must have the same length as bands")


class MergePipe(PipelineTask):
    """Merge per-patch FPFS shape catalogs into one tract-level catalog
    and apply the WCS distortion correction.
    """

    _DefaultName = "MergePipe"
    ConfigClass = MergePipeConfig

    def __init__(
        self,
        *,
        config: MergePipeConfig | None = None,
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
        assert isinstance(self.config, MergePipeConfig)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        outputs = self.run(tract=tract, **inputs)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        *,
        srcList,
        skymap,
        tract: int,
        pzList=None,
    ) -> Struct:
        """Merge the per-patch catalogs for one tract.

        Parameters
        ----------
        srcList : list
            Deferred handles (or tables) for each patch's anacal catalog.
        skymap : BaseSkyMap
            Sky map used to obtain the tract WCS.
        tract : int
            Tract identifier of the quantum.
        pzList : list or None, optional
            Deferred handles (or tables) for each patch's photo-z point
            catalog. When provided, the photo-z columns are joined onto
            the merged tract catalog by ``object_id`` after the band
            combination and WCS correction.
        """
        if not srcList:
            raise RuntimeError(f"No source catalogs for tract {tract}")
        try:
            n_patch_x = int(skymap[tract].getNumPatches()[0])
        except Exception:
            n_patch_x = None
        catalog = self._collect_src_tables(
            srcList, tract=tract, n_patch_x=n_patch_x)

        weights = self._derive_band_weights(catalog)
        catalog = self._combine_band_moments(catalog, weights)

        if self.config.do_wcs_correction:
            tract_info = skymap[tract]
            wcs = tract_info.getWcs()
            pixel_scale = float(wcs.getPixelScale().asArcseconds())
            catalog = self._apply_wcs_correction(catalog, wcs, pixel_scale)

        pz_cols: list[str] = []
        if pzList:
            before = set(catalog.colnames)
            catalog = self._join_photoz(catalog, pzList)
            pz_cols = [c for c in catalog.colnames if c not in before]

        if self.config.flipu_wcs:
            catalog = self._apply_flipu(catalog, pz_cols)

        catalog = self._finalize_columns(catalog, pz_cols)
        if len(catalog) == 0:
            # All surviving per-patch rows were dropped by the wsel /
            # is_primary / quality filters in `_finalize_columns`.
            # Don't write a zero-row tract product to disk; raise
            # NoWorkFound so bps marks this quantum as SKIPPED and
            # downstream consumers see no merged catalog for the tract
            # (same semantics as an upstream task with no detections).
            raise NoWorkFound(
                f"merged tract catalog is empty after filtering "
                f"(tract={tract}); skipping output."
            )
        return Struct(mergedCatalog=catalog)

    def _finalize_columns(
        self,
        catalog: Table,
        pz_cols: list[str],
    ) -> Table:
        """Project the catalog onto the published schema and keep only
        ``is_primary`` rows.

        Kept columns:

        - Detection / position / selection: ``ra``, ``dec``, ``x1``,
          ``x2``, ``x1_det``, ``x2_det``, ``wsel``, ``dwsel_dg1``,
          ``dwsel_dg2``, ``n_mask_base``, ``bkg``, ``dbkg_dg1``,
          ``dbkg_dg2``, ``tract_id``, ``patch_x``, ``patch_y``,
          ``is_primary``, ``object_id``.
        - Band-combined ellipticity and full shear response (already
          WCS-corrected): ``fpfs1_e1``, ``fpfs1_e2``, ``fpfs1_de1_dg1``,
          ``fpfs1_de1_dg2``, ``fpfs1_de2_dg1``, ``fpfs1_de2_dg2``.
        - Band-combined shape magnitude ``esq = e1**2 + e2**2`` and its
          first-order shear response ``desq_dg{1,2} = 2*(e1*de1_dg{c} +
          e2*de2_dg{c})``, derived from the WCS-corrected fpfs1
          ellipticities above. Enables downstream ``|e|^2 < emax**2``
          cuts with the ±γ variant built from ``desq_dg{c}``.
        - Band-combined shapelet moments and full shear responses:
          ``fpfs1_m00``, ``fpfs1_dm00_dg1``, ``fpfs1_dm00_dg2``,
          ``fpfs1_m20``, ``fpfs1_dm20_dg1``, ``fpfs1_dm20_dg2``.
        - Per-band fluxes: for each band ``b`` in ``self.config.bands``
          we keep ``{b}_flux_fpfs1``, ``{b}_dflux_fpfs1_dg1``,
          ``{b}_dflux_fpfs1_dg2``, ``{b}_flux_fpfs1_err``, and the flux
          S/N ``{b}_s2n_fpfs1``, ``{b}_ds2n_fpfs1_dg1``,
          ``{b}_ds2n_fpfs1_dg2``.
        - Photo-z columns appended by ``_join_photoz`` (when present).

        Per-band shapes/moments, the detection-band raw FPFS columns
        (``fpfs_e*``, ``fpfs_m*``) and the intermediate band-combined
        moments (``fpfs1_m00`` etc.) are dropped.

        Rows are filtered by ``is_primary`` and ``wsel > 1e-5`` so that
        downstream consumers do not need to repeat those cuts.
        """
        bands = list(self.config.bands)
        p = self.config.fpfs_prefix
        if "is_primary" not in catalog.colnames:
            raise RuntimeError("merge finalize: catalog is missing 'is_primary'")
        if "wsel" not in catalog.colnames:
            raise RuntimeError("merge finalize: catalog is missing 'wsel'")
        keep_rows = np.asarray(catalog["is_primary"], dtype=bool) & (np.asarray(catalog["wsel"]) > 1e-5)
        catalog = catalog[keep_rows]
        # Back-compat shim: rename legacy sigma_mag / dsigma_mag columns
        # (per-patch catalogs measured before add_magnitude_columns was
        # updated) so the current keep list finds them under the new
        # ``mag_err`` / ``dmag_err`` names.  Check ``dsigma_mag`` first
        # since ``sigma_mag`` is a substring of it.
        for c in list(catalog.colnames):
            new = c
            if "_dsigma_mag_" in c:
                new = c.replace("_dsigma_mag_", "_dmag_")
                if new.endswith("_dg1"):
                    new = new[:-4] + "_err_dg1"
                elif new.endswith("_dg2"):
                    new = new[:-4] + "_err_dg2"
            elif "_sigma_mag_" in c:
                new = c.replace("_sigma_mag_", "_mag_") + "_err"
            if new != c:
                catalog.rename_column(c, new)
        # Band-combined shape magnitude and its first-order shear
        # response. Computed on the already-WCS-corrected (and
        # flipu-corrected, when enabled) fpfs1 shape so downstream
        # cuts (TXPipe / step3) can apply |e|^2 < emax**2 in one
        # column read.
        e1 = np.asarray(catalog[f"{p}e1"], dtype=np.float64)
        e2 = np.asarray(catalog[f"{p}e2"], dtype=np.float64)
        de1_dg1 = np.asarray(catalog[f"{p}de1_dg1"], dtype=np.float64)
        de1_dg2 = np.asarray(catalog[f"{p}de1_dg2"], dtype=np.float64)
        de2_dg1 = np.asarray(catalog[f"{p}de2_dg1"], dtype=np.float64)
        de2_dg2 = np.asarray(catalog[f"{p}de2_dg2"], dtype=np.float64)
        catalog["esq"] = e1 * e1 + e2 * e2
        catalog["desq_dg1"] = 2.0 * (e1 * de1_dg1 + e2 * de2_dg1)
        catalog["desq_dg2"] = 2.0 * (e1 * de1_dg2 + e2 * de2_dg2)
        keep: list[str] = [
            "ra",
            "dec",
            "x1",
            "x2",
            "x1_det",
            "x2_det",
            "wsel",
            "dwsel_dg1",
            "dwsel_dg2",
            "n_mask_base",
            "tract_id",
            "patch_x",
            "patch_y",
            # Local background under the source, from the detection
            # image (AnaCal detector::measure_pixel), with its shear
            # response so a cut on it can be de-biased like any other
            # selection. Carried because a coherent background offset
            # -- scattered light, a sky-subtraction failure in one
            # visit -- is not caught by n_mask_base or by the coadd
            # mask planes, but does show up here: on such a region the
            # median bkg runs ~7x the surrounding value.
            "bkg",
            "dbkg_dg1",
            "dbkg_dg2",
            "object_id",
            f"{p}e1",
            f"{p}e2",
            f"{p}de1_dg1",
            f"{p}de1_dg2",
            f"{p}de2_dg1",
            f"{p}de2_dg2",
            "esq",
            "desq_dg1",
            "desq_dg2",
            f"{p}m00",
            f"{p}dm00_dg1",
            f"{p}dm00_dg2",
            f"{p}m20",
            f"{p}dm20_dg1",
            f"{p}dm20_dg2",
        ]
        # Per-source PSF-quality columns, carried through when the
        # measurement produced them. n_mask_discontinuity counts
        # INEXACT_PSF pixels in a box around the source (current
        # schema); psf_mask_value/psf_mask_frac are the older pair the
        # HSC scripts appended before the column moved into the
        # measurement task.
        for col in (
            "n_mask_discontinuity",
            "psf_mask_value",
            "psf_mask_frac",
        ):
            if col in catalog.colnames:
                keep.append(col)
        for b in bands:
            keep.extend(
                [
                    f"{b}_flux_fpfs1",
                    f"{b}_dflux_fpfs1_dg1",
                    f"{b}_dflux_fpfs1_dg2",
                    f"{b}_flux_fpfs1_err",
                    # Per-band flux S/N + shear response (s2n = flux/flux_err;
                    # spin-0, carried per-band like the flux itself).
                    f"{b}_s2n_fpfs1",
                    f"{b}_ds2n_fpfs1_dg1",
                    f"{b}_ds2n_fpfs1_dg2",
                    # Per-band AB magnitude + shear response
                    # (add_magnitude_columns; smooth-truncated
                    # flux_to_mag on the fixed MAG_ZERO_AB scale).
                    f"{b}_mag_fpfs1",
                    f"{b}_dmag_fpfs1_dg1",
                    f"{b}_dmag_fpfs1_dg2",
                    f"{b}_mag_fpfs1_err",
                    f"{b}_dmag_fpfs1_err_dg1",
                    f"{b}_dmag_fpfs1_err_dg2",
                ]
            )
            # Per-band gauss2 fluxes (only present when
            # measureCellCoadds was run with do_measure_flux_gauss=True).
            for col in (
                f"{b}_flux_gauss2",
                f"{b}_flux_gauss2_err",
                f"{b}_dflux_gauss2_dg1",
                f"{b}_dflux_gauss2_dg2",
                # AB magnitude + shear response for the gauss2 flux family
                # (added by add_magnitude_columns when gauss2 is present).
                f"{b}_mag_gauss2",
                f"{b}_dmag_gauss2_dg1",
                f"{b}_dmag_gauss2_dg2",
                f"{b}_mag_gauss2_err",
                f"{b}_dmag_gauss2_err_dg1",
                f"{b}_dmag_gauss2_err_dg2",
                # Coadd depth at the source: the Gaussian-weighted mean
                # of this band's nImage (visit-count map) over the
                # source footprint, so a source straddling a chip-gap
                # edge reports the blend rather than its centre pixel.
                # Present only when the measurement was given an
                # nImage for this band -- DP2 persists none, and the
                # HSC maps are a separate download.
                f"{b}_n_inputs",
            ):
                if col in catalog.colnames:
                    keep.append(col)
            # DRP-style PSF HSM moment columns (written by
            # measureCellCoadds when doPsfHsmMoments=True). Raw
            # pixel-frame values carried through unchanged — identical
            # for every source in a cell (per-cell broadcast). We keep
            # the canonical 4 second-order fields (xx, yy, xy, flag)
            # and every present higher-order column; older catalogs
            # that still carry `_flag_no_psf` / `_flag_galsim` are
            # filtered out here so the merged schema stays clean.
            psf_2nd = [
                f"{b}_ext_shapeHSM_HsmPsfMoments_xx",
                f"{b}_ext_shapeHSM_HsmPsfMoments_yy",
                f"{b}_ext_shapeHSM_HsmPsfMoments_xy",
                f"{b}_ext_shapeHSM_HsmPsfMoments_flag",
            ]
            for col in psf_2nd:
                if col in catalog.colnames:
                    keep.append(col)
            ho_prefix = f"{b}_ext_shapeHSM_HigherOrderMomentsPSF_"
            keep.extend(c for c in catalog.colnames if c.startswith(ho_prefix))
        keep.extend(pz_cols)

        missing = [c for c in keep if c not in catalog.colnames]
        if missing:
            raise RuntimeError(f"merge finalize: missing expected columns: {missing}")
        return catalog[keep]

    def _collect_src_tables(self, srcList, tract=None,
                            n_patch_x=None) -> Table:
        """Materialise per-patch anacal catalogs and stack them into one
        tract-level table, stamping where each row came from.

        ``tract_id`` / ``patch_x`` / ``patch_y`` are added HERE because
        this is the last point at which the per-patch identity exists:
        after the vstack a row is just a row, and any later attempt to
        recover its patch has to reverse-engineer it from ``object_id``.
        Downstream work needs them constantly -- inspecting an artefact,
        dropping one patch, splitting a field by region.

        The patch index is the butler's sequential id; it is unpacked
        with the skymap's own patches-per-row so this stays correct for
        a skymap that is not 9x9.
        """
        tables = []
        for h in srcList:
            t = h.get() if hasattr(h, "get") else h
            data_id = getattr(h, "dataId", None)
            if data_id is not None and tract is not None \
                    and n_patch_x is not None and len(t):
                seq = int(data_id["patch"])
                t["tract_id"] = np.full(len(t), int(tract), dtype=np.int32)
                t["patch_x"] = np.full(len(t), seq % n_patch_x,
                                       dtype=np.int32)
                t["patch_y"] = np.full(len(t), seq // n_patch_x,
                                       dtype=np.int32)
            tables.append(t)
        return vstack(tables, metadata_conflicts="silent")

    def _join_photoz(self, catalog: Table, pzList) -> Table:
        """Join the photo-z columns onto ``catalog`` matched by
        ``object_id``.

        Both catalogs are sorted by ``object_id`` and required to carry
        identical ID sequences after sorting; any mismatch (length
        difference, missing IDs, extra IDs, duplicates) raises. Once
        verified, the photo-z columns are appended row-aligned.
        """
        if "object_id" not in catalog.colnames:
            raise RuntimeError("merged anacal catalog is missing the 'object_id' column")
        pz_tables = [h.get() if hasattr(h, "get") else h for h in pzList]
        if not pz_tables:
            return catalog
        pz_cat = vstack(pz_tables, metadata_conflicts="silent")
        if "object_id" not in pz_cat.colnames:
            raise RuntimeError("photo-z catalog is missing the 'object_id' column")

        catalog.sort("object_id")
        pz_cat.sort("object_id")

        cat_id = np.asarray(catalog["object_id"])
        pz_id = np.asarray(pz_cat["object_id"])
        if not np.array_equal(cat_id, pz_id):
            raise RuntimeError(
                "object_id mismatch between merged anacal catalog and "
                "photo-z catalog after sorting; refusing to join"
            )

        for col in pz_cat.colnames:
            if col in catalog.colnames:
                continue
            catalog[col] = pz_cat[col]
        return catalog

    def _derive_band_weights(self, catalog: Table) -> np.ndarray:
        bands = list(self.config.bands)
        if len(self.config.band_weights) == len(bands):
            w = np.asarray(self.config.band_weights, dtype=float)
        else:
            w = np.empty(len(bands), dtype=float)
            for i, b in enumerate(bands):
                err = np.asarray(catalog[f"{b}_flux_fpfs1_err"])
                # Median is finite/positive given valid measurements; if
                # all values are bad treat the band as unweighted.
                med = float(np.median(err[np.isfinite(err) & (err > 0)]))
                w[i] = 1.0 / med**2 if med > 0 else 0.0
        if w.sum() <= 0.0:
            raise RuntimeError("All band weights are zero or negative.")
        w = w / w.sum()
        # Provenance: which band actually carries the shear is not
        # obvious from the mode alone, so record it per tract.
        self.log.info(
            "band weights (%s): %s",
            "fixed" if len(self.config.band_weights) == len(bands)
            else "derived 1/median(flux_fpfs1_err)**2",
            ", ".join("%s=%.1f%%" % (b, 100 * x) for b, x in zip(bands, w)),
        )
        return w

    def _combine_band_moments(
        self,
        catalog: Table,
        weights: np.ndarray,
    ) -> Table:
        bands = list(self.config.bands)
        p = self.config.fpfs_prefix
        n = len(catalog)
        m00 = np.zeros(n)
        m20 = np.zeros(n)
        m22c = np.zeros(n)
        m22s = np.zeros(n)
        dm00_dg1 = np.zeros(n)
        dm00_dg2 = np.zeros(n)
        dm20_dg1 = np.zeros(n)
        dm20_dg2 = np.zeros(n)
        dm22c_dg1 = np.zeros(n)
        dm22c_dg2 = np.zeros(n)
        dm22s_dg1 = np.zeros(n)
        dm22s_dg2 = np.zeros(n)
        for b, w in zip(bands, weights):
            m00 += w * np.asarray(catalog[f"{b}_{p}m00"])
            m20 += w * np.asarray(catalog[f"{b}_{p}m20"])
            m22c += w * np.asarray(catalog[f"{b}_{p}m22c"])
            m22s += w * np.asarray(catalog[f"{b}_{p}m22s"])
            dm00_dg1 += w * np.asarray(catalog[f"{b}_{p}dm00_dg1"])
            dm00_dg2 += w * np.asarray(catalog[f"{b}_{p}dm00_dg2"])
            dm20_dg1 += w * np.asarray(catalog[f"{b}_{p}dm20_dg1"])
            dm20_dg2 += w * np.asarray(catalog[f"{b}_{p}dm20_dg2"])
            dm22c_dg1 += w * np.asarray(catalog[f"{b}_{p}dm22c_dg1"])
            dm22c_dg2 += w * np.asarray(catalog[f"{b}_{p}dm22c_dg2"])
            dm22s_dg1 += w * np.asarray(catalog[f"{b}_{p}dm22s_dg1"])
            dm22s_dg2 += w * np.asarray(catalog[f"{b}_{p}dm22s_dg2"])

        # fpfs_c0 is already on the MAG_ZERO_AB scale, matching the moments.
        c0 = self.config.fpfs_c0
        denom = m00 + c0
        e1 = m22c / denom
        e2 = m22s / denom
        de1_dg1 = dm22c_dg1 / denom - m22c / denom**2 * dm00_dg1
        de1_dg2 = dm22c_dg2 / denom - m22c / denom**2 * dm00_dg2
        de2_dg1 = dm22s_dg1 / denom - m22s / denom**2 * dm00_dg1
        de2_dg2 = dm22s_dg2 / denom - m22s / denom**2 * dm00_dg2

        # ``m22c``, ``m22s`` and their ``dm22*_dg*`` derivatives are
        # consumed locally to derive the ellipticity + shear response,
        # but are not surfaced in the merged output.
        for col, val in (
            (f"{p}m00", m00),
            (f"{p}m20", m20),
            (f"{p}dm00_dg1", dm00_dg1),
            (f"{p}dm00_dg2", dm00_dg2),
            (f"{p}dm20_dg1", dm20_dg1),
            (f"{p}dm20_dg2", dm20_dg2),
            (f"{p}e1", e1),
            (f"{p}e2", e2),
            (f"{p}de1_dg1", de1_dg1),
            (f"{p}de1_dg2", de1_dg2),
            (f"{p}de2_dg1", de2_dg1),
            (f"{p}de2_dg2", de2_dg2),
        ):
            catalog[col] = val
        return catalog

    def _apply_wcs_correction(
        self,
        catalog: Table,
        wcs,
        pixel_scale: float,
    ) -> Table:
        p = self.config.fpfs_prefix
        # Reproject sky positions to tract pixel coordinates so the WCS
        # linearization is evaluated at exactly the object's ra/dec.
        x_pix, y_pix = sky_to_pixel(
            wcs,
            np.asarray(catalog["ra"]),
            np.asarray(catalog["dec"]),
        )

        n = len(catalog)
        g1_w = np.empty(n)
        g2_w = np.empty(n)
        rho_w = np.empty(n)
        kappa_w = np.empty(n)
        for i in range(n):
            g1, g2, rho, kappa = extract_perturbation_dm_wcs(
                wcs,
                lgeom.Point2D(float(x_pix[i]), float(y_pix[i])),
                pixel_scale,
            )
            g1_w[i] = g1
            g2_w[i] = g2
            rho_w[i] = rho
            kappa_w[i] = kappa

        e1 = np.asarray(catalog[f"{p}e1"])
        e2 = np.asarray(catalog[f"{p}e2"])
        de1_dg1 = np.asarray(catalog[f"{p}de1_dg1"])
        de1_dg2 = np.asarray(catalog[f"{p}de1_dg2"])
        de2_dg1 = np.asarray(catalog[f"{p}de2_dg1"])
        de2_dg2 = np.asarray(catalog[f"{p}de2_dg2"])

        # Step 1: undo the WCS shear to first order.  This is the FULL 2x2
        # contraction R_ij * g_j -- the off-diagonal response is not zero for
        # FPFS (de1_dg2 carries the -sqrt(3)*m44s term), so dropping the cross
        # terms would leave a residual proportional to the local WCS shear.
        e1_s = e1 - g1_w * de1_dg1 - g2_w * de1_dg2
        e2_s = e2 - g1_w * de2_dg1 - g2_w * de2_dg2

        # Step 2: undo WCS rotation by -2*rho on the spin-2 ellipticity.
        cos2 = np.cos(2.0 * rho_w)
        sin2 = np.sin(2.0 * rho_w)
        e1_c = e1_s * cos2 + e2_s * sin2
        e2_c = -e1_s * sin2 + e2_s * cos2

        catalog[f"{p}e1"] = e1_c
        catalog[f"{p}e2"] = e2_c

        # NOTE: the shear response (de*_dg*) and the spin-0 quantities (m00,
        # m20, fluxes) are deliberately NOT transformed by the WCS field
        # distortion.  Galaxies are randomly oriented, so the distortion does
        # not change their expectation; only the spin-2 ellipticity, which
        # carries the coherent signal, needs correcting.  ``kappa_w`` is
        # extracted above for the same reason it is unused here.
        return catalog

    def _apply_flipu(
        self,
        catalog: Table,
        pz_cols: list[str],
    ) -> Table:
        """Convert from GalSim u=West to LSST u=East convention.

        Negates the spin-2 ``e2`` and every derivative w.r.t. ``g2``
        (band-combined ``dm00_dg2`` / ``dm20_dg2``, the off-diagonal
        ``de1_dg2`` / ``de2_dg1`` if present, and per-band
        ``{b}_dflux_fpfs1_dg2``); the diagonal ``de2_dg2`` is left alone
        because both ``e2`` and ``g2`` flip sign and the two cancel.
        Photo-z columns ending in ``_2p`` are swapped with their ``_2m``
        partners since ``g2`` reverses sign.
        """
        p = self.config.fpfs_prefix
        bands = list(self.config.bands)

        if f"{p}e2" in catalog.colnames:
            catalog[f"{p}e2"] = -np.asarray(catalog[f"{p}e2"])

        for col in (
            f"{p}de1_dg2",
            f"{p}de2_dg1",
            f"{p}dm00_dg2",
            f"{p}dm20_dg2",
            "dwsel_dg2",
        ):
            if col in catalog.colnames:
                catalog[col] = -np.asarray(catalog[col])

        for b in bands:
            # Every per-band g2 response flips, not just the flux: s2n and
            # the magnitudes are built from dflux_dg2 (s2n = flux/flux_err,
            # dmag = dmag_dflux * dflux_dg2), so they must flip with it or
            # they end up with the opposite sign to the flux they derive
            # from.  Gaussian-aperture fluxes flip for the same reason.
            for col in (
                f"{b}_dflux_fpfs1_dg2",
                f"{b}_ds2n_fpfs1_dg2",
                f"{b}_dmag_fpfs1_dg2",
                f"{b}_dmag_fpfs1_err_dg2",
                f"{b}_dflux_gauss2_dg2",
                f"{b}_dmag_gauss2_dg2",
                f"{b}_dmag_gauss2_err_dg2",
            ):
                if col in catalog.colnames:
                    catalog[col] = -np.asarray(catalog[col])

            # PSF HSM moments: under x → -x the 2nd-order spin-2 cross
            # component (Ixy) and every higher-order M_{pq} with p odd
            # (e.g. M_12, M_30, M_13, M_31) flip sign. Reflection
            # symmetry doesn't touch the full WCS shear / kappa /
            # rotation — that intentionally stays out of this step.
            ixy_col = f"{b}_ext_shapeHSM_HsmPsfMoments_xy"
            if ixy_col in catalog.colnames:
                catalog[ixy_col] = -np.asarray(catalog[ixy_col])
            ho_prefix = f"{b}_ext_shapeHSM_HigherOrderMomentsPSF_"
            for col in list(catalog.colnames):
                if not col.startswith(ho_prefix):
                    continue
                suffix = col[len(ho_prefix):]
                if len(suffix) != 2 or not suffix.isdigit():
                    continue  # skip _flag etc.
                p = int(suffix[0])
                if p % 2 == 1:
                    catalog[col] = -np.asarray(catalog[col])

        for col_2p in [c for c in pz_cols if c.endswith("_2p")]:
            col_2m = col_2p[:-3] + "_2m"
            if col_2m not in catalog.colnames:
                continue
            tmp = np.asarray(catalog[col_2p]).copy()
            catalog[col_2p] = np.asarray(catalog[col_2m])
            catalog[col_2m] = tmp
        return catalog
