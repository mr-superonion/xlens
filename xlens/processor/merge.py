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
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter

from ..wcs import extract_perturbation_dm_wcs


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
        doc="Bands whose ``{band}_fpfs1_*`` moments are combined.",
        default=["g", "r", "i", "z"],
    )
    band_weights = ListField[float](
        doc=(
            "Optional fixed band weights matching ``bands`` (need not be "
            "normalised). When empty, weights are derived from the median "
            "1/(``{band}_flux_fpfs1_err``)**2 of the stacked catalog."
        ),
        default=[],
    )
    fpfs_c0 = Field[float](
        doc="C0 normalisation in ``e = m22c / (m00 + c0)``.",
        default=8.4,
    )
    mag_zero = Field[float](
        doc=("Photometric zeropoint used to scale ``fpfs_c0`` " "(``c0 *= 10**((mag_zero - 30) / 2.5)``)."),
        default=30.0,
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
            "swap the photo-z ``_2p`` and ``_2m`` distortion columns."
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
        catalog = self._collect_src_tables(srcList)

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
          ``dwsel_dg2``, ``mask_value``, ``is_primary``, ``object_id``.
        - Band-combined ellipticity and full shear response (already
          WCS-corrected): ``fpfs1_e1``, ``fpfs1_e2``, ``fpfs1_de1_dg1``,
          ``fpfs1_de1_dg2``, ``fpfs1_de2_dg1``, ``fpfs1_de2_dg2``.
        - Band-combined shapelet moments and full shear responses:
          ``fpfs1_m00``, ``fpfs1_dm00_dg1``, ``fpfs1_dm00_dg2``,
          ``fpfs1_m20``, ``fpfs1_dm20_dg1``, ``fpfs1_dm20_dg2``.
        - Per-band fluxes: for each band ``b`` in ``self.config.bands``
          we keep ``{b}_flux_fpfs1``, ``{b}_dflux_fpfs1_dg1``,
          ``{b}_dflux_fpfs1_dg2``, ``{b}_flux_fpfs1_err``.
        - Photo-z columns appended by ``_join_photoz`` (when present).

        Per-band shapes/moments, the detection-band raw FPFS columns
        (``fpfs_e*``, ``fpfs_m*``), the intermediate band-combined
        moments (``fpfs1_m00`` etc.) and ``is_peak`` are dropped.

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
            "mask_value",
            "object_id",
            f"{p}e1",
            f"{p}e2",
            f"{p}de1_dg1",
            f"{p}de1_dg2",
            f"{p}de2_dg1",
            f"{p}de2_dg2",
            f"{p}m00",
            f"{p}dm00_dg1",
            f"{p}dm00_dg2",
            f"{p}m20",
            f"{p}dm20_dg1",
            f"{p}dm20_dg2",
        ]
        for b in bands:
            keep.extend(
                [
                    f"{b}_flux_fpfs1",
                    f"{b}_dflux_fpfs1_dg1",
                    f"{b}_dflux_fpfs1_dg2",
                    f"{b}_flux_fpfs1_err",
                ]
            )
        keep.extend(pz_cols)

        missing = [c for c in keep if c not in catalog.colnames]
        if missing:
            raise RuntimeError(f"merge finalize: missing expected columns: {missing}")
        return catalog[keep]

    def _collect_src_tables(self, srcList) -> Table:
        """Materialise per-patch anacal catalogs and stack them into one
        tract-level table.
        """
        return vstack(
            [h.get() if hasattr(h, "get") else h for h in srcList],
            metadata_conflicts="silent",
        )

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
        return w / w.sum()

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

        c0 = self.config.fpfs_c0 * 10.0 ** ((self.config.mag_zero - 30.0) / 2.5)
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
        x_pix, y_pix = wcs.skyToPixelArray(
            np.asarray(catalog["ra"]),
            np.asarray(catalog["dec"]),
            degrees=True,
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
        de2_dg2 = np.asarray(catalog[f"{p}de2_dg2"])

        # Step 1: undo WCS shear via the diagonal response.
        e1_s = e1 - g1_w * de1_dg1
        e2_s = e2 - g2_w * de2_dg2

        # Step 2: undo WCS rotation by -2*rho on the spin-2 ellipticity.
        cos2 = np.cos(2.0 * rho_w)
        sin2 = np.sin(2.0 * rho_w)
        e1_c = e1_s * cos2 + e2_s * sin2
        e2_c = -e1_s * sin2 + e2_s * cos2

        catalog[f"{p}e1"] = e1_c
        catalog[f"{p}e2"] = e2_c
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
            col = f"{b}_dflux_fpfs1_dg2"
            if col in catalog.colnames:
                catalog[col] = -np.asarray(catalog[col])

        for col_2p in [c for c in pz_cols if c.endswith("_2p")]:
            col_2m = col_2p[:-3] + "_2m"
            if col_2m not in catalog.colnames:
                continue
            tmp = np.asarray(catalog[col_2p]).copy()
            catalog[col_2p] = np.asarray(catalog[col_2m])
            catalog[col_2m] = tmp
        return catalog
