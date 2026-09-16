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

"""Compare simulation and observation catalogs with 1-D and 2-D histograms.

:class:`CompareSimObsTask` is the sim-vs-obs analogue of
:class:`~xlens.analysis.shear_diagnostics.ShearStatsPipe`: it applies the
**same basic source selection** (the cluster-test cuts) to a simulation
catalog and an observation catalog of the same tract, then bins the same
property registries -- :data:`~xlens.analysis.shear_diagnostics.HIST_BINS`
(1-D) and :data:`~xlens.analysis.shear_diagnostics.HIST2D_BINS` (2-D) -- for
each.

The single output ``histStats`` is a stacked table of bin COUNTS, one row per
(``source``, histogram, bin), from which every plot is derived downstream:

* **1-D**: rows with ``iy == -1``; ``count`` versus the bin centre
  ``0.5 * (x_min + x_max)``, one curve per ``source`` ("sim"/"obs").
* **2-D**: rows with ``iy >= 0``; reshape ``count`` onto the ``(ix, iy)``
  grid (``nx = ix.max()+1``, ``ny = iy.max()+1``) and contour/imshow it,
  one panel per ``source``; the bin edges are in ``x_min/x_max`` and
  ``y_min/y_max``.

Counts are additive across tracts: stacking tracts = summing ``count`` over
rows with the same (``source``, ``name``, ``ix``, ``iy``). The selection and
binning are inherited unchanged from ``ShearStatsPipe``, so a sim/obs
comparison uses exactly the figures2 cuts and axes.
"""

__all__ = [
    "CompareSimObsConfig",
    "CompareSimObsTask",
    "CompareSimObsConnections",
]

import numpy as np
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base import PipelineTaskConnections, Struct
from numpy.typing import NDArray

from .shear_diagnostics import (
    HIST2D_BINS,
    HIST_BINS,
    ShearStatsPipe,
    ShearStatsPipeConfig,
    _axis_edges,
    _eval_columns,
)

#: One row per (source, histogram, bin). 2-D histograms fill (ix, iy) with the
#: y edges; 1-D ones use iy = -1 and NaN y edges. Same layout as
#: ``shear_diagnostics.HIST_DTYPE`` with a leading ``source`` label.
COMPARE_HIST_DTYPE = np.dtype([
    ("source", "U4"),      # "sim" or "obs"
    ("name", "U32"),
    ("ix", np.int32),
    ("iy", np.int32),
    ("x_min", np.float64),
    ("x_max", np.float64),
    ("y_min", np.float64),
    ("y_max", np.float64),
    ("count", np.int64),
])


class CompareSimObsConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract"),
    defaultTemplates={
        "simName": "sim_flagship_anacal",
        "obsName": "deep_coadd_cell_anacal",
    },
):
    simCatalog = cT.Input(
        doc="Merged simulation catalog for the tract (same schema as obs).",
        name="{simName}_merged",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
    )
    obsCatalog = cT.Input(
        doc="Merged observation (real data) catalog for the tract.",
        name="{obsName}_merged",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
    )
    histStats = cT.Output(
        doc=(
            "Stacked 1-D and 2-D histogram counts for sim and obs "
            "(one row per source/histogram/bin; decode per the module "
            "docstring)."
        ),
        name="sim_obs_hist_stats",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
    )


class CompareSimObsConfig(
    ShearStatsPipeConfig,
    pipelineConnections=CompareSimObsConnections,
):
    """Selection + binning config, inherited verbatim from ``ShearStatsPipe``.

    Every selection field (``survey``, ``mag_max``, ``snr_min``, ``esq_max``,
    ``n_mask_base_max``, ``trace_min``, ``psf_fwhm_max``, the per-band
    ``mag_max_bands`` / ``flux_err_*_bands`` / ``psf_fwhm_*_bands``) is inherited
    so a sim/obs comparison uses the identical cut. The mean-shear-only fields
    (``bin_ranges_file``, ``bin_percentile_trim``) are carried along but unused
    here.
    """


class CompareSimObsTask(ShearStatsPipe):
    """1-D/2-D histogram comparison of a sim vs an obs catalog for one tract.

    Reuses ``ShearStatsPipe``'s ``_select`` / ``_col`` / ``_localize`` so the
    selection and per-survey column binding are identical; only the inputs
    (two catalogs) and the output (histogram counts tagged by source) differ.
    """

    _DefaultName = "CompareSimObsTask"
    ConfigClass = CompareSimObsConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(
            sim_catalog=inputs["simCatalog"],
            obs_catalog=inputs["obsCatalog"],
        )
        butlerQC.put(outputs, outputRefs)

    @staticmethod
    def _as_array(catalog) -> NDArray:
        """ArrowAstropy Table (or numpy structured array) -> numpy array."""
        if hasattr(catalog, "as_array"):
            return np.asarray(catalog.as_array())
        return np.asarray(catalog)

    def _histograms(self, cat: NDArray, source: str) -> list:
        """1-D (HIST_BINS) and 2-D (HIST2D_BINS) count blocks for one source.

        ``cat`` is already selected. Missing-column histograms are skipped
        with a warning (a catalog need not carry every band).
        """
        rows = []

        for name, axis in HIST_BINS.items():
            expr, edges = _axis_edges(axis)
            expr = self._localize(expr)
            if len(edges) < 2:
                continue
            try:
                x = _eval_columns(cat, expr)
            except KeyError as err:
                self.log.warning(
                    "%s histogram %r skipped: missing column %s",
                    source, name, err,
                )
                continue
            counts, _ = np.histogram(x[np.isfinite(x)], bins=edges)
            block = np.zeros(len(counts), dtype=COMPARE_HIST_DTYPE)
            block["source"] = source
            block["name"] = name
            block["ix"] = np.arange(len(counts))
            block["iy"] = -1
            block["x_min"] = edges[:-1]
            block["x_max"] = edges[1:]
            block["y_min"] = np.nan
            block["y_max"] = np.nan
            block["count"] = counts
            rows.append(block)

        for name, (xaxis, yaxis) in HIST2D_BINS.items():
            xexpr, xedges = _axis_edges(xaxis)
            yexpr, yedges = _axis_edges(yaxis)
            xexpr = self._localize(xexpr)
            yexpr = self._localize(yexpr)
            if len(xedges) < 2 or len(yedges) < 2:
                continue
            try:
                x = _eval_columns(cat, xexpr)
                y = _eval_columns(cat, yexpr)
            except KeyError as err:
                self.log.warning(
                    "%s 2-D histogram %r skipped: missing column %s",
                    source, name, err,
                )
                continue
            good = np.isfinite(x) & np.isfinite(y)
            counts, _, _ = np.histogram2d(
                x[good], y[good], bins=(xedges, yedges)
            )
            nx, ny = counts.shape
            block = np.zeros(nx * ny, dtype=COMPARE_HIST_DTYPE)
            block["source"] = source
            block["name"] = name
            block["ix"] = np.repeat(np.arange(nx), ny)
            block["iy"] = np.tile(np.arange(ny), nx)
            block["x_min"] = xedges[:-1][block["ix"]]
            block["x_max"] = xedges[1:][block["ix"]]
            block["y_min"] = yedges[:-1][block["iy"]]
            block["y_max"] = yedges[1:][block["iy"]]
            block["count"] = counts.astype(np.int64).ravel()
            rows.append(block)

        return rows

    def run(self, *, sim_catalog, obs_catalog) -> Struct:
        """Select and histogram the sim and obs catalogs of one tract.

        Parameters
        ----------
        sim_catalog, obs_catalog
            Merged catalogs (ArrowAstropy tables or numpy structured arrays)
            sharing the schema the selection and registries expect.

        Returns
        -------
        Struct
            histStats : numpy structured array (COMPARE_HIST_DTYPE)
                Stacked 1-D and 2-D bin counts for both sources.
        """
        rows = []
        for source, catalog in (
            ("sim", sim_catalog),
            ("obs", obs_catalog),
        ):
            cat = self._as_array(catalog)
            cat = cat[self._select(cat)]
            self.log.info("%s: selected %d sources", source, len(cat))
            rows.extend(self._histograms(cat, source))

        hists = (
            np.concatenate(rows) if rows
            else np.zeros(0, dtype=COMPARE_HIST_DTYPE)
        )
        return Struct(histStats=hists)
