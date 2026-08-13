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

"""Per-tract shear-catalog diagnostics: mean shear binned by object /
PSF properties, and the standard sanity histograms.

:class:`ShearStatsPipe` consumes ONE tract's merged anacal catalog and
writes RAW SUMS, never ratios, exactly like the cluster lensing test:
per property bin it stores ``sum(w e1)``, ``sum(w e2)``, the two
component responses and ``n_gal``, so any collection of tracts can be
stacked by plain addition and the mean shear bootstrapped over tracts:

    <gamma_1>(bin) = sum_we1 / sum_r1,   <gamma_2>(bin) = sum_we2 / sum_r2

with the bin abscissa placed at the weighted mean property,
``sum_wx / sum_w``.  Everything (mean-shear bins and histograms) is
computed AFTER the basic source selection (``i_mag_gauss2 < 25``,
``i_s2n_fpfs1 > 10``, ``|e| < 0.4`` by default -- the cluster-test
cuts).

Shear convention (same as the cluster test): ``we_i = wsel * fpfs1_e_i``
and the per-object response ``r_i = wsel * de_i/dg_i + dwsel/dg_i * e_i``;
the cluster test's single ``sum_R`` is ``(sum_r1 + sum_r2) / 2``.

The binning definitions live in the module-level registries below
(:data:`PROPERTY_BINS`, :data:`HIST_BINS`, :data:`HIST2D_BINS`), plain
code rather than configuration: each entry is a numpy expression over
catalog columns plus its bin edges.  An entry whose columns are absent
from the catalog is skipped with a warning, not an error -- so the PSF
properties (from the ``{survey}_{band}_ext_shapeHSM_HsmPsfMoments_*``
columns written under ``doPsfHsmMoments``) activate automatically once
the measurement carries them.
"""

__all__ = [
    "ShearStatsPipeConfig",
    "ShearStatsPipe",
    "ShearStatsPipeConnections",
]

from typing import Any

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.pex.config import Field
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from numpy.typing import NDArray

# ----------------------------------------------------------------------
# Binning registries.  Axis = (expr, lo, hi, nbins, log); ``expr`` is a
# numpy expression over catalog columns (``np.`` allowed).  Entries with
# missing columns are skipped at run time with a warning.
# ----------------------------------------------------------------------
_XX = "hsc_i_ext_shapeHSM_HsmPsfMoments_xx"
_YY = "hsc_i_ext_shapeHSM_HsmPsfMoments_yy"
_XY = "hsc_i_ext_shapeHSM_HsmPsfMoments_xy"

# Mean-shear null-test properties.
PROPERTY_BINS = {
    "i_mag": ("hsc_i_mag_gauss2", 20.0, 25.0, 10, False),
    "i_snr": ("hsc_i_s2n_fpfs1", 10.0, 320.0, 10, True),
    "gmr_color": ("hsc_g_mag_gauss2 - hsc_r_mag_gauss2", -0.5, 2.5, 12, False),
    "rmi_color": ("hsc_r_mag_gauss2 - hsc_i_mag_gauss2", -0.5, 2.0, 12, False),
    "imz_color": ("hsc_i_mag_gauss2 - hsc_z_mag_gauss2", -0.5, 1.5, 12, False),
    "zmy_color": ("hsc_z_mag_gauss2 - hsc_y_mag_gauss2", -0.5, 1.0, 12, False),
    "mask_value": ("mask_value", 0.0, 200.0, 10, False),
    "discontinuity": ("discontinuity_mask_value", 0.0, 1000.0, 10, False),
    "psf_e1": (f"({_XX} - {_YY}) / ({_XX} + {_YY})", -0.06, 0.06, 12, False),
    "psf_e2": (f"2.0 * {_XY} / ({_XX} + {_YY})", -0.06, 0.06, 12, False),
    "psf_size": (f"np.sqrt(0.5 * ({_XX} + {_YY}))", 1.5, 3.0, 12, False),
}

# 1-D sanity histograms.
HIST_BINS = {
    "g_mag": ("hsc_g_mag_gauss2", 18.0, 27.0, 90, False),
    "r_mag": ("hsc_r_mag_gauss2", 18.0, 27.0, 90, False),
    "i_mag": ("hsc_i_mag_gauss2", 18.0, 27.0, 90, False),
    "z_mag": ("hsc_z_mag_gauss2", 18.0, 27.0, 90, False),
    "y_mag": ("hsc_y_mag_gauss2", 18.0, 27.0, 90, False),
    "abs_we1": ("np.abs(wsel * fpfs1_e1)", 0.0, 0.4, 80, False),
    "abs_we2": ("np.abs(wsel * fpfs1_e2)", 0.0, 0.4, 80, False),
    "response": (
        "0.5 * (wsel * fpfs1_de1_dg1 + dwsel_dg1 * fpfs1_e1"
        " + wsel * fpfs1_de2_dg2 + dwsel_dg2 * fpfs1_e2)",
        -1.0, 3.0, 80, False,
    ),
}

# 2-D histograms: name -> (x axis, y axis).
HIST2D_BINS = {
    "cmd_i_rmi": (
        ("hsc_i_mag_gauss2", 18.0, 26.0, 64, False),
        ("hsc_r_mag_gauss2 - hsc_i_mag_gauss2", -0.5, 2.5, 64, False),
    ),
}

# One row per (property, bin) of the mean-shear test.  All sums are over
# the selected sources in the bin; stacking tracts = adding rows with the
# same (property, bin).
MEANSHEAR_DTYPE = np.dtype([
    ("property", "U32"),
    ("bin", np.int32),
    ("x_min", np.float64),
    ("x_max", np.float64),
    ("n_gal", np.int64),
    ("sum_w", np.float64),
    ("sum_wx", np.float64),
    ("sum_we1", np.float64),
    ("sum_we2", np.float64),
    ("sum_r1", np.float64),
    ("sum_r2", np.float64),
])

# One row per (histogram, bin); 2-D histograms use (ix, iy) with the y
# edge columns filled, 1-D ones have iy = -1 and NaN y edges.
HIST_DTYPE = np.dtype([
    ("name", "U32"),
    ("ix", np.int32),
    ("iy", np.int32),
    ("x_min", np.float64),
    ("x_max", np.float64),
    ("y_min", np.float64),
    ("y_max", np.float64),
    ("count", np.int64),
])


def _axis_edges(axis) -> tuple[str, NDArray]:
    """Registry axis tuple -> (expr, bin edges)."""
    expr, lo, hi, nbins, log = axis
    if log:
        return expr, np.geomspace(lo, hi, nbins + 1)
    return expr, np.linspace(lo, hi, nbins + 1)


def _eval_columns(catalog: NDArray, expr: str) -> NDArray:
    """Evaluate a column expression (``np.`` allowed) on the catalog.

    Raises KeyError naming the first missing column, so callers can skip
    an entry whose inputs this catalog does not carry.
    """
    namespace = {}
    for name in catalog.dtype.names:
        if name in expr:
            namespace[name] = np.asarray(catalog[name], dtype=np.float64)
    try:
        return np.asarray(
            eval(expr, {"__builtins__": {}, "np": np}, namespace),
            dtype=np.float64,
        )
    except NameError as err:
        raise KeyError(str(err)) from err


class ShearStatsPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract"),
    defaultTemplates={"inputName": "deep_coadd"},
):
    mergedCatalog = cT.Input(
        doc="Tract-level merged anacal catalog from MergePipe.",
        name="{inputName}_anacal_merged",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
    )
    meanShearStats = cT.Output(
        doc=(
            "Mean-shear null-test sums: one row per (property, bin) "
            "with n_gal, sum_w, sum_wx, sum_we1, sum_we2, sum_r1, "
            "sum_r2 over the selected sources. Raw sums so tracts "
            "stack by addition; <gamma_i> = sum_we_i / sum_r_i and "
            "the bin abscissa is <x> = sum_wx / sum_w."
        ),
        name="{inputName}_anacal_meanshear_stats",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
    )
    histStats = cT.Output(
        doc=(
            "Histogram counts after the basic selection: one row per "
            "(histogram, bin); 2-D histograms use (ix, iy) with y "
            "edges, 1-D ones have iy = -1."
        ),
        name="{inputName}_anacal_hist_stats",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
    )


class ShearStatsPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=ShearStatsPipeConnections,
):
    mag_max = Field[float](
        doc="Basic selection: keep hsc_i_mag_gauss2 < mag_max (finite).",
        default=25.0,
    )
    snr_min = Field[float](
        doc="Basic selection: keep hsc_i_s2n_fpfs1 > snr_min.",
        default=10.0,
    )
    esq_max = Field[float](
        doc="Basic selection: keep esq = |e|^2 < esq_max (|e| < 0.4).",
        default=0.16,
    )


class ShearStatsPipe(PipelineTask):
    """Mean-shear null tests and sanity histograms for one tract."""

    _DefaultName = "ShearStatsPipe"
    ConfigClass = ShearStatsPipeConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, ShearStatsPipeConfig)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(catalog=inputs["mergedCatalog"])
        butlerQC.put(outputs, outputRefs)

    def _select(self, cat: NDArray) -> NDArray:
        """The basic source selection (cluster-test cuts)."""
        assert isinstance(self.config, ShearStatsPipeConfig)
        mag = np.asarray(cat["hsc_i_mag_gauss2"], dtype=np.float64)
        snr = np.asarray(cat["hsc_i_s2n_fpfs1"], dtype=np.float64)
        esq = np.asarray(cat["esq"], dtype=np.float64)
        return (
            np.isfinite(mag)
            & (mag < self.config.mag_max)
            & (snr > self.config.snr_min)
            & (esq < self.config.esq_max)
        )

    def run(self, *, catalog) -> Struct:
        assert isinstance(self.config, ShearStatsPipeConfig)
        # ArrowAstropy inputs arrive as astropy Tables; scripts may pass
        # numpy structured arrays directly.
        if hasattr(catalog, "as_array"):
            cat = np.asarray(catalog.as_array())
        else:
            cat = np.asarray(catalog)

        cat = cat[self._select(cat)]
        self.log.info("selected %d sources", len(cat))

        # Same estimator pieces as the cluster test.
        e1 = np.asarray(cat["fpfs1_e1"], dtype=np.float64)
        e2 = np.asarray(cat["fpfs1_e2"], dtype=np.float64)
        w = np.asarray(cat["wsel"], dtype=np.float64)
        we1 = w * e1
        we2 = w * e2
        r1 = np.asarray(cat["fpfs1_de1_dg1"], dtype=np.float64) * w \
            + np.asarray(cat["dwsel_dg1"], dtype=np.float64) * e1
        r2 = np.asarray(cat["fpfs1_de2_dg2"], dtype=np.float64) * w \
            + np.asarray(cat["dwsel_dg2"], dtype=np.float64) * e2

        stats_rows = []
        for name, axis in PROPERTY_BINS.items():
            expr, edges = _axis_edges(axis)
            try:
                x = _eval_columns(cat, expr)
            except KeyError as err:
                self.log.warning(
                    "property %r skipped: missing column %s", name, err
                )
                continue
            good = np.isfinite(x)
            idx = np.digitize(x[good], edges) - 1
            inside = (idx >= 0) & (idx < len(edges) - 1)
            idx = idx[inside]
            block = np.zeros(len(edges) - 1, dtype=MEANSHEAR_DTYPE)
            block["property"] = name
            block["bin"] = np.arange(len(edges) - 1)
            block["x_min"] = edges[:-1]
            block["x_max"] = edges[1:]
            block["n_gal"] = np.bincount(idx, minlength=len(edges) - 1)
            sums = {
                "sum_w": w[good][inside],
                "sum_wx": (w * x)[good][inside],
                "sum_we1": we1[good][inside],
                "sum_we2": we2[good][inside],
                "sum_r1": r1[good][inside],
                "sum_r2": r2[good][inside],
            }
            for col, values in sums.items():
                block[col] = np.bincount(
                    idx, weights=values, minlength=len(edges) - 1
                )
            stats_rows.append(block)
        mean_shear = (
            np.concatenate(stats_rows) if stats_rows
            else np.zeros(0, dtype=MEANSHEAR_DTYPE)
        )

        hist_rows = []
        for name, axis in HIST_BINS.items():
            expr, edges = _axis_edges(axis)
            try:
                x = _eval_columns(cat, expr)
            except KeyError as err:
                self.log.warning(
                    "histogram %r skipped: missing column %s", name, err
                )
                continue
            counts, _ = np.histogram(x[np.isfinite(x)], bins=edges)
            block = np.zeros(len(counts), dtype=HIST_DTYPE)
            block["name"] = name
            block["ix"] = np.arange(len(counts))
            block["iy"] = -1
            block["x_min"] = edges[:-1]
            block["x_max"] = edges[1:]
            block["y_min"] = np.nan
            block["y_max"] = np.nan
            block["count"] = counts
            hist_rows.append(block)
        for name, (xaxis, yaxis) in HIST2D_BINS.items():
            xexpr, xedges = _axis_edges(xaxis)
            yexpr, yedges = _axis_edges(yaxis)
            try:
                x = _eval_columns(cat, xexpr)
                y = _eval_columns(cat, yexpr)
            except KeyError as err:
                self.log.warning(
                    "2-D histogram %r skipped: missing column %s", name, err
                )
                continue
            good = np.isfinite(x) & np.isfinite(y)
            counts, _, _ = np.histogram2d(
                x[good], y[good], bins=(xedges, yedges)
            )
            nx, ny = counts.shape
            block = np.zeros(nx * ny, dtype=HIST_DTYPE)
            block["name"] = name
            block["ix"] = np.repeat(np.arange(nx), ny)
            block["iy"] = np.tile(np.arange(ny), nx)
            block["x_min"] = xedges[:-1][block["ix"]]
            block["x_max"] = xedges[1:][block["ix"]]
            block["y_min"] = yedges[:-1][block["iy"]]
            block["y_max"] = yedges[1:][block["iy"]]
            block["count"] = counts.astype(np.int64).ravel()
            hist_rows.append(block)
        hists = (
            np.concatenate(hist_rows) if hist_rows
            else np.zeros(0, dtype=HIST_DTYPE)
        )

        return Struct(meanShearStats=mean_shear, histStats=hists)
