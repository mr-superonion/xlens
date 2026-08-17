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
``i_s2n_fpfs1 > 10``, ``|e| < 0.4``, ``n_mask_base < 0.035``,
``trace = (m00 + m20)/m00 > 0.1``, i-band ``PSF FWHM < 0.8``
arcsec by default --
the cluster-test cuts).

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
from lsst.pex.config import DictField, Field
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
    "gmr_color": ("hsc_g_mag_gauss2 - hsc_r_mag_gauss2", -0.5, 2.5, 12, False),
    "rmi_color": ("hsc_r_mag_gauss2 - hsc_i_mag_gauss2", -0.5, 2.0, 12, False),
    "imz_color": ("hsc_i_mag_gauss2 - hsc_z_mag_gauss2", -0.5, 1.5, 12, False),
    "zmy_color": ("hsc_z_mag_gauss2 - hsc_y_mag_gauss2", -0.5, 1.0, 12, False),
    "n_mask_base": ("n_mask_base", 0.0, 0.035, 10, False),
    # Local background under the source, from the detection image. The
    # coadd mask planes miss a coherent background offset -- scattered
    # light, or a sky-subtraction failure in one visit -- but this does
    # not: on such a region the median runs several times the
    # surrounding value, while n_mask_base stays exactly 0 and the
    # coverage cuts do nothing. Log-spaced, since the distribution
    # spans decades and is positive through the bulk.
    # Focused on the bulk: 88% of sources sit below bkg = 1, and
    # the low tail reaches -0.6 at p0.1 (the true minimum, -24, is
    # a lone outlier). Deliberately NOT percentile-trimmed -- see
    # the fixed-range list in compute_bin_ranges.py.
    "bkg": ("bkg", -0.6, 1.0, 5, False),
    "discontinuity": ("n_mask_discontinuity", 0.0, 1.0, 10, False),

    "g_flux_err": ("hsc_g_flux_gauss2_err", 22.0, 32.0, 10, False),
    "r_flux_err": ("hsc_r_flux_gauss2_err", 37.0, 52.0, 10, False),
    "i_flux_err": ("hsc_i_flux_gauss2_err", 46.0, 82.0, 10, False),
    "z_flux_err": ("hsc_z_flux_gauss2_err", 95.0, 130.0, 10, False),
    "y_flux_err": ("hsc_y_flux_gauss2_err", 200.0, 275.0, 10, False),
    "g_mag": ("hsc_g_mag_gauss2", 21.0, 27.0, 12, False),
    "r_mag": ("hsc_r_mag_gauss2", 20.0, 25.5, 11, False),
    "z_mag": ("hsc_z_mag_gauss2", 19.5, 25.0, 11, False),
    "y_mag": ("hsc_y_mag_gauss2", 19.0, 25.5, 13, False),
    "trace": ("(fpfs1_m00 + fpfs1_m20) / fpfs1_m00", 0.1, 1.6, 12, False),
    "abs_e": ("np.sqrt(esq)", 0.0, 0.4, 10, False),
    # sky position: wide ranges covering all fields; bins outside the
    # footprint stay empty and are dropped by the plot's n_gal floor
    "ra": ("ra", 0.0, 360.0, 72, False),
    "dec": ("dec", -8.0, 46.0, 27, False),
}

# Per-band flux S/N (hsc_<b>_s2n_fpfs1). The selection cuts on the
# i-band S/N only, so the other bands still reach low values -- a shear
# trend against them is the test for noise-driven shape bias in the
# bands that were never cut on. Log-spaced like the i-band entry;
# ranges are superseded by --bin-trim when it is used.
_SNR_RANGE = {
    "g": (1.0, 100.0), "r": (1.0, 150.0), "i": (10.0, 320.0),
    "z": (1.0, 150.0), "y": (1.0, 100.0),
}
for _b in "grizy":
    _lo, _hi = _SNR_RANGE[_b]
    PROPERTY_BINS["%s_snr" % _b] = (
        "hsc_%s_s2n_fpfs1" % _b, _lo, _hi, 10, True,
    )

# Per-band coadd depth: the Gaussian-weighted mean of that band's
# nImage over the source footprint (hsc_<b>_n_inputs, written by the
# measurement when the visit-count maps are supplied). Ranges are the
# observed 1-99 percentiles over 24 tracts spanning all six fields;
# g,r are shallower than i,z,y, matching HSC's full-depth criterion of
# >= 4 visits in g,r and >= 6 in i,z,y. Sources with fewer inputs sit
# where the coadd's outlier rejection has least leverage, so a shear
# trend against this is the direct test for artefact contamination.
_N_INPUTS_RANGE = {
    "g": (2.0, 10.0), "r": (2.0, 7.0), "i": (3.0, 12.0),
    "z": (3.0, 10.0), "y": (3.0, 10.0),
}
for _b in "grizy":
    _lo, _hi = _N_INPUTS_RANGE[_b]
    PROPERTY_BINS["%s_n_inputs" % _b] = (
        "hsc_%s_n_inputs" % _b, _lo, _hi, int(_hi - _lo), False,
    )

# Per-band PSF properties from the HSM moment columns. FWHM [arcsec] =
# 2*sqrt(2 ln 2) * sigma [pixel] * 0.168; ranges from the observed
# 1-99 percentiles per band (i is tightened by the FWHM < 0.8 cut).
_PSF_FWHM_RANGE = {
    "g": (0.55, 1.15), "r": (0.45, 1.05), "i": (0.55, 0.8),
    "z": (0.45, 0.9), "y": (0.5, 0.9),
}
for _b in "grizy":
    _bxx = f"hsc_{_b}_ext_shapeHSM_HsmPsfMoments_xx"
    _byy = f"hsc_{_b}_ext_shapeHSM_HsmPsfMoments_yy"
    _bxy = f"hsc_{_b}_ext_shapeHSM_HsmPsfMoments_xy"
    PROPERTY_BINS[f"psf_e1_{_b}"] = (
        f"({_bxx} - {_byy}) / ({_bxx} + {_byy})", -0.1, 0.1, 12, False)
    PROPERTY_BINS[f"psf_e2_{_b}"] = (
        f"2.0 * {_bxy} / ({_bxx} + {_byy})", -0.1, 0.1, 12, False)
    _lo, _hi = _PSF_FWHM_RANGE[_b]
    PROPERTY_BINS[f"psf_fwhm_{_b}"] = (
        f"2.3548200 * 0.168 * np.sqrt(0.5 * ({_bxx} + {_byy}))",
        _lo, _hi, 12, False)

# Shear response per object, R = (R_1 + R_2)/2, the denominator of
# <gamma> = sum(w e) / sum(R). Spelled out here because both the 1-D
# histogram and the magnitude-response 2-D histogram need it.
_RESPONSE_EXPR = (
    "0.5 * (wsel * fpfs1_de1_dg1 + dwsel_dg1 * fpfs1_e1"
    " + wsel * fpfs1_de2_dg2 + dwsel_dg2 * fpfs1_e2)"
)

# FPFS resolution, (M00 + M20)/M00. The catalog carries no galaxy
# second moments, so this is the size-like quantity available: it is
# what the basic ``trace_min`` cut acts on, small for sources the PSF
# barely resolves and approaching 1 for well-resolved ones.
_RESOLUTION_EXPR = "(fpfs1_m00 + fpfs1_m20) / fpfs1_m00"

# 1-D sanity histograms. Ranges cover the observed p0.1-p99.9 over 8
# tracts spanning all six fields, so the tails are visible without the
# bulk being squeezed into two bins.
HIST_BINS = {
    "g_mag": ("hsc_g_mag_gauss2", 18.0, 27.0, 90, False),
    "r_mag": ("hsc_r_mag_gauss2", 18.0, 27.0, 90, False),
    "i_mag": ("hsc_i_mag_gauss2", 18.0, 27.0, 90, False),
    "z_mag": ("hsc_z_mag_gauss2", 18.0, 27.0, 90, False),
    "y_mag": ("hsc_y_mag_gauss2", 18.0, 27.0, 90, False),
    "abs_we1": ("np.abs(wsel * fpfs1_e1)", 0.0, 0.4, 80, False),
    "abs_we2": ("np.abs(wsel * fpfs1_e2)", 0.0, 0.4, 80, False),
    "response": (_RESPONSE_EXPR, -1.0, 3.0, 80, False),
    "resolution": (_RESOLUTION_EXPR, 0.0, 1.6, 80, False),
}

# Per-band S/N, flux error and PSF FWHM. Only the i-band S/N is cut on
# (s2n > 10), so the others keep their full range; the three quantities
# span very different values band to band, which is why S/N and the
# flux error are log-spaced -- on a linear axis g would collapse onto
# the origin while y ran off the end.
for _b in "grizy":
    HIST_BINS["%s_snr" % _b] = (
        "hsc_%s_s2n_fpfs1" % _b, 1.0, 3000.0, 72, True,
    )
    HIST_BINS["%s_flux_err" % _b] = (
        "hsc_%s_flux_gauss2_err" % _b, 15.0, 330.0, 80, True,
    )
    _bxx = f"hsc_{_b}_ext_shapeHSM_HsmPsfMoments_xx"
    _byy = f"hsc_{_b}_ext_shapeHSM_HsmPsfMoments_yy"
    HIST_BINS["%s_psf_fwhm" % _b] = (
        f"2.3548200 * 0.168 * np.sqrt(0.5 * ({_bxx} + {_byy}))",
        0.40, 1.20, 80, False,
    )
    # Fine bins on purpose: n_inputs is a comb of integer spikes with a
    # continuum between them (sources straddling a coverage step), and
    # a per-visit binning hides that structure.
    HIST_BINS["%s_n_inputs" % _b] = (
        "hsc_%s_n_inputs" % _b, 0.0, 14.0, 70, False,
    )

# 2-D histograms: name -> (x axis, y axis). The four below mirror the
# corner plots of the S23B shape-catalog notebook (4_test_histogram):
# i magnitude against colour (twice), against the shear response, and
# against size. The x range stops at the i < 25 basic cut rather than
# the notebook's 26, since nothing survives above it.
HIST2D_BINS = {
    "cmd_i_rmi": (
        ("hsc_i_mag_gauss2", 20.0, 25.0, 60, False),
        ("hsc_r_mag_gauss2 - hsc_i_mag_gauss2", -0.2, 1.3, 60, False),
    ),
    "cmd_i_imz": (
        ("hsc_i_mag_gauss2", 20.0, 25.0, 60, False),
        ("hsc_i_mag_gauss2 - hsc_z_mag_gauss2", -0.5, 1.0, 60, False),
    ),
    "mag_response": (
        ("hsc_i_mag_gauss2", 20.0, 25.0, 60, False),
        (_RESPONSE_EXPR, -0.5, 1.2, 60, False),
    ),
    "mag_resolution": (
        ("hsc_i_mag_gauss2", 20.0, 25.0, 60, False),
        (_RESOLUTION_EXPR, 0.1, 1.5, 60, False),
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


def _axis_edges(axis, span=None) -> tuple[str, NDArray]:
    """Registry axis tuple -> (expr, bin edges).

    ``span`` replaces the registry's hand-picked ``(lo, hi)`` with a
    data-derived pair -- the ``[trim, 100-trim]`` percentiles from
    :func:`percentile_span`.  The bin COUNT and spacing are unchanged:
    the bins stay uniform (or log), only the range moves, so the
    figures keep evenly spaced bins rather than equal-count ones.
    """
    expr, lo, hi, nbins, log = axis
    if span is not None:
        lo, hi = float(span[0]), float(span[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return expr, np.array([], dtype=np.float64)
    if log and lo > 0:
        return expr, np.geomspace(lo, hi, nbins + 1)
    # A percentile-trimmed range can reach <= 0 on a quantity that is
    # mostly positive (bkg goes slightly negative where the sky is
    # over-subtracted). Fall back to linear rather than failing.
    if log:
        return expr, np.linspace(lo, hi, nbins + 1)
    return expr, np.linspace(lo, hi, nbins + 1)


def percentile_span(values: NDArray, trim: float) -> tuple[float, float]:
    """``(lo, hi)`` at the ``[trim, 100-trim]`` percentiles.

    The registry's fixed ranges were hand-picked and are stale as soon
    as the selection changes: bins fall outside the data (empty, then
    dropped by the plot) or clip a real tail. Trimming by percentile
    puts the axis where the sources actually are.  ``trim`` is in
    PERCENT, so 2.0 means 2%-98%.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(v, [trim, 100.0 - trim])
    return (float(lo), float(hi))


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
    n_mask_base_max = Field[float](
        doc=(
            "Basic selection: keep n_mask_base < n_mask_base_max, where "
            "n_mask_base is the Gaussian-weighted masked FRACTION in "
            "[0, 1]."
        ),
        default=0.035,
    )
    trace_min = Field[float](
        doc=(
            "Basic selection: keep trace = (fpfs1_m00 + fpfs1_m20) / "
            "fpfs1_m00 > trace_min (removes poorly resolved sources)."
        ),
        default=0.1,
    )
    mag_max_bands = DictField(
        keytype=str, itemtype=float, default={},
        doc=(
            "Per-band upper limit on {survey}_{band}_mag_gauss2, e.g. "
            "{'g': 28.0, 'r': 27.0}. Empty means no per-band cut.\n\n"
            "This is a SANITY cut, not a depth cut: flux_to_mag "
            "saturates at 40 for a non-detection or negative flux, and "
            "those sentinels reach the 99.9th percentile in g and y. "
            "Left in, they land in the extreme colour bins and dominate "
            "them. The i band is already covered by mag_max."
        ),
    )
    flux_err_min_bands = DictField(
        keytype=str, itemtype=float, default={},
        doc=(
            "Per-band LOWER limit on {survey}_{band}_flux_gauss2_err, "
            "e.g. {'g': 25.0}. Empty means no lower limit.\n\n"
            "flux_gauss2_err is a fixed-aperture flux error, so it is "
            "set by the cell's noise and PSF rather than by the source "
            "-- within a patch it varies by only a few percent, while "
            "across the survey it spans a factor of two. Cutting on it "
            "therefore selects on DEPTH and seeing, not on the galaxy."
        ),
    )
    flux_err_max_bands = DictField(
        keytype=str, itemtype=float, default={},
        doc=(
            "Per-band UPPER limit on {survey}_{band}_flux_gauss2_err. "
            "See flux_err_min_bands."
        ),
    )
    psf_fwhm_min_bands = DictField(
        keytype=str, itemtype=float, default={},
        doc=(
            "Per-band LOWER limit on the PSF FWHM [arcsec], e.g. "
            "{'r': 0.48, 'i': 0.48}. Empty means no lower limit. The "
            "FWHM comes from the ext_shapeHSM PSF moments of that band; "
            "a band whose columns are absent is skipped with a warning "
            "rather than dropping every source."
        ),
    )
    psf_fwhm_max_bands = DictField(
        keytype=str, itemtype=float, default={},
        doc=(
            "Per-band UPPER limit on the PSF FWHM [arcsec], e.g. "
            "{'r': 1.1, 'i': 0.8, 'z': 1.1}. Empty means no upper "
            "limit. Independent of the single-band psf_fwhm_max, which "
            "cuts only on i -- set that to 0 when using this."
        ),
    )
    n_inputs_min_bands = DictField(
        keytype=str, itemtype=float, default={},
        doc=(
            "Per-band floor on {survey}_{band}_n_inputs, e.g. "
            "{'g': 3.5, 'i': 4.5}. A source is kept when EVERY listed "
            "band is STRICTLY ABOVE its threshold; bands not listed "
            "are not cut on. Put thresholds BETWEEN integers: the "
            "distribution is a comb of integer spikes (uniform-coverage "
            "interiors) and a floor just above an integer deletes a "
            "whole spike. "
            "Empty disables the cut.\n\n"
            "Per band because the survey is not equally deep in all of "
            "them -- HSC's full-depth criterion is >= 4 visits in g,r "
            "and >= 6 in i,z,y -- so one global number either wastes "
            "the deep bands or guts the shallow ones. n_inputs is a "
            "Gaussian-weighted mean over the source footprint, so it "
            "is continuous, not an integer count."
        ),
    )
    bin_percentile_trim = Field[float](
        doc=(
            "Percentile trim defining each property's bin RANGE: the "
            "axis spans [trim, 100-trim] percent of the selected "
            "sample, e.g. 2.0 gives 2%-98%. Bins stay evenly spaced "
            "inside that range -- this moves the limits, it does not "
            "make equal-count bins. 0 keeps the hand-picked ranges in "
            "PROPERTY_BINS.\n\n"
            "The percentiles CANNOT be evaluated per tract: the "
            "per-tract sums are added across tracts, so every tract "
            "must bin identically. They are measured once over a "
            "sample of tracts and passed in via bin_ranges_file; this "
            "field records which trim produced them."
        ),
        default=0.0,
    )
    bin_ranges_file = Field[str](
        doc=(
            "FITS table of shared per-property bin ranges (columns: "
            "property, lo, hi) from the HSC compute_bin_ranges.py. "
            "Empty means use the ranges in PROPERTY_BINS."
        ),
        default="",
    )
    psf_fwhm_max = Field[float](
        doc=(
            "Basic selection: keep i-band PSF FWHM < psf_fwhm_max "
            "[arcsec] (from the ext_shapeHSM moment columns; skipped "
            "with a warning when the catalog does not carry them; "
            "<= 0 disables)."
        ),
        default=0.8,
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

    def _bin_ranges(self) -> dict:
        """``{property: (lo, hi)}`` from ``config.bin_ranges_file``."""
        assert isinstance(self.config, ShearStatsPipeConfig)
        cached = getattr(self, "_bin_ranges_cache", None)
        if cached is not None:
            return cached
        out: dict = {}
        path = self.config.bin_ranges_file
        if path:
            from astropy.table import Table

            tab = Table.read(path)
            for row in tab:
                out[str(row["property"])] = (float(row["lo"]),
                                             float(row["hi"]))
            self.log.info(
                "bin ranges for %d properties from %s (trim %.3g%%)",
                len(out), path, self.config.bin_percentile_trim,
            )
        self._bin_ranges_cache = out
        return out

    def _select(self, cat: NDArray) -> NDArray:
        """The basic source selection (cluster-test cuts)."""
        assert isinstance(self.config, ShearStatsPipeConfig)
        mag = np.asarray(cat["hsc_i_mag_gauss2"], dtype=np.float64)
        snr = np.asarray(cat["hsc_i_s2n_fpfs1"], dtype=np.float64)
        esq = np.asarray(cat["esq"], dtype=np.float64)
        mval = np.asarray(cat["n_mask_base"], dtype=np.float64)
        m00 = np.asarray(cat["fpfs1_m00"], dtype=np.float64)
        m20 = np.asarray(cat["fpfs1_m20"], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            trace = (m00 + m20) / m00
        sel = (
            np.isfinite(mag)
            & (mag < self.config.mag_max)
            & (snr > self.config.snr_min)
            & (esq < self.config.esq_max)
            & (mval < self.config.n_mask_base_max)
            & (trace > self.config.trace_min)
        )
        if self.config.psf_fwhm_max > 0:
            if _XX in cat.dtype.names and _YY in cat.dtype.names:
                fwhm = 2.3548200 * 0.168 * np.sqrt(0.5 * (
                    np.asarray(cat[_XX], dtype=np.float64)
                    + np.asarray(cat[_YY], dtype=np.float64)
                ))
                sel &= fwhm < self.config.psf_fwhm_max
            else:
                self.log.warning(
                    "PSF FWHM cut skipped: no %s in the catalog", _XX
                )

        # Per-band magnitude sanity cut (see mag_max_bands).
        for band, hi in sorted(self.config.mag_max_bands.items()):
            col = "hsc_%s_mag_gauss2" % band
            if col not in (cat.dtype.names or ()):
                self.log.warning(
                    "magnitude cut skipped for band %s: no %s", band, col
                )
                continue
            m = np.asarray(cat[col], dtype=np.float64)
            sel &= np.isfinite(m) & (m < hi)

        # Per-band flux-error window (a depth/seeing selection).
        for band in sorted(
            set(self.config.flux_err_min_bands)
            | set(self.config.flux_err_max_bands)
        ):
            col = "hsc_%s_flux_gauss2_err" % band
            if col not in (cat.dtype.names or ()):
                self.log.warning(
                    "flux_err cut skipped for band %s: no %s", band, col
                )
                continue
            fe = np.asarray(cat[col], dtype=np.float64)
            lo = self.config.flux_err_min_bands.get(band)
            hi = self.config.flux_err_max_bands.get(band)
            sel &= np.isfinite(fe)
            if lo is not None:
                sel &= fe > lo
            if hi is not None:
                sel &= fe < hi

        # Per-band PSF FWHM window. A band with no HSM columns is
        # skipped loudly instead of silently emptying the selection.
        names = cat.dtype.names or ()
        for band in sorted(
            set(self.config.psf_fwhm_min_bands) | set(self.config.psf_fwhm_max_bands)
        ):
            bxx = "hsc_%s_ext_shapeHSM_HsmPsfMoments_xx" % band
            byy = "hsc_%s_ext_shapeHSM_HsmPsfMoments_yy" % band
            if bxx not in names or byy not in names:
                self.log.warning(
                    "PSF FWHM cut skipped for band %s: no %s", band, bxx
                )
                continue
            fwhm = 2.3548200 * 0.168 * np.sqrt(0.5 * (
                np.asarray(cat[bxx], dtype=np.float64)
                + np.asarray(cat[byy], dtype=np.float64)
            ))
            lo = self.config.psf_fwhm_min_bands.get(band)
            hi = self.config.psf_fwhm_max_bands.get(band)
            if lo is not None:
                sel &= fwhm > lo
            if hi is not None:
                sel &= fwhm < hi

        # Coadd-depth floor, per band (inclusive).
        for band, lo in sorted(self.config.n_inputs_min_bands.items()):
            col = "hsc_%s_n_inputs" % band
            if col not in names:
                self.log.warning(
                    "n_inputs cut skipped for band %s: no %s", band, col
                )
                continue
            sel &= np.asarray(cat[col], dtype=np.float64) > lo
        return sel

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
        spans = self._bin_ranges()
        for name, axis in PROPERTY_BINS.items():
            expr, edges = _axis_edges(axis, spans.get(name))
            if len(edges) < 2:
                continue
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
