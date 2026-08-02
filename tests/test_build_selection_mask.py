"""Unit tests for ``xlens.catalog.base.build_selection_mask``.

The function was extracted from ``ShearEstimator._measure`` to expose
the (mag, size, shape, optional photo-z) selection logic as a public
helper. These tests pin down:

* equivalence with the original inline logic, across the dg
  perturbations used for selection-bias accounting;
* parity for the ``shape_name`` input regardless of whether the caller
  passes ``"fpfs"`` or the trailing-underscore form ``"fpfs_"`` that
  ``_measure`` itself supplies;
* parity with extinction subtracted, since callers in DP1 work do pass
  per-band ``a_{b}`` corrections;
* parity when ``mag_max`` is a per-band dict vs a scalar (only at the
  same scalar threshold) — the dict path is the production use.
"""
import numpy as np

from xlens.catalog.base import (
    build_selection_mask,
    get_esq,
    get_trace,
)
from xlens.catalog.utils import _resolve_cut, _resolve_cut_name

SN = "fpfs_"
SHAPE = "fpfs"
BANDS = ["lsst_g", "lsst_r", "lsst_i", "lsst_z"]
FLUX = "fpfs1"
MAG_ZERO = 31.4
EMAX = 0.3
TRACE_MIN = 0.05
MAG_MAX = {"lsst_g": 25.5, "lsst_r": 25.0, "lsst_i": 23.5, "lsst_z": 24.5}


def _inline_mask(src, *, comp, dg, extinction=None,
                 bands=BANDS, mag_max=MAG_MAX, mag_zero=MAG_ZERO,
                 flux_name=FLUX, sn=SN, emax=EMAX, trace_min=TRACE_MIN):
    """Verbatim pre-refactor mask logic (used as the reference)."""
    fn = _resolve_cut_name(flux_name)
    magx = _resolve_cut(mag_max, bands=bands)
    emax2 = emax * emax
    esq_s = get_esq(src, comp=comp, dg=dg, sn=sn)
    trace_s = get_trace(src, comp=comp, dg=dg, sn=sn)
    mask_s = (esq_s < emax2) & (trace_s > trace_min)
    for b in bands:
        flux_b = src[f"{b}_flux{fn}"] + dg * src[f"{b}_dflux{fn}_dg{comp}"]
        mag_b = np.full(len(src), 40.0, dtype=np.float64)
        pos = flux_b > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mag_b[pos] = mag_zero - 2.5 * np.log10(flux_b[pos])
        if extinction is not None:
            mag_b = mag_b - extinction[f"a_{b}"]
        mask_s &= mag_b < magx[b]
    return mask_s


def _make_synth(n=10_000, seed=12345):
    """Synthetic structured array with every column the helper reads."""
    rng = np.random.default_rng(seed)
    dtype = [
        (f"{SN}e1", "f8"), (f"{SN}e2", "f8"),
        (f"{SN}de1_dg1", "f8"), (f"{SN}de2_dg1", "f8"),
        (f"{SN}de1_dg2", "f8"), (f"{SN}de2_dg2", "f8"),
        (f"{SN}m0", "f8"), (f"{SN}m2", "f8"),
        (f"{SN}dm0_dg1", "f8"), (f"{SN}dm0_dg2", "f8"),
        (f"{SN}dm2_dg1", "f8"), (f"{SN}dm2_dg2", "f8"),
    ]
    for b in BANDS:
        dtype += [(f"{b}_flux_{FLUX}", "f8"),
                  (f"{b}_dflux_{FLUX}_dg1", "f8"),
                  (f"{b}_dflux_{FLUX}_dg2", "f8")]
    src = np.zeros(n, dtype=dtype)
    src[f"{SN}e1"] = rng.normal(0, 0.15, n)
    src[f"{SN}e2"] = rng.normal(0, 0.15, n)
    for c in (1, 2):
        for k in (1, 2):
            src[f"{SN}de{k}_dg{c}"] = rng.normal(0, 0.1, n)
    src[f"{SN}m0"] = rng.uniform(1.0, 1500.0, n)
    size_frac = rng.uniform(0.02, 0.5, n)
    src[f"{SN}m2"] = size_frac * src[f"{SN}m0"]
    for k in (1, 2):
        src[f"{SN}dm0_dg{k}"] = rng.normal(0, 50, n)
        src[f"{SN}dm2_dg{k}"] = rng.normal(0, 20, n)
    for b in BANDS:
        # Spans roughly mag 21.4 .. 28 with MAG_ZERO=31.4.
        src[f"{b}_flux_{FLUX}"] = 10 ** rng.uniform(0.7, 3.4, n)
        src[f"{b}_dflux_{FLUX}_dg1"] = rng.normal(0, 1.5, n)
        src[f"{b}_dflux_{FLUX}_dg2"] = rng.normal(0, 1.5, n)
    ext_dtype = [(f"a_{b}", "f8") for b in BANDS]
    extinction = np.zeros(n, dtype=ext_dtype)
    for b in BANDS:
        extinction[f"a_{b}"] = rng.uniform(0.0, 0.3, n)
    return src, extinction


def test_build_selection_mask_matches_inline_logic():
    """For every (comp, dg, extinction) combo, the helper must return
    the same boolean mask as the original inline loop."""
    src, extinction = _make_synth()
    for comp in (1, 2):
        for dg in (0.0, +0.02, -0.02):
            for ext in (None, extinction):
                m_ref = _inline_mask(src, comp=comp, dg=dg, extinction=ext)
                m, z = build_selection_mask(
                    src, comp=comp, dg=dg, shape_name=SHAPE,
                    bands=BANDS, mag_max=MAG_MAX, mag_zero=MAG_ZERO,
                    flux_name=FLUX, emax=EMAX, trace_min=TRACE_MIN,
                    extinction=ext,
                )
                assert z is None, "no z_estimator was given"
                assert np.array_equal(m, m_ref), (
                    f"mask mismatch at comp={comp} dg={dg} ext={ext is not None}: "
                    f"new.sum={m.sum()} ref.sum={m_ref.sum()}"
                )
                # Sanity: the mask actually selects something but not the
                # whole catalog — so the test exercises the cuts.
                assert 0 < m.sum() < len(src), (
                    f"trivial mask at comp={comp} dg={dg}: kept {m.sum()}"
                )


def test_build_selection_mask_shape_name_with_underscore():
    """``shape_name='fpfs'`` and ``shape_name='fpfs_'`` should produce
    the same mask — ShearEstimator._measure passes the underscore form
    (``self.sn``); user code passes the bare name."""
    src, _ = _make_synth(n=1000, seed=42)
    args = dict(
        src=src, comp=1, dg=0.0,
        bands=BANDS, mag_max=MAG_MAX, mag_zero=MAG_ZERO,
        flux_name=FLUX, emax=EMAX, trace_min=TRACE_MIN,
    )
    m_bare, _ = build_selection_mask(**args, shape_name="fpfs")
    m_us, _ = build_selection_mask(**args, shape_name="fpfs_")
    assert np.array_equal(m_bare, m_us)


def test_build_selection_mask_dg_perturbation_changes_mask():
    """The dg perturbation should move at least one source across one
    of the cuts (mag or shape) — if not, the test data isn't exercising
    the dg-aware code path and the equivalence test above is vacuous."""
    src, _ = _make_synth()
    m0, _ = build_selection_mask(
        src, comp=1, dg=0.0, shape_name=SHAPE,
        bands=BANDS, mag_max=MAG_MAX, mag_zero=MAG_ZERO,
        flux_name=FLUX, emax=EMAX, trace_min=TRACE_MIN,
    )
    m_plus, _ = build_selection_mask(
        src, comp=1, dg=+0.02, shape_name=SHAPE,
        bands=BANDS, mag_max=MAG_MAX, mag_zero=MAG_ZERO,
        flux_name=FLUX, emax=EMAX, trace_min=TRACE_MIN,
    )
    m_minus, _ = build_selection_mask(
        src, comp=1, dg=-0.02, shape_name=SHAPE,
        bands=BANDS, mag_max=MAG_MAX, mag_zero=MAG_ZERO,
        flux_name=FLUX, emax=EMAX, trace_min=TRACE_MIN,
    )
    assert (m_plus ^ m0).any() or (m_minus ^ m0).any(), (
        "dg perturbation did not change the mask anywhere"
    )


def test_build_selection_mask_returns_z_none_when_no_estimator():
    src, _ = _make_synth(n=200)
    _, z = build_selection_mask(
        src, comp=1, dg=0.0, shape_name=SHAPE,
        bands=BANDS, mag_max=MAG_MAX, mag_zero=MAG_ZERO,
        flux_name=FLUX, emax=EMAX, trace_min=TRACE_MIN,
    )
    assert z is None


def test_build_selection_mask_scalar_mag_max_equals_dict_form():
    """Passing mag_max=24.0 must give the same mask as passing
    {band: 24.0 for band in bands}."""
    src, _ = _make_synth(n=500, seed=7)
    m_scalar, _ = build_selection_mask(
        src, comp=1, dg=0.0, shape_name=SHAPE,
        bands=BANDS, mag_max=24.0, mag_zero=MAG_ZERO,
        flux_name=FLUX, emax=EMAX, trace_min=TRACE_MIN,
    )
    m_dict, _ = build_selection_mask(
        src, comp=1, dg=0.0, shape_name=SHAPE,
        bands=BANDS, mag_max={b: 24.0 for b in BANDS}, mag_zero=MAG_ZERO,
        flux_name=FLUX, emax=EMAX, trace_min=TRACE_MIN,
    )
    assert np.array_equal(m_scalar, m_dict)
