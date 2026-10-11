import os
import pickle
from pathlib import Path

import fitsio
import numpy as np
from numpy.lib import recfunctions as rfn

import xlens
from xlens.utils.catalog import ShearEstimator, measure_shear

# Survey-prefixed bands the pipeline now uses; the bundled
# ``catalog.fits`` has single-letter per-band columns, so re-key them to
# ``lsst_<band>_...`` on load.
PREFIXED_BANDS = [f"lsst_{b}" for b in "ugrizy"]
REF_BAND = "lsst_i"


def _read_prefixed(fname):
    catalog = fitsio.read(fname)
    rename = {
        n: f"lsst_{n}"
        for n in catalog.dtype.names
        if len(n) >= 2 and n[0] in "ugrizy" and n[1] == "_"
    }
    return np.asarray(rfn.rename_fields(catalog, rename)) if rename else catalog


def test_pz_point_estimates():
    # Set up the configuration
    DATA_DIR = Path(__file__).parent / "data"
    pfname = os.path.join(DATA_DIR, "pz_pdfs_test.fits")
    pdfs = fitsio.read(pfname)
    out = xlens.utils.catalog.redshift.get_point_estimates_from_pdfs(pdfs)
    key_target = ['zmode', 'z025', 'z160', 'z500', 'z840', 'z975', 'zbest']
    assert list(out.keys()) == key_target
    zbest_target = np.array([4.28967696, 0.72257506, 1.52362052])
    np.testing.assert_almost_equal(
        out["zbest"],
        zbest_target,
    )
    zmode_target = np.array([0.34, 0.73, 1.53])
    np.testing.assert_almost_equal(
        out["zmode"],
        zmode_target,
    )
    return


def test_pz():
    # Set up the configuration
    DATA_DIR = Path(__file__).parent / "data"
    fname = os.path.join(DATA_DIR, "catalog.fits")
    catalog = _read_prefixed(fname)
    model_fname = os.path.join(DATA_DIR, "model_inform_fzboost.pkl")
    with open(model_fname, "rb") as f:
        pz_obj = pickle.load(f)
    fzbobj = xlens.utils.catalog.redshift.flexzboostEstimator(pz_obj)
    out = fzbobj.get_z(
        catalog,
        mag_zero=30.0,
        flux_name="gauss2",
        bands=PREFIXED_BANDS,
        ref_band=REF_BAND,
    )
    key_target = ['zmode', 'z025', 'z160', 'z500', 'z840', 'z975', 'zbest']
    assert list(out.keys()) == key_target
    zbest_target = np.array([4.28967696, 0.72257506, 1.52362052])
    np.testing.assert_almost_equal(
        out["zbest"],
        zbest_target,
    )
    zmode_target = np.array([0.34, 0.73, 1.53])
    np.testing.assert_almost_equal(
        out["zmode"],
        zmode_target,
    )
    return


def test_measure_shear_consistency():
    DATA_DIR = Path(__file__).parent / "data"
    fname = os.path.join(DATA_DIR, "catalog.fits")
    catalog = _read_prefixed(fname)
    model_fname = os.path.join(DATA_DIR, "model_inform_fzboost.pkl")
    with open(model_fname, "rb") as f:
        pz_obj = pickle.load(f)
    fzbobj = xlens.utils.catalog.redshift.flexzboostEstimator(pz_obj)

    emax = 0.3
    zbounds = [0.3, 0.6, 0.9, 1.2]
    dg = 0.02
    target = "g1g2"
    do_correction = True
    z_width95_max = 2.75
    mag_zero = 30.0
    bands = PREFIXED_BANDS
    ref_band = REF_BAND
    trace_min = 0.05

    mag_max_list = [27.5, 27.0, 26.0, 24.6, 25.0, 25.5]
    flux_min = {
        b: 10.0 ** ((mag_zero - mmax) / 2.5)
        for b, mmax in zip(bands, mag_max_list)
    }
    mag_max = {b: mmax for b, mmax in zip(bands, mag_max_list)}

    shear_kwargs = {
        "flux_min": flux_min,
        "z_estimator": fzbobj,
        "emax": emax,
        "trace_min": trace_min,
        "zbounds": zbounds,
        "dg": dg,
        "target": target,
        "do_correction": do_correction,
        "z_width95_max": z_width95_max,
        "mag_zero": mag_zero,
        "flux_name": "gauss2",
        "bands": bands,
        "ref_band": ref_band,
    }
    out1 = measure_shear(
        src=catalog,
        **shear_kwargs,
    )
    shear_obj = ShearEstimator(
        mag_max=mag_max,
        emax=emax,
        trace_min=trace_min,
        mag_zero=mag_zero,
        flux_name="gauss2",
        bands=bands,
        ref_band=ref_band,
        z_estimator=fzbobj,
        zbounds=zbounds,
        z_width95_max=z_width95_max,
        dg=dg,
    )
    out2 = shear_obj.measure_shear(catalog, target=target)

    np.testing.assert_almost_equal(
        out1["e1"],
        out2["e1"],
    )
    np.testing.assert_almost_equal(
        out1["e2"],
        out2["e2"],
    )
    np.testing.assert_almost_equal(
        out1["r1"],
        out2["r1"] + out2["r1_det"],
    )
    np.testing.assert_almost_equal(
        out1["r2"],
        out2["r2"] + out2["r2_det"],
    )
    np.testing.assert_almost_equal(
        out1["r1_sel"],
        out2["r1_sel"],
    )
    np.testing.assert_almost_equal(
        out1["r2_sel"],
        out2["r2_sel"],
    )
    return


def _load_fzboost():
    DATA_DIR = Path(__file__).parent / "data"
    with open(os.path.join(DATA_DIR, "model_inform_fzboost.pkl"), "rb") as f:
        return xlens.utils.catalog.redshift.flexzboostEstimator(pickle.load(f))


def _catalog():
    return _read_prefixed(os.path.join(Path(__file__).parent / "data", "catalog.fits"))


_UNBINNED_KW = dict(
    flux_min=10.0 ** ((30.0 - 25.5) / 2.5),
    emax=0.3,
    trace_min=0.05,
    dg=0.02,
    mag_zero=30.0,
    flux_name="gauss2",
    bands=PREFIXED_BANDS,
    ref_band=REF_BAND,
)


def test_measure_shear_unbinned_equals_one_wide_bin():
    """z_estimator=None must give the same numbers as a single bin that
    contains every source, for every target, without touching a photo-z."""
    catalog = _catalog()
    fzb = _load_fzboost()
    for target in ("g1", "g2", "g1g2"):
        out0 = measure_shear(src=catalog, target=target, **_UNBINNED_KW)
        out1 = measure_shear(
            src=catalog, target=target, z_estimator=fzb,
            zbounds=[-1.0, 1.0e9], z_width95_max=np.inf, **_UNBINNED_KW,
        )
        assert set(out0) == set(out1)
        for k in out0:
            assert out0[k].shape == (1,), (k, out0[k].shape)
            np.testing.assert_allclose(out0[k], out1[k], rtol=1e-12, atol=0.0)
    assert out0["n_gal"][0] > 0


def test_measure_shear_binning_arguments_are_validated():
    class Boom:
        def get_zsel(self, *a, **k):
            raise AssertionError("estimator must not be consulted when unbinned")

    catalog = _catalog()
    measure_shear(src=catalog, target="g1", **_UNBINNED_KW)
    for bad in (dict(zbounds=[0.0, 1.0]),                 # bounds, no estimator
                dict(z_estimator=Boom()),                 # estimator, no bounds
                dict(z_estimator=Boom(), zbounds=[0.5])): # a single edge
        try:
            measure_shear(src=catalog, target="g1", **bad, **_UNBINNED_KW)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} must raise ValueError")


def _make_synth_catalog(n=4000, seed=7, bands=PREFIXED_BANDS, flux_name="gauss2"):
    """Synthetic catalog with every column measure_shear reads.

    The bundled ``catalog.fits`` has three rows, which is enough to pin the
    photo-z path against a reference but not to test a cut statistically.
    """
    rng = np.random.default_rng(seed)
    dtype = [
        ("wsel", "f8"), ("dwsel_dg1", "f8"), ("dwsel_dg2", "f8"),
        ("fpfs_e1", "f8"), ("fpfs_e2", "f8"),
        ("fpfs_de1_dg1", "f8"), ("fpfs_de1_dg2", "f8"),
        ("fpfs_de2_dg1", "f8"), ("fpfs_de2_dg2", "f8"),
        ("fpfs_m0", "f8"), ("fpfs_m2", "f8"),
        ("fpfs_dm0_dg1", "f8"), ("fpfs_dm0_dg2", "f8"),
        ("fpfs_dm2_dg1", "f8"), ("fpfs_dm2_dg2", "f8"),
    ]
    for b in bands:
        dtype += [(f"{b}_flux_{flux_name}", "f8"),
                  (f"{b}_dflux_{flux_name}_dg1", "f8"),
                  (f"{b}_dflux_{flux_name}_dg2", "f8")]
    src = np.zeros(n, dtype=dtype)
    src["wsel"] = rng.uniform(0.1, 1.0, n)
    src["dwsel_dg1"] = rng.normal(0, 0.1, n)
    src["dwsel_dg2"] = rng.normal(0, 0.1, n)
    src["fpfs_e1"] = rng.normal(0, 0.15, n)
    src["fpfs_e2"] = rng.normal(0, 0.15, n)
    for c in (1, 2):
        for k in (1, 2):
            src[f"fpfs_de{k}_dg{c}"] = rng.normal(0, 0.1, n)
        src[f"fpfs_dm0_dg{c}"] = rng.normal(0, 50, n)
        src[f"fpfs_dm2_dg{c}"] = rng.normal(0, 20, n)
    src["fpfs_m0"] = rng.uniform(1.0, 1500.0, n)
    src["fpfs_m2"] = rng.uniform(0.02, 0.5, n) * src["fpfs_m0"]
    for b in bands:
        src[f"{b}_flux_{flux_name}"] = 10 ** rng.uniform(0.7, 3.4, n)
        src[f"{b}_dflux_{flux_name}_dg1"] = rng.normal(0, 1.5, n)
        src[f"{b}_dflux_{flux_name}_dg2"] = rng.normal(0, 1.5, n)
    return src


def test_measure_shear_trace_min_none_skips_the_cut():
    """None is 'no size cut'.  It must (a) equal -inf when every trace is
    finite, (b) keep sources with a non-positive monopole that 0.0 drops,
    and (c) give way to a real threshold exactly as an explicit mask says."""
    from xlens.utils.catalog.base import get_esq, get_trace

    src = _make_synth_catalog()
    # sources with a non-positive monopole that pass every other cut
    bad = slice(0, 25)
    src["fpfs_m0"][bad] = -1.0
    src["fpfs_e1"][bad] = 0.0
    src["fpfs_e2"][bad] = 0.0
    for b in PREFIXED_BANDS:
        src[f"{b}_flux_gauss2"][bad] = 1.0e3
    kw = dict(_UNBINNED_KW)
    kw.pop("trace_min")
    kw["flux_min"] = 10.0

    sel = get_esq(src) < kw["emax"] ** 2
    for b in PREFIXED_BANDS:
        sel &= src[f"{b}_flux_gauss2"] > kw["flux_min"]
    trace = get_trace(src)
    assert np.all(np.isfinite(trace)) and sel[bad].all()

    out_none = measure_shear(src=src, target="g1", trace_min=None, **kw)
    out_inf = measure_shear(src=src, target="g1", trace_min=-np.inf, **kw)
    assert out_none["n_gal"][0] == np.count_nonzero(sel)
    for k in out_none:
        np.testing.assert_allclose(out_none[k], out_inf[k], rtol=1e-12)

    out_zero = measure_shear(src=src, target="g1", trace_min=0.0, **kw)
    assert out_zero["n_gal"][0] == np.count_nonzero(sel & (trace > 0.0))
    assert out_none["n_gal"][0] - out_zero["n_gal"][0] == 25

    tm = float(np.median(trace[sel]))
    out_cut = measure_shear(src=src, target="g1", trace_min=tm, **kw)
    assert out_cut["n_gal"][0] == np.count_nonzero(sel & (trace > tm))
    assert 0 < out_cut["n_gal"][0] < out_none["n_gal"][0]


def test_shear_estimator_unbinned_matches_measure_shear():
    """ShearEstimator without a z_estimator used to crash in np.bincount;
    it must now agree with the module function on the same cuts."""
    catalog = _catalog()
    mag_zero, mag_max = 30.0, 25.5
    flux_min = 10.0 ** ((mag_zero - mag_max) / 2.5)
    out1 = measure_shear(
        src=catalog, target="g1g2", flux_min=flux_min, emax=0.3,
        trace_min=0.05, dg=0.02, mag_zero=mag_zero, flux_name="gauss2",
        bands=PREFIXED_BANDS, ref_band=REF_BAND,
    )
    est = ShearEstimator(
        mag_max=mag_max, emax=0.3, trace_min=0.05, mag_zero=mag_zero,
        flux_name="gauss2", bands=PREFIXED_BANDS, ref_band=REF_BAND, dg=0.02,
    )
    out2 = est.measure_shear(catalog, target="g1g2")
    for a, b in (("e1", "e1"), ("r1", "r1"), ("r1_sel", "r1_sel"),
                 ("e2", "e2"), ("r2", "r2"), ("r2_sel", "r2_sel"),
                 ("n_gal", "num")):
        assert out1[a].shape == (1,) and out2[b].shape == (1,), (a, b)
        np.testing.assert_allclose(out1[a], out2[b], rtol=1e-10, atol=1e-12)
    try:
        ShearEstimator(mag_max=mag_max, zbounds=[0.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("ShearEstimator: zbounds without z_estimator must raise")
