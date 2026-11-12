"""The optimal-shape-weight fitter: its full response must equal the
finite-difference shear response of the weighted, selected sample, and
the fit must beat the unweighted estimator when the noise depends on S/N."""
import numpy as np
import pytest
from astropy.table import Table

from xlens.utils.catalog.shape_weight_fit import (
    Cut,
    _response_name,
    fit_shape_weight,
    prepare_shape_weight_sample,
    shape_weight_fom,
)


def _catalog(n=200000, seed=3, g1=0.0, g2=0.0, noise_amp=6.0):
    """Shapes e = e_int + noise(S/N) + g with unit response; S/N responds
    to both shear components alike (ds2n/dg_j = 0.5 S/N_0), so the cut and
    the weight both select and R11 = R22."""
    rng = np.random.default_rng(seed)
    snr0 = 10 ** rng.uniform(np.log10(3.0), 2.0, n)
    snr = snr0 * (1.0 + 0.5 * (g1 + g2))
    e_int = rng.normal(0.0, 0.25, (2, n))
    noise = rng.normal(0.0, 1.0, (2, n)) * (noise_amp / snr0)
    e1 = e_int[0] + noise[0] + g1
    e2 = e_int[1] + noise[1] + g2
    t = Table()
    t["fpfs1_e1"], t["fpfs1_e2"] = e1, e2
    t["fpfs1_de1_dg1"] = np.ones(n)
    t["fpfs1_de2_dg2"] = np.ones(n)
    t["fpfs1_s2n"] = snr
    t["fpfs1_ds2n_dg1"] = 0.5 * snr0
    t["fpfs1_ds2n_dg2"] = 0.5 * snr0
    t["esq"] = e1**2 + e2**2
    t["desq_dg1"] = 2.0 * e1
    t["desq_dg2"] = 2.0 * e2
    t["wsel"] = np.ones(n)
    t["dwsel_dg1"] = np.zeros(n)
    t["dwsel_dg2"] = np.zeros(n)
    t["tract_id"] = rng.integers(0, 20, n)
    return t


def _cuts(t):
    return [Cut(np.asarray(t["fpfs1_s2n"]), ">", 8.0, np.asarray(t["fpfs1_ds2n_dg1"]),
                np.asarray(t["fpfs1_ds2n_dg2"]))]


@pytest.mark.parametrize("params", [None, [1.0, 2.0, 12.0, 0.1, 0.1]])
def test_full_response_matches_finite_difference(params):
    """<R> from the analytic + finite-difference terms equals the shear
    response of sum(W e_j) measured on catalogs sheared by +-g along each
    component, averaged over the two components as <R> is."""
    g = 0.002
    t0 = _catalog()
    # the selection term's finite difference uses the same step as the
    # sheared catalogs, so both count the same objects crossing the cut
    s = prepare_shape_weight_sample(t0, cuts=_cuts(t0), dg=g)
    _, parts = shape_weight_fom(params or [1, 1, 1, 0, 1], s, flat=params is None)

    def shear_sum(gg, comp):
        t = _catalog(**{f"g{comp}": gg})
        x = prepare_shape_weight_sample(t, cuts=_cuts(t))
        if params is None:
            W = x["w"]
        else:
            from xlens.utils.catalog.shape_weight import shape_weight_f1, shape_weight_f2
            W = x["w"] * shape_weight_f1(x["snr"], *params[1:4])[0] * shape_weight_f2(x["v"], params[4])[0]
        return np.sum((W * x[f"e{comp}"])[x["s0"]])

    r_fd = 0.5 * sum((shear_sum(g, c) - shear_sum(-g, c)) / (2 * g) for c in (1, 2)) / parts["n"]
    assert parts["sel"] != 0.0
    assert r_fd == pytest.approx(parts["R"], rel=2e-3)


def test_fom_independent_of_amp():
    t = _catalog(n=20000)
    s = prepare_shape_weight_sample(t, cuts=_cuts(t))
    a, _ = shape_weight_fom([1.0, 2.0, 12.0, 0.1, 0.1], s)
    b, _ = shape_weight_fom([7.0, 2.0, 12.0, 0.1, 0.1], s)
    assert a == pytest.approx(b, rel=1e-12)


def test_fit_beats_unweighted_on_holdout():
    t = _catalog()
    s = prepare_shape_weight_sample(t, cuts=_cuts(t), groups=t["tract_id"])
    res = fit_shape_weight(s, seed=1, verbose=False)
    assert res["gain_holdout"] > 1.05
    assert np.all(np.isfinite(res["params"]))
    # amp normalises the mean weight over the fit sample to 1 (parts
    # report f1 f2 without amp, which cancels in the figure of merit)
    _, parts = shape_weight_fom(res["params"], s)
    assert res["params"][0] * parts["mean_w"] == pytest.approx(1.0, abs=0.05)


def test_response_names():
    assert _response_name("fpfs1_s2n", 1) == "fpfs1_ds2n_dg1"
    assert _response_name("hsc_i_mag_fpfs1", 2) == "hsc_i_dmag_fpfs1_dg2"
    assert _response_name("hsc_i_s2n_fpfs1", 1) == "hsc_i_ds2n_fpfs1_dg1"
    assert _response_name("esq", 2) == "desq_dg2"
