"""The optimal shape weight on per-image catalogs: the response of
``wsel * w_opt * e`` is the finite difference of the whole quantity under a
shear of every input, and measure_shear / ShearEstimator use exactly that."""
import numpy as np
import pytest

from xlens.utils.catalog import ShearEstimator, measure_shear
from xlens.utils.catalog.shape_weight import (
    DEFAULT_SHAPE_WEIGHT_PARAMS,
    ds2n_column_name,
    optimal_shape_weight,
    parse_shape_weight_params,
    shape_weight_from_columns,
)

PARAMS = list(DEFAULT_SHAPE_WEIGHT_PARAMS)
DG = 1e-5


def _catalog(n=2000, seed=11):
    rng = np.random.RandomState(seed)
    cols = {
        "wsel": rng.uniform(0.2, 1.0, n),
        "dwsel_dg1": rng.normal(0, 0.1, n),
        "dwsel_dg2": rng.normal(0, 0.1, n),
        "fpfs_e1": rng.normal(0, 0.2, n),
        "fpfs_e2": rng.normal(0, 0.2, n),
        "fpfs_de1_dg1": rng.uniform(0.5, 1.5, n),
        "fpfs_de1_dg2": rng.normal(0, 0.2, n),
        "fpfs_de2_dg1": rng.normal(0, 0.2, n),
        "fpfs_de2_dg2": rng.uniform(0.5, 1.5, n),
        "fpfs_m0": rng.uniform(1.0, 2.0, n),
        "fpfs_dm0_dg1": rng.normal(0, 0.1, n),
        "fpfs_dm0_dg2": rng.normal(0, 0.1, n),
        "fpfs_m2": rng.uniform(0.5, 2.0, n),
        "fpfs_dm2_dg1": rng.normal(0, 0.1, n),
        "fpfs_dm2_dg2": rng.normal(0, 0.1, n),
        "lsst_i_flux_fpfs1": rng.uniform(20, 500, n),
        "lsst_i_dflux_fpfs1_dg1": rng.normal(0, 5, n),
        "lsst_i_dflux_fpfs1_dg2": rng.normal(0, 5, n),
        "lsst_i_s2n_fpfs1": rng.uniform(5, 100, n),
        "lsst_i_ds2n_fpfs1_dg1": rng.normal(0, 5, n),
        "lsst_i_ds2n_fpfs1_dg2": rng.normal(0, 5, n),
    }
    cat = np.zeros(n, dtype=[(k, "f8") for k in cols])
    for k, v in cols.items():
        cat[k] = v
    return cat


def _sheared(cat, comp, sign):
    """Every input moved to first order by sign * DG in g_comp."""
    out = cat.copy()
    for base in ("fpfs_e1", "fpfs_e2", "fpfs_m0", "fpfs_m2",
                 "lsst_i_flux_fpfs1", "lsst_i_s2n_fpfs1", "wsel"):
        if base == "wsel":
            dcol = f"dwsel_dg{comp}"
        elif base.startswith("fpfs_"):
            dcol = base.replace("fpfs_", "fpfs_d") + f"_dg{comp}"
        else:
            dcol = ds2n_column_name(base, comp) if "s2n" in base \
                else base.replace("_flux_", "_dflux_") + f"_dg{comp}"
        out[base] = cat[base] + sign * DG * cat[dcol]
    return out


def test_ds2n_column_name():
    assert ds2n_column_name("lsst_i_s2n_fpfs1", 1) == "lsst_i_ds2n_fpfs1_dg1"
    assert ds2n_column_name("fpfs1_s2n", 2) == "fpfs1_ds2n_dg2"
    with pytest.raises(ValueError):
        ds2n_column_name("lsst_i_flux_fpfs1", 1)


def test_parse_params():
    assert parse_shape_weight_params([1.6, 14.0, 0.06, 0.0875])[0] == 1.0
    with pytest.raises(ValueError):
        parse_shape_weight_params([1.0, 2.0, 3.0])


def test_weight_from_columns_response_is_finite_difference():
    cat = _catalog()
    w, dw1, dw2 = shape_weight_from_columns(cat, PARAMS)
    amp, alpha, snr0, beta, esq_scale = parse_shape_weight_params(PARAMS)
    e1, e2 = cat["fpfs_e1"], cat["fpfs_e2"]
    w_ref, _, _ = optimal_shape_weight(
        cat["lsst_i_s2n_fpfs1"], e1 * e1 + e2 * e2,
        amp, alpha, snr0, beta, esq_scale,
    )
    np.testing.assert_allclose(w, w_ref, rtol=1e-12)
    assert np.all(w > 0)  # no cut on |e|
    for comp, dw in ((1, dw1), (2, dw2)):
        wp = shape_weight_from_columns(_sheared(cat, comp, +1), PARAMS)[0]
        wm = shape_weight_from_columns(_sheared(cat, comp, -1), PARAMS)[0]
        np.testing.assert_allclose(dw, (wp - wm) / (2 * DG), rtol=1e-5,
                                   atol=1e-8)


def test_measure_shear_with_optimal_weight():
    cat = _catalog()
    kw = dict(z_estimator=None, zbounds=None, flux_min=30.0, emax=0.8,
              trace_min=None, dg=0.02, flux_name="fpfs1", shape_name="fpfs",
              bands=("lsst_i",), ref_band="lsst_i")
    plain = measure_shear(src=cat, target="g1", **kw)
    out = measure_shear(src=cat, target="g1", shape_weight_params=PARAMS,
                        **kw)
    w, dw1, _ = shape_weight_from_columns(cat, PARAMS)
    sel = cat["lsst_i_flux_fpfs1"] > 30.0
    sel &= cat["fpfs_e1"] ** 2 + cat["fpfs_e2"] ** 2 < 0.64
    ww = cat["wsel"] * w
    np.testing.assert_allclose(out["n_gal"], plain["n_gal"])
    np.testing.assert_allclose(out["e"], [np.sum((ww * cat["fpfs_e1"])[sel])],
                               rtol=1e-12)
    r_ref = np.sum((ww * cat["fpfs_de1_dg1"]
                    + (cat["dwsel_dg1"] * w + cat["wsel"] * dw1)
                    * cat["fpfs_e1"])[sel])
    np.testing.assert_allclose(out["r"], [r_ref], rtol=1e-12)
    # the selection term is a finite difference of the cuts only, with
    # the weight held fixed, so it is the plain one reweighted
    assert out["r_sel"].shape == plain["r_sel"].shape
    # ShearEstimator agrees with the function
    est = ShearEstimator(mag_max=31.4 - 2.5 * np.log10(30.0), emax=0.8,
                         trace_min=None, flux_name="fpfs1",
                         shape_name="fpfs", bands=("lsst_i",),
                         ref_band="lsst_i", dg=0.02,
                         shape_weight_params=PARAMS)
    out2 = est.measure_shear(cat, target="g1")
    np.testing.assert_allclose(out["e"], out2["e"], rtol=1e-10)
    np.testing.assert_allclose(out["r"], out2["r"] + out2["r_det"],
                               rtol=1e-10)
    np.testing.assert_allclose(out["r_sel"], out2["r_sel"], rtol=1e-10)
    # empty / None parameters reproduce the unweighted estimator
    out0 = measure_shear(src=cat, target="g1", shape_weight_params=[], **kw)
    np.testing.assert_allclose(out0["e"], plain["e"])
    np.testing.assert_allclose(out0["r"], plain["r"])
