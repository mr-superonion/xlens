"""Unit tests for the per-tract shear diagnostics task."""
import numpy as np

from xlens.analysis import shear_diagnostics as sd
from xlens.analysis.shear_diagnostics import (
    HIST_DTYPE,
    MEANSHEAR_DTYPE,
    ShearStatsPipe,
    ShearStatsPipeConfig,
    _axis_edges,
)


def _make_catalog(n=1000, seed=3):
    rng = np.random.RandomState(seed)
    cols = {
        "wsel": rng.uniform(0.2, 1.0, n),
        "dwsel_dg1": rng.normal(0, 0.1, n),
        "dwsel_dg2": rng.normal(0, 0.1, n),
        "fpfs1_e1": rng.normal(0, 0.1, n),
        "fpfs1_e2": rng.normal(0, 0.1, n),
        "fpfs1_de1_dg1": rng.uniform(0.5, 1.5, n),
        "fpfs1_de2_dg2": rng.uniform(0.5, 1.5, n),
        "esq": rng.uniform(0, 0.3, n),
        "desq_dg1": rng.normal(0, 0.2, n),
        "desq_dg2": rng.normal(0, 0.2, n),
        "fpfs1_s2n": rng.uniform(5, 100, n),
        "fpfs1_ds2n_dg1": rng.normal(0, 5.0, n),
        "fpfs1_ds2n_dg2": rng.normal(0, 5.0, n),
        "fpfs1_m00": rng.uniform(1.0, 2.0, n),
        # ARCSEC**2, not pixel**2: the ext_shapeHSM PSF moments are
        # converted at measurement time, so psf_fwhm_max compares a
        # FWHM in arcsec directly. 0.09 -> 2.3548*sqrt(0.09) = 0.71",
        # inside the 0.8" default cut. (The old 3.24 was pixel**2 and
        # now reads as a 4.2" PSF, which empties the selection.)
        "hsc_i_ext_shapeHSM_HsmPsfMoments_xx": np.full(n, 0.09),
        "hsc_i_ext_shapeHSM_HsmPsfMoments_yy": np.full(n, 0.09),
        "hsc_i_ext_shapeHSM_HsmPsfMoments_xy": np.zeros(n),
        "fpfs1_m20": rng.uniform(-0.5, 2.0, n),
        "hsc_i_mag_fpfs1": rng.uniform(20, 26.5, n),
        "hsc_i_s2n_fpfs1": rng.uniform(5, 100, n),
        "hsc_g_mag_fpfs1": rng.uniform(20, 27, n),
        "hsc_r_mag_fpfs1": rng.uniform(20, 26.5, n),
        "hsc_z_mag_fpfs1": rng.uniform(20, 26, n),
        "hsc_y_mag_fpfs1": rng.uniform(20, 26, n),
        "n_mask_base": rng.uniform(0.0, 0.3, n),
        "n_mask_discontinuity": rng.uniform(0.0, 1.0, n),
    }
    cat = np.zeros(n, dtype=[(k, "f8") for k in cols])
    for k, v in cols.items():
        cat[k] = v
    return cat


def _selected(cat, config):
    fwhm = 2.3548200 * np.sqrt(0.5 * (
        cat["hsc_i_ext_shapeHSM_HsmPsfMoments_xx"]
        + cat["hsc_i_ext_shapeHSM_HsmPsfMoments_yy"]
    ))
    return cat[
        (cat["hsc_i_mag_fpfs1"] < config.mag_max)
        & (cat["hsc_i_s2n_fpfs1"] > config.snr_min)
        & (cat["esq"] < config.esq_max)
        & (cat["n_mask_base"] < config.n_mask_base_max)
        & ((cat["fpfs1_m00"] + cat["fpfs1_m20"]) / cat["fpfs1_m00"]
           > config.trace_min)
        & (fwhm < config.psf_fwhm_max)
    ]


def test_axis_edges():
    expr, edges = _axis_edges(("a - b", -1.0, 1.0, 4, False))
    assert expr == "a - b"
    np.testing.assert_allclose(edges, np.linspace(-1, 1, 5))
    _, edges = _axis_edges(("snr", 10.0, 1000.0, 2, True))
    np.testing.assert_allclose(edges, [10, 100, 1000])


def test_selection_and_sums(monkeypatch):
    monkeypatch.setattr(
        sd, "PROPERTY_BINS",
        {"i_mag": ("hsc_i_mag_fpfs1", 20.0, 25.0, 5, False)},
    )
    monkeypatch.setattr(
        sd, "HIST_BINS",
        {"abs_we1": ("np.abs(wsel * fpfs1_e1)", 0.0, 0.4, 8, False)},
    )
    monkeypatch.setattr(sd, "HIST2D_BINS", {})
    cat = _make_catalog()
    config = ShearStatsPipeConfig()
    task = ShearStatsPipe(config=config)
    out = task.run(catalog=cat)

    sel = _selected(cat, config)
    stats = out.meanShearStats
    assert stats.dtype == MEANSHEAR_DTYPE
    assert len(stats) == 5
    # per-bin sums against a direct computation
    for row in stats:
        inbin = sel[
            (sel["hsc_i_mag_fpfs1"] >= row["x_min"])
            & (sel["hsc_i_mag_fpfs1"] < row["x_max"])
        ]
        w = inbin["wsel"]
        assert row["n_gal"] == len(inbin)
        np.testing.assert_allclose(row["sum_w"], np.sum(w))
        np.testing.assert_allclose(
            row["sum_wx"], np.sum(w * inbin["hsc_i_mag_fpfs1"])
        )
        np.testing.assert_allclose(row["sum_we1"], np.sum(w * inbin["fpfs1_e1"]))
        np.testing.assert_allclose(
            row["sum_r1"],
            np.sum(
                w * inbin["fpfs1_de1_dg1"]
                + inbin["dwsel_dg1"] * inbin["fpfs1_e1"]
            ),
        )
    # everything selected lands in some bin (range covers the cut)
    assert stats["n_gal"].sum() == len(sel)

    hists = out.histStats
    assert hists.dtype == HIST_DTYPE
    assert (hists["iy"] == -1).all()
    assert hists["count"].sum() == len(sel)  # |we| < 0.4 by construction


def test_missing_column_skipped(monkeypatch):
    monkeypatch.setattr(
        sd, "PROPERTY_BINS",
        {
            "ok": ("hsc_i_mag_fpfs1", 20.0, 25.0, 5, False),
            "psf_e1": ("nonexistent_column", 0.0, 1.0, 5, False),
        },
    )
    monkeypatch.setattr(sd, "HIST_BINS", {})
    monkeypatch.setattr(sd, "HIST2D_BINS", {})
    task = ShearStatsPipe(config=ShearStatsPipeConfig())
    out = task.run(catalog=_make_catalog())
    assert set(out.meanShearStats["property"]) == {"ok"}


def test_hist2d_counts(monkeypatch):
    monkeypatch.setattr(sd, "PROPERTY_BINS", {})
    monkeypatch.setattr(sd, "HIST_BINS", {})
    monkeypatch.setattr(
        sd, "HIST2D_BINS",
        {
            "cmd": (
                ("hsc_i_mag_fpfs1", 20.0, 25.0, 4, False),
                ("hsc_r_mag_fpfs1 - hsc_i_mag_fpfs1", -3.0, 3.0, 3, False),
            ),
        },
    )
    cat = _make_catalog()
    config = ShearStatsPipeConfig()
    task = ShearStatsPipe(config=config)
    out = task.run(catalog=cat)
    h = out.histStats
    assert len(h) == 12 and (h["iy"] >= 0).all()
    sel = _selected(cat, config)
    expect, _, _ = np.histogram2d(
        sel["hsc_i_mag_fpfs1"],
        sel["hsc_r_mag_fpfs1"] - sel["hsc_i_mag_fpfs1"],
        bins=(np.linspace(20, 25, 5), np.linspace(-3, 3, 4)),
    )
    got = np.zeros((4, 3))
    got[h["ix"], h["iy"]] = h["count"]
    np.testing.assert_array_equal(got, expect)


def test_default_registries_on_full_columns():
    """Defaults run end to end; PSF properties skip (no HSM columns)."""
    cat = _make_catalog()
    task = ShearStatsPipe(config=ShearStatsPipeConfig())
    out = task.run(catalog=cat)
    props = set(out.meanShearStats["property"])
    assert "i_mag" in props and "gmr_color" in props
    assert "psf_e1_i" in props and "psf_e2_i" in props
    names = set(out.histStats["name"])
    assert {"i_mag", "abs_we1", "abs_we2", "response", "cmd_i_rmi"} <= names


def test_snr_column(monkeypatch):
    """snr_column redirects the S/N cut; the weight is always wsel."""
    monkeypatch.setattr(
        sd, "PROPERTY_BINS",
        {"i_mag": ("hsc_i_mag_fpfs1", 20.0, 25.0, 1, False)},
    )
    monkeypatch.setattr(sd, "HIST_BINS", {})
    monkeypatch.setattr(sd, "HIST2D_BINS", {})
    cat = _make_catalog()
    rng = np.random.RandomState(5)
    cat["fpfs1_s2n"] = rng.uniform(3, 200, len(cat))
    config = ShearStatsPipeConfig()
    config.snr_column = "fpfs1_s2n"
    config.snr_min = 20.0
    out = ShearStatsPipe(config=config).run(catalog=cat)
    fwhm = 2.3548200 * np.sqrt(0.5 * (cat["hsc_i_ext_shapeHSM_HsmPsfMoments_xx"]
                                      + cat["hsc_i_ext_shapeHSM_HsmPsfMoments_yy"]))
    sel = cat[(cat["hsc_i_mag_fpfs1"] < config.mag_max) & (cat["fpfs1_s2n"] > 20.0)
              & (cat["esq"] < config.esq_max) & (cat["n_mask_base"] < config.n_mask_base_max)
              & ((cat["fpfs1_m00"] + cat["fpfs1_m20"]) / cat["fpfs1_m00"] > config.trace_min)
              & (fwhm < config.psf_fwhm_max)]
    inbin = sel[(sel["hsc_i_mag_fpfs1"] >= 20.0) & (sel["hsc_i_mag_fpfs1"] < 25.0)]
    row = out.meanShearStats[0]
    assert row["n_gal"] == len(inbin)
    np.testing.assert_allclose(row["sum_we1"], np.sum(inbin["wsel"] * inbin["fpfs1_e1"]))
    np.testing.assert_allclose(
        row["sum_r1"],
        np.sum(inbin["wsel"] * inbin["fpfs1_de1_dg1"] + inbin["dwsel_dg1"] * inbin["fpfs1_e1"]))


def test_optimal_shape_weight(monkeypatch):
    """With shape_weight_params the sums are of e_opt = w_opt e, with the
    response the shear derivative of the weighted shape."""
    monkeypatch.setattr(
        sd, "PROPERTY_BINS",
        {"i_mag": ("hsc_i_mag_fpfs1", 20.0, 25.0, 1, False)},
    )
    monkeypatch.setattr(sd, "HIST_BINS", {})
    monkeypatch.setattr(sd, "HIST2D_BINS", {})
    params = [3.0, 1.6, 14.0, 0.06, 0.0875]
    cat = _make_catalog()
    config = ShearStatsPipeConfig(shape_weight_params=params)
    config.validate()
    out = ShearStatsPipe(config=config).run(catalog=cat)
    assert config.esq_max == ShearStatsPipeConfig().esq_max
    row = out.meanShearStats[0]

    sel = _selected(cat, config)
    s, v = sel["fpfs1_s2n"], sel["esq"]
    w_opt = (params[0] * (s / 50.0) ** params[3]
             / (1.0 + (params[2] / s) ** params[1])
             * np.exp(-v / params[4]))
    assert np.all(w_opt > 0)
    w = sel["wsel"]
    assert row["n_gal"] == len(sel)
    np.testing.assert_allclose(row["sum_w"], np.sum(w))
    np.testing.assert_allclose(
        row["sum_we1"], np.sum(w * w_opt * sel["fpfs1_e1"]), rtol=1e-12
    )
    # response = finite difference of wsel * w_opt(s, v) * e1 when the
    # inputs are sheared: e1 += dg de1/dg1, s += dg ds2n/dg1, ...
    dg = 1e-5

    def we1(sign):
        e1 = sel["fpfs1_e1"] + sign * dg * sel["fpfs1_de1_dg1"]
        ss = s + sign * dg * sel["fpfs1_ds2n_dg1"]
        vv = v + sign * dg * sel["desq_dg1"]
        ww = w + sign * dg * sel["dwsel_dg1"]
        wo = (params[0] * (ss / 50.0) ** params[3]
              / (1.0 + (params[2] / ss) ** params[1])
              * np.exp(-vv / params[4]))
        return ww * wo * e1
    fd = np.sum(we1(+1) - we1(-1)) / (2 * dg)
    np.testing.assert_allclose(row["sum_r1"], fd, rtol=1e-6)
    # the unweighted run is recovered with an empty parameter list
    ref = ShearStatsPipe(config=ShearStatsPipeConfig()).run(catalog=cat)
    np.testing.assert_allclose(
        ref.meanShearStats[0]["sum_we1"], np.sum(w * sel["fpfs1_e1"])
    )


def test_shape_weight_params_validated():
    import pytest
    config = ShearStatsPipeConfig(shape_weight_params=[1.0, 2.0])
    with pytest.raises(ValueError):
        config.validate()
