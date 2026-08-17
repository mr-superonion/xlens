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
        "fpfs1_m00": rng.uniform(1.0, 2.0, n),
        "hsc_i_ext_shapeHSM_HsmPsfMoments_xx": np.full(n, 3.24),
        "hsc_i_ext_shapeHSM_HsmPsfMoments_yy": np.full(n, 3.24),
        "hsc_i_ext_shapeHSM_HsmPsfMoments_xy": np.zeros(n),
        "fpfs1_m20": rng.uniform(-0.5, 2.0, n),
        "hsc_i_mag_gauss2": rng.uniform(20, 26.5, n),
        "hsc_i_s2n_fpfs1": rng.uniform(5, 100, n),
        "hsc_g_mag_gauss2": rng.uniform(20, 27, n),
        "hsc_r_mag_gauss2": rng.uniform(20, 26.5, n),
        "hsc_z_mag_gauss2": rng.uniform(20, 26, n),
        "hsc_y_mag_gauss2": rng.uniform(20, 26, n),
        "n_mask_base": rng.uniform(0.0, 0.3, n),
        "n_mask_discontinuity": rng.uniform(0.0, 1.0, n),
    }
    cat = np.zeros(n, dtype=[(k, "f8") for k in cols])
    for k, v in cols.items():
        cat[k] = v
    return cat


def _selected(cat, config):
    return cat[
        (cat["hsc_i_mag_gauss2"] < config.mag_max)
        & (cat["hsc_i_s2n_fpfs1"] > config.snr_min)
        & (cat["esq"] < config.esq_max)
        & (cat["n_mask_base"] < config.n_mask_base_max)
        & ((cat["fpfs1_m00"] + cat["fpfs1_m20"]) / cat["fpfs1_m00"]
           > config.trace_min)
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
        {"i_mag": ("hsc_i_mag_gauss2", 20.0, 25.0, 5, False)},
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
            (sel["hsc_i_mag_gauss2"] >= row["x_min"])
            & (sel["hsc_i_mag_gauss2"] < row["x_max"])
        ]
        w = inbin["wsel"]
        assert row["n_gal"] == len(inbin)
        np.testing.assert_allclose(row["sum_w"], np.sum(w))
        np.testing.assert_allclose(
            row["sum_wx"], np.sum(w * inbin["hsc_i_mag_gauss2"])
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
            "ok": ("hsc_i_mag_gauss2", 20.0, 25.0, 5, False),
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
                ("hsc_i_mag_gauss2", 20.0, 25.0, 4, False),
                ("hsc_r_mag_gauss2 - hsc_i_mag_gauss2", -3.0, 3.0, 3, False),
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
        sel["hsc_i_mag_gauss2"],
        sel["hsc_r_mag_gauss2"] - sel["hsc_i_mag_gauss2"],
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
