"""Per-object S/N^2 band weights, the combined S/N and the optimal shape
weight in MergePipe: every shear derivative must equal the finite
difference of the quantity it belongs to."""
import numpy as np
from astropy.table import Table

from xlens.processor.merge import (
    MergePipe,
    MergePipeConfig,
    shape_weight_f1,
    shape_weight_f2,
)

BANDS = ["r", "i", "z"]
FIXED = [0.3, 0.5, 0.2]
DG = 1e-4


def _pipe(**kw):
    cfg = MergePipeConfig()
    cfg.bands = BANDS
    cfg.band_weights = FIXED
    cfg.do_flexzboost = False
    cfg.do_wcs_correction = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    cfg.validate()
    return MergePipe(config=cfg)


def _perband(n=400, seed=2):
    """Per-band inputs plus, for each, its 'true' shear derivative."""
    rng = np.random.default_rng(seed)
    p = MergePipeConfig().fpfs_prefix
    t = Table()
    for b in BANDS:
        for X, scale in (("m00", 30.0), ("m20", 5.0), ("m22c", 3.0), ("m22s", 3.0)):
            t[f"{b}_{p}{X}"] = rng.normal(scale, 0.3 * scale, n) if X == "m00" else rng.normal(0, scale, n)
            for c in (1, 2):
                t[f"{b}_{p}d{X}_dg{c}"] = rng.normal(0, scale, n)
        err = rng.uniform(1.0, 3.0, n)
        flux = rng.uniform(5.0, 80.0, n) * err
        t[f"{b}_flux_fpfs1"] = flux
        t[f"{b}_flux_fpfs1_err"] = err
        for c in (1, 2):
            t[f"{b}_dflux_fpfs1_dg{c}"] = rng.normal(0, 5.0, n)
        t[f"{b}_s2n_fpfs1"] = flux / err
        for c in (1, 2):
            t[f"{b}_ds2n_fpfs1_dg{c}"] = t[f"{b}_dflux_fpfs1_dg{c}"] / err
        for col in (f"{b}_mag_fpfs1", f"{b}_dmag_fpfs1_dg1", f"{b}_dmag_fpfs1_dg2",
                    f"{b}_mag_fpfs1_err", f"{b}_dmag_fpfs1_err_dg1", f"{b}_dmag_fpfs1_err_dg2"):
            t[col] = rng.normal(size=n)
    t["wsel"] = rng.uniform(0.3, 1.0, n)
    t["dwsel_dg1"] = rng.normal(0, 0.1, n)
    t["dwsel_dg2"] = rng.normal(0, 0.1, n)
    return t


def _sheared(t, c, sign):
    """Every per-band input moved to first order along g_c."""
    p = MergePipeConfig().fpfs_prefix
    u = t.copy()
    for b in BANDS:
        for X in ("m00", "m20", "m22c", "m22s"):
            u[f"{b}_{p}{X}"] = t[f"{b}_{p}{X}"] + sign * DG * t[f"{b}_{p}d{X}_dg{c}"]
        u[f"{b}_flux_fpfs1"] = t[f"{b}_flux_fpfs1"] + sign * DG * t[f"{b}_dflux_fpfs1_dg{c}"]
        u[f"{b}_s2n_fpfs1"] = u[f"{b}_flux_fpfs1"] / t[f"{b}_flux_fpfs1_err"]
    u["wsel"] = t["wsel"] + sign * DG * t[f"dwsel_dg{c}"]
    return u


def _combined(pipe, t):
    what, dwhat = pipe._band_weights(t)
    return pipe._combine_band_moments(t.copy(), what, dwhat)


def test_fixed_weights_unchanged():
    pipe = _pipe()
    t = _perband()
    out = _combined(pipe, t)
    p = pipe.config.fpfs_prefix
    m00 = sum(w * np.asarray(t[f"{b}_{p}m00"]) for b, w in zip(BANDS, FIXED))
    np.testing.assert_allclose(out[f"{p}m00"], m00, rtol=1e-12)
    dm = sum(w * np.asarray(t[f"{b}_{p}dm22c_dg1"]) for b, w in zip(BANDS, FIXED))
    d00 = sum(w * np.asarray(t[f"{b}_{p}dm00_dg1"]) for b, w in zip(BANDS, FIXED))
    m22c = sum(w * np.asarray(t[f"{b}_{p}m22c"]) for b, w in zip(BANDS, FIXED))
    den = m00 + pipe.config.fpfs_c0
    np.testing.assert_allclose(out[f"{p}de1_dg1"], dm / den - m22c / den**2 * d00, rtol=1e-10)


def test_snr2_weights_normalised_and_exclude_zero_bands():
    pipe = _pipe(per_object_snr2_weights=True, band_weights=[0.0, 0.5, 0.5])
    t = _perband()
    what, dwhat = pipe._band_weights(t)
    np.testing.assert_allclose(what.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(what[:, 0] == 0.0) and np.all(dwhat[:, 0, :] == 0.0)
    np.testing.assert_allclose(dwhat.sum(axis=1), 0.0, atol=1e-12)
    s_i, s_z = np.asarray(t["i_s2n_fpfs1"]), np.asarray(t["z_s2n_fpfs1"])
    np.testing.assert_allclose(what[:, 1], s_i**2 / (s_i**2 + s_z**2), rtol=1e-12)


def test_snr2_weights_fallback_rows():
    pipe = _pipe(per_object_snr2_weights=True)
    t = _perband()
    for b in BANDS:
        t[f"{b}_s2n_fpfs1"][:3] = np.nan
    what, dwhat = pipe._band_weights(t)
    np.testing.assert_allclose(what[:3], np.tile(FIXED, (3, 1)))
    assert np.all(dwhat[:3] == 0.0)


def test_snr2_derivatives_match_finite_difference():
    pipe = _pipe(per_object_snr2_weights=True)
    t = _perband()
    p = pipe.config.fpfs_prefix
    out = _combined(pipe, t)
    for c in (1, 2):
        plus = _combined(pipe, _sheared(t, c, +1))
        minus = _combined(pipe, _sheared(t, c, -1))
        for col in (f"{p}m00", f"{p}m20", f"{p}e1", f"{p}e2", f"{p}s2n"):
            fd = (np.asarray(plus[col]) - np.asarray(minus[col])) / (2 * DG)
            dcol = {f"{p}m00": f"{p}dm00_dg{c}", f"{p}m20": f"{p}dm20_dg{c}",
                    f"{p}e1": f"{p}de1_dg{c}", f"{p}e2": f"{p}de2_dg{c}",
                    f"{p}s2n": f"{p}ds2n_dg{c}"}[col]
            np.testing.assert_allclose(np.asarray(out[dcol]), fd, rtol=2e-5, atol=2e-6)


def test_shape_weight_applied_to_shape():
    params = [3.0, 1.6, 14.0, 0.06, 0.0875, 0.565, 0.64]
    pipe = _pipe(per_object_snr2_weights=True, shape_weight_params=params)
    plain = _pipe(per_object_snr2_weights=True)
    t = _perband()

    def finalize(pp, tab):
        tab = _combined(pp, tab)
        tab["is_primary"] = np.ones(len(tab), dtype=bool)
        tab["object_id"] = np.arange(len(tab))
        for k in ("ra", "dec", "x1", "x2", "x1_det", "x2_det", "n_mask_base",
                  "bkg", "dbkg_dg1", "dbkg_dg2", "tract_id", "patch_x", "patch_y"):
            tab[k] = np.zeros(len(tab))
        return pp._finalize_columns(tab, [])
    out, ref = finalize(pipe, t), finalize(plain, t)
    assert "w_shape" not in out.colnames and "dw_shape_dg1" not in out.colnames
    f1, _ = shape_weight_f1(ref["fpfs1_s2n"], *params[1:4], params[0])
    f2, _ = shape_weight_f2(ref["esq"], *params[4:])
    # wsel, esq untouched; the shape carries the weight
    np.testing.assert_allclose(out["wsel"], ref["wsel"], rtol=1e-12)
    np.testing.assert_allclose(out["esq"], ref["esq"], rtol=1e-12)
    for i in (1, 2):
        np.testing.assert_allclose(out[f"fpfs1_e{i}"], np.asarray(ref[f"fpfs1_e{i}"]) * f1 * f2, rtol=1e-12)
    # and fpfs1_de_i_dg_j is the derivative of the weighted shape (finite difference of the whole pipeline)
    for j in (1, 2):
        plus, minus = finalize(pipe, _sheared(t, j, +1)), finalize(pipe, _sheared(t, j, -1))
        for i in (1, 2):
            fd = (np.asarray(plus[f"fpfs1_e{i}"]) - np.asarray(minus[f"fpfs1_e{i}"])) / (2 * DG)
            np.testing.assert_allclose(np.asarray(out[f"fpfs1_de{i}_dg{j}"]), fd, rtol=2e-5, atol=1e-7)
    f2z, df2z = shape_weight_f2(np.array([0.64, 0.7]), *params[4:])
    assert np.all(f2z == 0.0) and np.all(df2z == 0.0)
