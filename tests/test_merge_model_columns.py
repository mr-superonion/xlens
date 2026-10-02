"""MergePipe carries the Gaussian-model columns (model_flux, model_mxx,
model_myy, model_mxy and their shear derivatives) to the merged catalog,
with the spin-2 part of the covariance treated like the FPFS shape."""
import numpy as np
from astropy.table import Table
from xlens.processor.merge import (
    MODEL_MERGE_COLUMNS,
    MergePipe,
    MergePipeConfig,
)
from xlens.utils.columns import MODEL_COLUMNS

BANDS = ["r", "i"]


def _pipe(**kw):
    cfg = MergePipeConfig()
    cfg.bands = BANDS
    cfg.do_flexzboost = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return MergePipe(config=cfg)


def _catalog(n=5, seed=1):
    rng = np.random.default_rng(seed)
    t = Table()
    p = MergePipeConfig().fpfs_prefix
    base = [
        "ra", "dec", "x1", "x2", "x1_det", "x2_det", "wsel", "dwsel_dg1",
        "dwsel_dg2", "n_mask_base", "bkg", "dbkg_dg1", "dbkg_dg2",
        f"{p}e1", f"{p}e2", f"{p}de1_dg1", f"{p}de1_dg2", f"{p}de2_dg1",
        f"{p}de2_dg2", f"{p}m00", f"{p}dm00_dg1", f"{p}dm00_dg2",
        f"{p}m20", f"{p}dm20_dg1", f"{p}dm20_dg2",
    ]
    for b in BANDS:
        base += [
            f"{b}_flux_fpfs1", f"{b}_dflux_fpfs1_dg1", f"{b}_dflux_fpfs1_dg2",
            f"{b}_flux_fpfs1_err", f"{b}_s2n_fpfs1", f"{b}_ds2n_fpfs1_dg1",
            f"{b}_ds2n_fpfs1_dg2", f"{b}_mag_fpfs1", f"{b}_dmag_fpfs1_dg1",
            f"{b}_dmag_fpfs1_dg2", f"{b}_mag_fpfs1_err",
            f"{b}_dmag_fpfs1_err_dg1", f"{b}_dmag_fpfs1_err_dg2",
        ]
    for c in base + list(MODEL_MERGE_COLUMNS):
        t[c] = rng.normal(size=n)
    t["wsel"] = 1.0
    t["is_primary"] = np.ones(n, dtype=bool)
    t["object_id"] = np.arange(n, dtype=np.int64)
    t["tract_id"] = np.zeros(n, dtype=np.int32)
    t["patch_x"] = np.zeros(n, dtype=np.int32)
    t["patch_y"] = np.zeros(n, dtype=np.int32)
    return t


def test_model_columns_are_kept():
    assert set(MODEL_MERGE_COLUMNS) == set(MODEL_COLUMNS.values())
    pipe = _pipe()
    cat = _catalog()
    out = pipe._finalize_columns(cat, [])
    for c in MODEL_MERGE_COLUMNS:
        assert c in out.colnames
        np.testing.assert_array_equal(out[c], cat[c])
    # still optional: a catalog without them merges as before
    cat2 = _catalog()
    cat2.remove_columns(MODEL_MERGE_COLUMNS)
    out2 = pipe._finalize_columns(cat2, [])
    assert not any(c in out2.colnames for c in MODEL_MERGE_COLUMNS)


def test_flipu_signs():
    pipe = _pipe()
    cat = _catalog()
    ref = {c: np.asarray(cat[c]).copy() for c in MODEL_MERGE_COLUMNS}
    out = pipe._apply_flipu(cat, [])
    flipped = {"model_mxy", "model_dflux_dg2", "model_dmxx_dg2",
               "model_dmyy_dg2", "model_dmxy_dg1"}
    for c in MODEL_MERGE_COLUMNS:
        sign = -1.0 if c in flipped else 1.0
        np.testing.assert_array_equal(out[c], sign * ref[c])
    # the model ellipticity transforms like the FPFS one: e2 -> -e2
    e2 = 2 * ref["model_mxy"] / (ref["model_mxx"] + ref["model_myy"])
    e2_out = 2 * np.asarray(out["model_mxy"]) / (
        np.asarray(out["model_mxx"]) + np.asarray(out["model_myy"])
    )
    np.testing.assert_allclose(e2_out, -e2)


def test_wcs_correction_rotates_model_covariance(monkeypatch):
    """With zero WCS shear the correction is a pure rotation of the
    covariance by -rho: the trace is unchanged and (mxx - myy, 2 mxy)
    rotates by -2 rho, exactly as fpfs1 (e1, e2) do."""
    import xlens.processor.merge as M

    rho = 0.3
    monkeypatch.setattr(
        M, "sky_to_pixel", lambda wcs, ra, dec: (np.zeros_like(ra), np.zeros_like(dec))
    )
    monkeypatch.setattr(
        M, "extract_perturbation_dm_wcs", lambda wcs, pt, scale: (0.0, 0.0, rho, 0.0)
    )
    pipe = _pipe()
    cat = _catalog()
    mxx, myy, mxy = (np.asarray(cat[f"model_{k}"]).copy() for k in ("mxx", "myy", "mxy"))
    p = pipe.config.fpfs_prefix
    e1, e2 = np.asarray(cat[f"{p}e1"]).copy(), np.asarray(cat[f"{p}e2"]).copy()
    out = pipe._apply_wcs_correction(cat, wcs=None, pixel_scale=0.2)
    c2, s2 = np.cos(2 * rho), np.sin(2 * rho)
    # fpfs reference behaviour
    np.testing.assert_allclose(out[f"{p}e1"], e1 * c2 + e2 * s2)
    np.testing.assert_allclose(out[f"{p}e2"], -e1 * s2 + e2 * c2)
    # model covariance: same rotation of the spin-2 part, trace kept
    q1, q2 = mxx - myy, 2 * mxy
    o_mxx, o_myy, o_mxy = (np.asarray(out[f"model_{k}"]) for k in ("mxx", "myy", "mxy"))
    np.testing.assert_allclose(o_mxx + o_myy, mxx + myy)
    np.testing.assert_allclose(o_mxx - o_myy, q1 * c2 + q2 * s2)
    np.testing.assert_allclose(2 * o_mxy, -q1 * s2 + q2 * c2)
    # the shear derivatives are untouched, like the fpfs ones
    for c in MODEL_MERGE_COLUMNS:
        if "_dg" in c:
            np.testing.assert_array_equal(out[c], cat[c])


def test_wcs_correction_undoes_shear_on_model_covariance(monkeypatch):
    import xlens.processor.merge as M

    g1w, g2w = 0.01, -0.02
    monkeypatch.setattr(
        M, "sky_to_pixel", lambda wcs, ra, dec: (np.zeros_like(ra), np.zeros_like(dec))
    )
    monkeypatch.setattr(
        M, "extract_perturbation_dm_wcs", lambda wcs, pt, scale: (g1w, g2w, 0.0, 0.0)
    )
    pipe = _pipe()
    cat = _catalog()
    ref = {c: np.asarray(cat[c]).copy() for c in cat.colnames}
    out = pipe._apply_wcs_correction(cat, wcs=None, pixel_scale=0.2)
    q1 = ref["model_mxx"] - ref["model_myy"]
    q2 = 2 * ref["model_mxy"]
    dq1_1 = ref["model_dmxx_dg1"] - ref["model_dmyy_dg1"]
    dq1_2 = ref["model_dmxx_dg2"] - ref["model_dmyy_dg2"]
    dq2_1, dq2_2 = 2 * ref["model_dmxy_dg1"], 2 * ref["model_dmxy_dg2"]
    np.testing.assert_allclose(
        np.asarray(out["model_mxx"]) - np.asarray(out["model_myy"]),
        q1 - g1w * dq1_1 - g2w * dq1_2,
    )
    np.testing.assert_allclose(
        2 * np.asarray(out["model_mxy"]), q2 - g1w * dq2_1 - g2w * dq2_2
    )
    np.testing.assert_allclose(
        np.asarray(out["model_mxx"]) + np.asarray(out["model_myy"]),
        ref["model_mxx"] + ref["model_myy"],
    )
