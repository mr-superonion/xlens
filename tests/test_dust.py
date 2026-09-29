"""Per-band Galactic extinction (xlens.utils.dust)."""

import os

import numpy as np
import pytest

from xlens.utils import dust

dust_extinction = pytest.importorskip("dust_extinction")

# dp2utils PR #1 reference values: three positions, R_V = 3.1, DP2 butler passbands.
RA = [61.0, 55.0, 9.45]
DEC = [-35.0, -40.0, -44.0]
EXPECTED = {
    ("G23", "csfd"): {
        "u": [0.027344, 0.055128, 0.033401], "g": [0.021212, 0.042766, 0.025911],
        "r": [0.015434, 0.031116, 0.018852], "i": [0.011597, 0.023382, 0.014166],
        "z": [0.009272, 0.018693, 0.011326], "y": [0.007524, 0.015168, 0.009190]},
    ("F99", "csfd"): {
        "u": [0.027128, 0.054694, 0.033138], "g": [0.021022, 0.042383, 0.025679],
        "r": [0.014788, 0.029814, 0.018064], "i": [0.010932, 0.022039, 0.013353],
        "z": [0.008657, 0.017453, 0.010575], "y": [0.007110, 0.014335, 0.008685]},
    ("F99", "sfd"): {
        "u": [0.025599, 0.058446, 0.031111], "g": [0.019837, 0.045291, 0.024109],
        "r": [0.013954, 0.031859, 0.016959], "i": [0.010315, 0.023551, 0.012536],
        "z": [0.008169, 0.018651, 0.009928], "y": [0.006709, 0.015318, 0.008154]},
}


def _has_maps():
    root = os.environ.get("XLENS_DUST_MAP_DIR") or dust.DUST_MAP_DIR
    return os.path.exists(os.path.join(root, "csfd", "csfd_ebv.fits"))


def test_band_average_of_narrow_passband_is_the_curve():
    """A narrow box passband recovers A_lambda/A_V at its wavelength."""
    import astropy.units as u
    from dust_extinction.parameter_averages import F99

    model = F99(Rv=3.1)
    lam = np.linspace(540.0, 560.0, 201)
    axav = dust.band_AxAv(model, lam, np.ones_like(lam))
    np.testing.assert_allclose(axav, model(550.0 * u.nm), rtol=2e-3)


def test_bundled_passbands_all_surveys():
    """Every bundled set loads and extinction falls with wavelength (A_g > A_r > ...)."""
    from dust_extinction.parameter_averages import F99, G23

    for survey, bands in dust.PASSBANDS.items():
        pb = dust.load_passbands(survey)
        assert tuple(pb) == bands
        for model in (F99(Rv=3.1), G23(Rv=3.1)):
            axav = [dust.band_AxAv(model, *pb[b]) for b in bands]
            assert np.all(np.diff(axav) < 0), (survey, type(model).__name__, axav)
            assert 0.05 < min(axav) and max(axav) < 2.0


def test_out_of_range_passband_raises():
    from dust_extinction.parameter_averages import F99

    lam = np.linspace(3000.0, 4000.0, 50)  # F99 stops at 3.3 micron
    with pytest.raises(ValueError, match="outside"):
        dust.band_AxAv(F99(Rv=3.1), lam, np.ones_like(lam))


def test_unknown_model_lists_choices():
    with pytest.raises(ValueError, match="F99"):
        dust._extinction_model("NOPE", 3.1)


@pytest.mark.skipif(not _has_maps(), reason="dust maps not available")
@pytest.mark.parametrize("model, dustmap", sorted(EXPECTED))
def test_dp2utils_reference_values(model, dustmap):
    """Matches dp2utils PR #1 for LSST (butler passbands when reachable)."""
    corr = dust.DustCorrector("lsst", Rv=3.1, model=model, dustmap=dustmap)
    Ax = corr(RA, DEC)
    assert set(Ax) == set("ugrizy")
    # 1e-4 with the DP2 butler curves; the bundled SVO LSST curves differ slightly
    rtol = 1e-4 if corr.passbands["g"][0].size == 1601 else 3e-2
    for band in "ugrizy":
        np.testing.assert_allclose(Ax[band], EXPECTED[(model, dustmap)][band], rtol=rtol, err_msg=band)
    arr = corr(RA, DEC, return_type="array")
    assert arr.shape == (3, 6)
    np.testing.assert_allclose(corr(RA, DEC, "r"), Ax["r"])
    assert set(corr(RA, DEC, ["g", "i"])) == {"g", "i"}


@pytest.mark.skipif(not _has_maps(), reason="dust maps not available")
def test_des_euclid_hsc_share_the_same_Av():
    """Different surveys, same sky: same A_V, band values ordered by wavelength."""
    corrs = {s: dust.DustCorrector(s, dustmap="sfd") for s in ("des", "euclid", "hsc")}
    Av = {s: c.get_Av(RA, DEC) for s, c in corrs.items()}
    np.testing.assert_allclose(Av["des"], Av["euclid"])
    np.testing.assert_allclose(Av["des"], Av["hsc"])
    vis = corrs["euclid"](RA, DEC, "vis")
    des = corrs["des"](RA, DEC)
    assert np.all(des["r"] > vis) and np.all(vis > des["i"])  # VIS spans r+i+z
