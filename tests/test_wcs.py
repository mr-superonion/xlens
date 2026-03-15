"""Tests for xlens.wcs: Jacobian decomposition, WCS conversion, and
ellipticity correction.
"""

import anacal
import galsim
import lsst.geom as geom
import numpy as np
import pytest
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig

import xlens
from xlens.wcs import (
    correct_ellipticity_wcs,
    correct_fpfs_spin2_wcs,
    extract_perturbation_dm_wcs,
    extract_perturbation_galsim_wcs,
    extract_perturbation_jwcs,
    jacobian_decomposition,
    jacobian_reconstruction,
    make_jwcs,
    make_tanwcs_dm,
    make_tanwcs_galsim,
    tanwcs_galsim2dm,
)

# ---------------------------------------------------------------------------
# Helpers for FPFS shape measurement tests
# ---------------------------------------------------------------------------



def _draw_images(gal_obj, psf_obj, wcs, stamp_size, center):
    """Draw galaxy and PSF images with given WCS."""
    gal_conv = galsim.Convolve(gal_obj, psf_obj)
    gal_image = gal_conv.drawImage(
        nx=stamp_size, ny=stamp_size,
        wcs=wcs, center=galsim.PositionD(center, center),
    )
    psf_image = psf_obj.drawImage(
        nx=stamp_size, ny=stamp_size,
        wcs=wcs, center=galsim.PositionD(center, center),
    )
    return gal_image, psf_image


def _make_fpfs_config_and_det(center, sigma_shapelets):
    """Build FpfsConfig and single-source detection array."""
    det_dtype = np.dtype([("y", np.float64), ("x", np.float64)])
    det = np.array([(center, center)], dtype=det_dtype)
    fpfs_config = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=sigma_shapelets,
    )
    return fpfs_config, det


def simulate_and_measure(
    gal_obj, psf_obj, wcs, stamp_size, center, pixel_scale, sigma_shapelets,
):
    """Draw galaxy+PSF with given WCS and measure FPFS shapes."""
    gal_image, psf_image = _draw_images(
        gal_obj, psf_obj, wcs, stamp_size, center,
    )
    fpfs_config, det = _make_fpfs_config_and_det(center, sigma_shapelets)
    return anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        pixel_scale=pixel_scale,
        mag_zero=30.0,
        noise_variance=0.0,
        gal_array=gal_image.array,
        psf_array=psf_image.array,
        mask_array=None,
        noise_array=None,
        detection=det,
        do_compute_detect_weight=False,
    )


def simulate_and_measure_linear(
    gal_obj, psf_obj, wcs, stamp_size, center, pixel_scale, sigma_shapelets,
):
    """Draw galaxy+PSF with given WCS and return raw linear shapelet moments."""
    gal_image, psf_image = _draw_images(
        gal_obj, psf_obj, wcs, stamp_size, center,
    )
    fpfs_config, det = _make_fpfs_config_and_det(center, sigma_shapelets)
    return anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        pixel_scale=pixel_scale,
        mag_zero=30.0,
        noise_variance=0.0,
        gal_array=gal_image.array,
        psf_array=psf_image.array,
        mask_array=None,
        noise_array=None,
        detection=det,
        do_compute_detect_weight=False,
        return_only_linear_modes=True,
        pack_linear_modes=True,
    )


# ---------------------------------------------------------------------------
# Test: WCS round-trip (existing test from test_simulator.py)
# ---------------------------------------------------------------------------


def test_wcs():
    config = RingsSkyMapConfig()
    config.numRings = 120
    config.projection = "TAN"
    config.tractOverlap = 1.0 / 60
    config.pixelScale = 0.168
    skymap = RingsSkyMap(config)
    tract_info = skymap[16012]

    wcs_galsim = xlens.wcs.tanwcs_dm2galsim(tract_info.getWcs())
    wcs_dm = xlens.wcs.tanwcs_galsim2dm(wcs_galsim)
    wcs_0 = tract_info.getWcs()

    np.testing.assert_almost_equal(
        wcs_dm.getCdMatrix(), wcs_0.getCdMatrix(),
    )
    np.testing.assert_almost_equal(
        wcs_dm.getPixelOrigin().x, wcs_0.getPixelOrigin().x,
    )
    np.testing.assert_almost_equal(
        wcs_dm.getPixelOrigin().y, wcs_0.getPixelOrigin().y,
    )
    np.testing.assert_almost_equal(
        wcs_dm.getSkyOrigin().getRa().asDegrees(),
        wcs_0.getSkyOrigin().getRa().asDegrees(),
    )
    np.testing.assert_almost_equal(
        wcs_dm.getSkyOrigin().getDec().asDegrees(),
        wcs_0.getSkyOrigin().getDec().asDegrees(),
    )

    J_target = np.array([
        [-8.14327128e-07, -9.05027013e-09],
        [-9.06489594e-09, 8.14380525e-07],
    ])
    lin = wcs_dm.linearizePixelToSky(
        geom.Point2D(32165.0, 19905.0), geom.radians,
    )
    J = np.array(lin.getLinear().getMatrix(), dtype=np.float64)
    np.testing.assert_almost_equal(J, J_target)

    skypos = wcs_galsim.toWorld(galsim.PositionD(x=0, y=0))
    skypos2 = wcs_dm.pixelToSky(geom.Point2D(0, 0))
    np.testing.assert_almost_equal(
        skypos2.getRa().asDegrees(), skypos.ra.deg, decimal=5,
    )
    np.testing.assert_almost_equal(
        skypos2.getDec().asDegrees(), skypos.dec.deg, decimal=5,
    )

    skypos3 = wcs_0.pixelToSky(geom.Point2D(0, 0))
    np.testing.assert_almost_equal(
        skypos3.getRa().asDegrees(), skypos.ra.deg, decimal=5,
    )
    np.testing.assert_almost_equal(
        skypos3.getDec().asDegrees(), skypos.dec.deg, decimal=5,
    )

    J = wcs_galsim.local(
        galsim.PositionD(35530.338170137315, 19914.25926115947)
    ).getMatrix() / 3600.0 / 180 * np.pi
    J_target = np.array([
        [-8.14242694e-07, -1.11992124e-08],
        [-1.12168502e-08, 8.14324975e-07],
    ])
    np.testing.assert_almost_equal(J, J_target, decimal=5)


# ---------------------------------------------------------------------------
# Test: reconstruction / decomposition round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "g1, g2, rho, kappa",
    [
        (0.02, -0.015, 0.01, 0.0),
        (0.05, 0.03, 0.1, 0.02),
        (-0.04, 0.06, -0.05, -0.01),
        (0.0, 0.0, 0.0, 0.0),
        (0.1, -0.1, 0.5, 0.03),
        (-0.15, 0.12, -0.3, 0.05),
        (0.0, 0.0, 1.0, 0.0),
    ],
)
def test_roundtrip(g1, g2, rho, kappa):
    """Decomposition must be the exact inverse of reconstruction."""
    pixel_scale = 0.168
    jac = jacobian_reconstruction(pixel_scale, g1, g2, rho, kappa)
    g1_r, g2_r, rho_r, kappa_r = jacobian_decomposition(
        jac, pixel_scale,
    )
    np.testing.assert_allclose(g1_r, g1, atol=1e-14)
    np.testing.assert_allclose(g2_r, g2, atol=1e-14)
    np.testing.assert_allclose(rho_r, rho, atol=1e-14)
    np.testing.assert_allclose(kappa_r, kappa, atol=1e-14)


# ---------------------------------------------------------------------------
# Test: GalSim JacobianWCS extraction round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "g1, g2, rho",
    [
        (0.025, -0.02, 0.0),
        (0.025, -0.02, 0.4),
        (0.0, 0.0, 0.1),
        (-0.03, 0.05, -0.2),
    ],
)
def test_extract_perturbation_jwcs(g1, g2, rho):
    """extract_perturbation_jwcs must recover make_jwcs inputs."""
    pixel_scale = 0.2
    wcs = make_jwcs(pixel_scale, g1, g2, rho)
    g1_r, g2_r, rho_r, kappa_r = extract_perturbation_jwcs(
        wcs, pixel_scale,
    )
    np.testing.assert_allclose(g1_r, g1, atol=1e-14)
    np.testing.assert_allclose(g2_r, g2, atol=1e-14)
    np.testing.assert_allclose(rho_r, rho, atol=1e-14)
    np.testing.assert_allclose(kappa_r, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Test: GalSim vs LSST shear extraction consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dec_deg", [0, 30, 60])
def test_galsim_vs_dm_extraction(dec_deg):
    """GalSim TanWCS and LSST SkyWcs yield the same perturbations."""
    pixel_scale = 0.168
    center = 100.0
    g1, g2, rho = 0.02, -0.015, 0.01

    wcs_gs = make_tanwcs_galsim(
        pixel_scale, g1, g2, rho, 180.0, dec_deg, center, center,
    )
    wcs_dm = tanwcs_galsim2dm(wcs_gs)

    gs_point = galsim.PositionD(center, center)
    g1_gs, g2_gs, rho_gs, _ = extract_perturbation_galsim_wcs(
        wcs_gs, gs_point, pixel_scale,
    )

    dm_point = geom.Point2D(center, center)
    g1_dm, g2_dm, rho_dm, _ = extract_perturbation_dm_wcs(
        wcs_dm, dm_point, pixel_scale,
    )

    # GalSim TanWCS at tangent point should recover input
    np.testing.assert_allclose(g1_gs, g1, atol=1e-8)
    np.testing.assert_allclose(g2_gs, g2, atol=1e-8)
    np.testing.assert_allclose(rho_gs, rho, atol=1e-8)

    # LSST extraction should agree with GalSim
    np.testing.assert_allclose(g1_dm, g1_gs, atol=1e-6)
    np.testing.assert_allclose(g2_dm, g2_gs, atol=1e-6)
    np.testing.assert_allclose(rho_dm, rho_gs, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: pixel-to-sky consistency between GalSim and LSST WCS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "g1, g2, rho, ra_deg, dec_deg",
    [
        (0.01, 0.02, 0.5, 180.0, 0.0),
        (0.2, -0.3, 0.03, 180.0, 30.0),
        (-0.05, 0.1, -0.1, 180.0, 60.0),
        (0.2, -0.3, 0.2, 180.0, -30.0),
        (-0.05, 0.1, -0.5, 180.0, -60.0),
    ],
)
def test_pixel_to_sky_consistency(g1, g2, rho, ra_deg, dec_deg):
    """GalSim and LSST WCS map the same pixels to the same sky."""
    pixel_scale = 0.168
    center = 100.0

    wcs_gs = make_tanwcs_galsim(
        pixel_scale, g1, g2, rho, ra_deg, dec_deg, center, center,
    )
    wcs_dm = make_tanwcs_dm(
        pixel_scale, g1, g2, rho, ra_deg, dec_deg, center, center,
    )

    npoints = 12
    rng = np.random.default_rng(42)
    x_pix = center + rng.uniform(-50, 50, size=npoints)
    y_pix = center + rng.uniform(-50, 50, size=npoints)

    for i in range(npoints):
        sky_gs = wcs_gs.toWorld(
            galsim.PositionD(x_pix[i], y_pix[i]),
        )
        sky_dm = wcs_dm.pixelToSky(
            geom.Point2D(x_pix[i], y_pix[i]),
        )
        diff_ra = abs(
            sky_gs.ra.rad - sky_dm.getRa().asRadians()
        )
        diff_dec = abs(
            sky_gs.dec.rad - sky_dm.getDec().asRadians()
        )
        assert diff_ra < 1e-12, (
            f"RA mismatch at pixel ({x_pix[i]:.1f}, "
            f"{y_pix[i]:.1f}): {diff_ra:.2e} rad"
        )
        assert diff_dec < 1e-12, (
            f"Dec mismatch at pixel ({x_pix[i]:.1f}, "
            f"{y_pix[i]:.1f}): {diff_dec:.2e} rad"
        )


# ---------------------------------------------------------------------------
# Test: FPFS isotropic galaxy WCS ellipticity correction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wcs_g1, wcs_g2, wcs_rho",
    [
        (0.025, -0.02, 0.4),
        (0.01, 0.03, 0.0),
        (-0.02, 0.015, -0.3),
        (0.0, 0.0, 0.2),
    ],
)
def test_isotropic_galaxy(wcs_g1, wcs_g2, wcs_rho):
    """WCS correction on an isotropic galaxy recovers the no-WCS reference."""
    pixel_scale = 0.2
    stamp_size = 64
    center = stamp_size // 2
    sigma_shapelets = 0.52

    wcs = make_jwcs(pixel_scale, wcs_g1, wcs_g2, wcs_rho)
    wcs_ref = make_jwcs(pixel_scale, 0.0, 0.0, 0.0)

    rec_g1, rec_g2, rec_rho, _ = extract_perturbation_jwcs(wcs, pixel_scale)

    gal = galsim.Exponential(half_light_radius=0.4, flux=1e5)
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0)

    ells = simulate_and_measure(
        gal, psf_obj, wcs, stamp_size, center, pixel_scale, sigma_shapelets,
    )
    ells_ref = simulate_and_measure(
        gal, psf_obj, wcs_ref, stamp_size, center,
        pixel_scale, sigma_shapelets,
    )

    g1_ref = ells_ref["fpfs1_e1"][0] / ells_ref["fpfs1_de1_dg1"][0]
    g2_ref = ells_ref["fpfs1_e2"][0] / ells_ref["fpfs1_de2_dg2"][0]

    e1_corr, e2_corr = correct_ellipticity_wcs(ells, rec_g1, rec_g2, rec_rho)
    g1_corr = e1_corr[0] / ells["fpfs1_de1_dg1"][0]
    g2_corr = e2_corr[0] / ells["fpfs1_de2_dg2"][0]

    np.testing.assert_allclose(g1_corr, g1_ref, atol=5e-4)
    np.testing.assert_allclose(g2_corr, g2_ref, atol=5e-4)

    # Also check linear moment correction via correct_fpfs_spin2_wcs
    data = simulate_and_measure_linear(
        gal, psf_obj, wcs, stamp_size, center, pixel_scale, sigma_shapelets,
    )
    data_ref = simulate_and_measure_linear(
        gal, psf_obj, wcs_ref, stamp_size, center,
        pixel_scale, sigma_shapelets,
    )
    corrected = correct_fpfs_spin2_wcs(
        data, rec_g1, rec_g2, rec_rho, prefix="fpfs1_",
    )
    for col in ["fpfs1_m22c", "fpfs1_m22s", "fpfs1_m42c", "fpfs1_m42s"]:
        np.testing.assert_allclose(
            corrected[col][0], data_ref[col][0], atol=1.0,
            err_msg=f"{col} mismatch",
        )


# ---------------------------------------------------------------------------
# Test: FPFS sheared galaxy quartet (45-degree rotation cancellation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wcs_g1, wcs_g2, wcs_rho, gal_g1, gal_g2",
    [
        (0.011, 0.022, 0.4, -0.34, 0.21),
        (0.025, -0.02, 0.0, 0.15, -0.10),
        (-0.015, 0.01, -0.2, 0.20, 0.25),
    ],
)
def test_sheared_galaxy(wcs_g1, wcs_g2, wcs_rho, gal_g1, gal_g2):
    """4 galaxies at 45*n-degree rotations: corrected avg ≈ 0."""
    pixel_scale = 0.2
    stamp_size = 64
    center = stamp_size // 2
    sigma_shapelets = 0.52

    wcs = make_jwcs(pixel_scale, wcs_g1, wcs_g2, wcs_rho)
    wcs_ref = make_jwcs(pixel_scale, 0.0, 0.0, 0.0)

    rec_g1, rec_g2, rec_rho, _ = extract_perturbation_jwcs(wcs, pixel_scale)

    gal_base = galsim.Exponential(half_light_radius=0.4, flux=1e5).shear(
        g1=gal_g1, g2=gal_g2,
    )
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0)

    angles = [0.0, 45.0, 90.0, 135.0]
    n_gal = len(angles)
    e1_ref_sum = 0.0
    e2_ref_sum = 0.0
    e1_raw_sum = 0.0
    e2_raw_sum = 0.0
    e1_corr_sum = 0.0
    e2_corr_sum = 0.0

    for ang in angles:
        gal = gal_base.rotate(ang * galsim.degrees)

        ells = simulate_and_measure(
            gal, psf_obj, wcs, stamp_size, center,
            pixel_scale, sigma_shapelets,
        )
        ells_ref = simulate_and_measure(
            gal, psf_obj, wcs_ref, stamp_size, center,
            pixel_scale, sigma_shapelets,
        )

        e1_ref_sum += ells_ref["fpfs1_e1"][0]
        e2_ref_sum += ells_ref["fpfs1_e2"][0]
        e1_raw_sum += ells["fpfs1_e1"][0]
        e2_raw_sum += ells["fpfs1_e2"][0]

        e1_c, e2_c = correct_ellipticity_wcs(ells, rec_g1, rec_g2, rec_rho)
        e1_corr_sum += e1_c[0]
        e2_corr_sum += e2_c[0]

    e1_ref_avg = e1_ref_sum / n_gal
    e2_ref_avg = e2_ref_sum / n_gal
    e1_raw_avg = e1_raw_sum / n_gal
    e2_raw_avg = e2_raw_sum / n_gal
    e1_corr_avg = e1_corr_sum / n_gal
    e2_corr_avg = e2_corr_sum / n_gal

    # Reference (no WCS distortion) average should be ~0
    np.testing.assert_allclose(e1_ref_avg, 0.0, atol=1e-2)
    np.testing.assert_allclose(e2_ref_avg, 0.0, atol=1e-2)

    # Corrected average should be ~0
    np.testing.assert_allclose(e1_corr_avg, 0.0, atol=1e-2)
    np.testing.assert_allclose(e2_corr_avg, 0.0, atol=1e-2)

    # Raw (uncorrected) average should NOT be ~0 (WCS distortion present)
    raw_mag = np.sqrt(e1_raw_avg**2 + e2_raw_avg**2)
    assert raw_mag > 1e-3, (
        f"Raw avg too small ({raw_mag:.2e}), WCS distortion not visible"
    )

    # Also check linear moment correction on a single sheared galaxy
    gal_single = gal_base
    data = simulate_and_measure_linear(
        gal_single, psf_obj, wcs, stamp_size, center,
        pixel_scale, sigma_shapelets,
    )
    data_ref = simulate_and_measure_linear(
        gal_single, psf_obj, wcs_ref, stamp_size, center,
        pixel_scale, sigma_shapelets,
    )
    corrected = correct_fpfs_spin2_wcs(
        data, rec_g1, rec_g2, rec_rho, prefix="fpfs1_",
    )
    for col in ["fpfs1_m22c", "fpfs1_m22s", "fpfs1_m42c", "fpfs1_m42s"]:
        np.testing.assert_allclose(
            corrected[col][0], data_ref[col][0],
            rtol=5e-2,
            err_msg=f"{col} mismatch",
        )
