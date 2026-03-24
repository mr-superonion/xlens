"""Test that shapelets_linear2ell reproduces the nonlinear
process_image output for both noiseless and noisy simulations."""

import anacal
import galsim
import numpy as np

from xlens.catalog.utils import shapelets_linear2ell


def _setup_sim(g1=0.0, g2=0.0, noise_var=0.0, seed=42):
    """Simulate a galaxy and return arrays needed for FPFS measurement."""
    stamp_size = 64
    pixel_scale = 0.2
    center = stamp_size / 2.0 + 0.5

    gal = galsim.Gaussian(sigma=0.3, flux=500)
    if g1 != 0.0 or g2 != 0.0:
        gal = gal.shear(g1=g1, g2=g2)
    psf = galsim.Gaussian(sigma=0.26)
    conv = galsim.Convolve(gal, psf)
    wcs = galsim.PixelScale(pixel_scale)

    gal_img = conv.drawImage(
        nx=stamp_size, ny=stamp_size, wcs=wcs,
        center=galsim.PositionD(center, center),
    )
    psf_img = psf.drawImage(
        nx=stamp_size, ny=stamp_size, wcs=wcs,
        center=galsim.PositionD(center, center),
    )

    rng = np.random.default_rng(seed)
    if noise_var > 0:
        noise_array = rng.normal(
            0, np.sqrt(noise_var), (stamp_size, stamp_size)
        )
    else:
        noise_array = None

    det_dtype = np.dtype([("y", np.float64), ("x", np.float64)])
    det = np.array([(center, center)], dtype=det_dtype)

    return {
        "gal_array": np.asarray(gal_img.array, np.float64),
        "psf_array": np.asarray(psf_img.array, np.float64),
        "noise_array": noise_array,
        "noise_var": noise_var,
        "det": det,
        "pixel_scale": pixel_scale,
    }


def _compare(sim, fpfs_config):
    """Run both nonlinear and linear paths, compare via
    shapelets_linear2ell."""
    C0 = fpfs_config.c0
    prefix = "fpfs1_"

    common = dict(
        fpfs_config=fpfs_config,
        pixel_scale=sim["pixel_scale"],
        mag_zero=30.0,
        noise_variance=sim["noise_var"],
        gal_array=sim["gal_array"],
        psf_array=sim["psf_array"],
        noise_array=sim["noise_array"],
        mask_array=None,
        detection=sim["det"],
        do_compute_detect_weight=False,
    )

    # Nonlinear (ground truth)
    nonlin = anacal.fpfs.process_image(**common)

    # Linear modes
    lin = anacal.fpfs.process_image(
        **common,
        return_only_linear_modes=True,
        pack_linear_modes=True,
    )

    # Convert linear → ellipticity
    ell = shapelets_linear2ell(lin, C0=C0, prefix=prefix)

    # Compare all shared fields
    fields = [
        "e1", "de1_dg1", "de1_dg2",
        "e2", "de2_dg1", "de2_dg2",
    ]
    for f in fields:
        np.testing.assert_allclose(
            ell[f"{prefix}{f}"],
            nonlin[f"{prefix}{f}"],
            atol=1e-14,
            err_msg=f"Mismatch in {prefix}{f}",
        )


def test_shapelets_linear2ell_noiseless():
    """Noiseless, sheared galaxy."""
    fpfs_config = anacal.fpfs.FpfsConfig(sigma_shapelets1=0.52)
    sim = _setup_sim(g1=0.05, g2=0.03, noise_var=0.0)
    _compare(sim, fpfs_config)


def test_shapelets_linear2ell_noisy():
    """Noisy, sheared galaxy with noise-bias correction."""
    fpfs_config = anacal.fpfs.FpfsConfig(sigma_shapelets1=0.52)
    sim = _setup_sim(g1=0.05, g2=0.03, noise_var=0.5)
    _compare(sim, fpfs_config)


def test_shapelets_linear2ell_round_noisy():
    """Noisy, round galaxy (e ≈ 0)."""
    fpfs_config = anacal.fpfs.FpfsConfig(sigma_shapelets1=0.52)
    sim = _setup_sim(g1=0.0, g2=0.0, noise_var=0.3)
    _compare(sim, fpfs_config)
