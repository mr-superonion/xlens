# This file is part of xlens.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Analytic magnitude derivative / shear-response checks for catalog.utils."""

import numpy as np

from xlens.catalog.utils import (
    MAG_CAP,
    MAG_ERR_FAC,
    MAG_KNEE,
    dmag_dflux,
    flux_to_mag,
    mag_shear_response,
)

MAG_ZERO = 31.4


def _flux_grid():
    """Fluxes spanning bright -> knee -> cap -> 0 -> negative (nJy)."""
    f_knee = 10.0 ** ((MAG_ZERO - MAG_KNEE) / 2.5)
    f_cap = 10.0 ** ((MAG_ZERO - MAG_CAP) / 2.5)
    bright = np.logspace(np.log10(f_knee) + 3.0, np.log10(f_knee) + 0.01, 40)
    roll = np.logspace(np.log10(f_knee), np.log10(f_cap), 200)
    faint = np.array([f_cap, 0.5 * f_cap, 0.0, -0.5, -5.0])
    return np.concatenate([bright, roll, faint])


def test_dmag_dflux_matches_finite_difference():
    """Analytic dmag/dflux == central finite difference of flux_to_mag."""
    flux = _flux_grid()
    ana = dmag_dflux(flux, MAG_ZERO)

    # central FD with a per-point relative step (fluxes span many decades)
    eps = np.maximum(np.abs(flux), 1e-3) * 1e-6
    m_plus, _ = flux_to_mag(flux + eps, MAG_ZERO)
    m_minus, _ = flux_to_mag(flux - eps, MAG_ZERO)
    fd = (m_plus - m_minus) / (2.0 * eps)

    # compare only where the FD stencil stays on one smooth branch (avoid the
    # flux=0 kink where f-eps crosses to negative flux)
    ok = flux - eps > 0
    np.testing.assert_allclose(ana[ok], fd[ok], rtol=1e-5, atol=1e-9)

    # exactly zero for non-positive flux (mag pinned at the cap)
    assert np.all(dmag_dflux(np.array([0.0, -1.0, -100.0]), MAG_ZERO) == 0.0)

    # exactly zero deep in the saturated band (mag == m_cap there)
    f_below_cap = 10.0 ** ((MAG_ZERO - (MAG_CAP + 0.5)) / 2.5)
    assert dmag_dflux(np.array([f_below_cap]), MAG_ZERO)[0] == 0.0


def test_bright_end_is_plain_log():
    """Brighter than the knee, dmag/dflux is the plain-log -MAG_ERR_FAC/flux."""
    f_knee = 10.0 ** ((MAG_ZERO - MAG_KNEE) / 2.5)
    flux = f_knee * np.array([2.0, 10.0, 1e3, 1e5])  # all brighter than the knee
    ana = dmag_dflux(flux, MAG_ZERO)
    np.testing.assert_allclose(ana, -MAG_ERR_FAC / flux, rtol=1e-12, atol=0.0)


def test_dmag_dflux_extinction_offset_invariant():
    """A constant per-band extinction shifts mag but not dmag/dflux."""
    flux = _flux_grid()
    a_ext = np.full_like(flux, 0.3)
    # away from the truncation the derivative is extinction-independent; compare
    # on the bright branch where both are the plain-log form
    ok = flux > 10.0 ** ((MAG_ZERO - (MAG_KNEE - 1.0)) / 2.5)
    d0 = dmag_dflux(flux, MAG_ZERO)
    da = dmag_dflux(flux, MAG_ZERO, a_ext=a_ext)
    np.testing.assert_allclose(da[ok], d0[ok], rtol=1e-12, atol=0.0)


def test_mag_shear_response():
    """dmag_dg / dsigma_m_dg identities and a flux-direction finite difference."""
    rng = np.random.RandomState(42)
    flux = _flux_grid()
    n = len(flux)
    dflux_dg1 = rng.normal(size=n)
    dflux_dg2 = rng.normal(size=n)
    flux_err = np.full(n, 0.7)
    a_ext = np.full(n, 0.1)

    mag, sigma_m, dmag_dg1, dmag_dg2, dsig_dg1, dsig_dg2 = mag_shear_response(
        flux, dflux_dg1, dflux_dg2, MAG_ZERO, flux_err=flux_err, a_ext=a_ext
    )

    # mag / sigma_m identical to flux_to_mag
    mag_ref, sig_ref = flux_to_mag(flux, MAG_ZERO, flux_err=flux_err, a_ext=a_ext)
    np.testing.assert_array_equal(mag, mag_ref)
    np.testing.assert_array_equal(sigma_m, sig_ref)

    # chain-rule identities
    dm_df = dmag_dflux(flux, MAG_ZERO, a_ext=a_ext)
    np.testing.assert_allclose(dmag_dg1, dm_df * dflux_dg1, rtol=0, atol=0)
    np.testing.assert_allclose(dmag_dg2, dm_df * dflux_dg2, rtol=0, atol=0)
    np.testing.assert_allclose(
        dsig_dg1, np.log(10.0) / 2.5 * sigma_m * dmag_dg1, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(dsig_dg1, sigma_m * dmag_dg1 / MAG_ERR_FAC, rtol=1e-12, atol=0)

    # dmag_dg1 == central FD of flux_to_mag along the dflux_dg1 direction.
    # Use a flux-relative step and skip the deep truncation tail (flux < mag 38),
    # where the smoothstep curvature makes any fixed-step FD unreliable.
    eps = 1e-6
    m_plus, _ = flux_to_mag(flux + eps * dflux_dg1, MAG_ZERO, a_ext=a_ext)
    m_minus, _ = flux_to_mag(flux - eps * dflux_dg1, MAG_ZERO, a_ext=a_ext)
    fd = (m_plus - m_minus) / (2.0 * eps)
    f_38 = 10.0 ** ((MAG_ZERO - 38.0) / 2.5)
    ok = (flux - eps * np.abs(dflux_dg1) > 0) & (flux > f_38)
    np.testing.assert_allclose(dmag_dg1[ok], fd[ok], rtol=1e-4, atol=1e-7)


def test_mag_shear_response_no_flux_err():
    """flux_err=None -> sigma_m and its responses are None; dmag_dg still returned."""
    flux = _flux_grid()
    n = len(flux)
    out = mag_shear_response(flux, np.ones(n), np.ones(n), MAG_ZERO)
    mag, sigma_m, dmag_dg1, dmag_dg2, dsig_dg1, dsig_dg2 = out
    assert sigma_m is None and dsig_dg1 is None and dsig_dg2 is None
    assert mag.shape == flux.shape
    np.testing.assert_array_equal(dmag_dg1, dmag_dflux(flux, MAG_ZERO))
