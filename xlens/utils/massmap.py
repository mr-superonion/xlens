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

"""Mass-map utilities.

Aperture-mass (Schirmer)
------------------------
schirmer_filter(radius, aperture_size, x_cut=0.15, a=6, b=150, c=47, d=50, *_)
    Schirmer Q-filter, tuned for NFW-like structures.
compute_mass_map(x_grid, y_grid, x, y, g1, g2, weights, q_filter,
                 filter_kwargs={})
    Per-grid-point aperture-mass E/B mode and variance.
build_flat_wcs_grid(ra_bcg, dec_bcg, ra, dec, n=161, half_extent_deg=0.16)
    Convenience: tangent-plane WCS centred on (ra_bcg, dec_bcg) →
    returns the source `(x, y)` (deg) and the `(x_grid, y_grid)` mesh.
"""
import numpy as np


def schirmer_filter(radius, aperture_size=8000, x_cut=0.15,
                    a=6, b=150, c=47, d=50, *_):
    """The Schirmer Q-filter."""
    x = radius / aperture_size
    return (1 / (1 + np.exp(a - b * x) + np.exp(-c + d * x))) * (
        np.tanh(x / x_cut) / (x / x_cut)
    )


def compute_mass_map(x_grid, y_grid, x, y, g1, g2, weights, q_filter,
                     filter_kwargs=None):
    """Aperture-mass E/B-mode and variance on an NxM grid.

    Parameters
    ----------
    x_grid, y_grid : ndarray
        2-D meshgrid at which to evaluate the aperture mass.
    x, y : ndarray
        Source coordinates (same units as `x_grid`).
    g1, g2 : ndarray
        Source shears.
    weights : ndarray
        Per-source weights.
    q_filter : callable
        Radial filter function `q(radius, **filter_kwargs)`.
    filter_kwargs : dict, optional
        Passed through to `q_filter`. Must include `aperture_size`.

    Returns
    -------
    Map_E, Map_B, Map_V : ndarray
        E-mode, B-mode, and variance aperture-mass maps.
    """
    if filter_kwargs is None:
        filter_kwargs = {}
    ny, nx = y_grid.shape
    Map_E = np.zeros((ny, nx))
    Map_B = np.zeros((ny, nx))
    Map_V = np.zeros((ny, nx))
    if "aperture_size" not in filter_kwargs:
        filter_area = np.pi * 0.55 ** 2
    else:
        filter_area = np.pi * filter_kwargs["aperture_size"] ** 2

    comp_shear = g1 + 1j * g2
    g_mag = np.mean(g1 ** 2 + g2 ** 2) * np.ones_like(g1)
    weight_sum = np.sum(weights)
    for i in range(ny):
        for j in range(nx):
            dx = x - x_grid[i, j]
            dy = y - y_grid[i, j]
            radius = np.sqrt(dx ** 2 + dy ** 2)
            theta = np.arctan2(dy, dx)
            rotated = comp_shear * -np.exp(-2j * theta)
            g_T = rotated.real
            g_X = rotated.imag
            qv = q_filter(radius, **filter_kwargs)
            Map_E[i, j] = np.sum(qv * g_T * weights) * filter_area / weight_sum
            Map_B[i, j] = np.sum(qv * g_X * weights) * filter_area / weight_sum
            Map_V[i, j] = (
                np.sum((qv ** 2) * g_mag * (weights ** 2))
                * (filter_area ** 2)
                / (2 * weight_sum ** 2)
            )
    return Map_E, Map_B, Map_V


def compute_mass_map_healpix(*, ra_center, dec_center, radius_deg,
                             src_ra, src_dec, g1, g2, weights,
                             q_filter, filter_kwargs=None,
                             nside, nest=True, verbose=False):
    """Curved-sky Schirmer aperture-mass on a HEALPix grid.

    Same Q-filter-weighted average as :func:`compute_mass_map`, but
    every source-to-pixel distance is the great-circle arc length on
    the sphere and the spin-2 rotation uses the spherical position
    angle. Avoids the gnomonic-projection inaccuracy that grows with
    sky offset.

    Parameters
    ----------
    ra_center, dec_center : float
        Cone centre in degrees. The function returns values for every
        HEALPix pixel within ``radius_deg`` of this point.
    radius_deg : float
        Cone radius in degrees.
    src_ra, src_dec : ndarray (deg)
        Source positions on the sphere.
    g1, g2 : ndarray
        Source shears (calibrated; same convention as
        :func:`compute_mass_map`).
    weights : ndarray
        Per-source weights.
    q_filter : callable
        Radial filter ``q(radius_deg, **filter_kwargs)``.
    filter_kwargs : dict, optional
        Passed to ``q_filter``. Must include ``aperture_size`` (deg).
    nside : int
        HEALPix resolution.
    nest : bool
        HEALPix scheme; True = NESTED (default; matches HealSparse).
    verbose : bool
        If True, print a progress dot every 10 % of pixels.

    Returns
    -------
    ipix : ndarray of int
        HEALPix pixel ids covered (length N_pix).
    Map_E, Map_B, Map_V : ndarray (length N_pix)
        E-mode, B-mode and variance aperture masses at each pixel.
    """
    import healpy as hp

    if filter_kwargs is None:
        filter_kwargs = {}
    if "aperture_size" not in filter_kwargs:
        filter_area = np.pi * 0.55 ** 2
    else:
        filter_area = np.pi * filter_kwargs["aperture_size"] ** 2

    # 1. HEALPix pixels inside the disc.
    vec = hp.ang2vec(ra_center, dec_center, lonlat=True)
    ipix = hp.query_disc(nside, vec, np.radians(radius_deg),
                         inclusive=True, nest=nest)
    pix_ra, pix_dec = hp.pix2ang(nside, ipix, lonlat=True, nest=nest)
    n_pix = len(ipix)
    Map_E = np.zeros(n_pix)
    Map_B = np.zeros(n_pix)
    Map_V = np.zeros(n_pix)

    # 2. Precompute source quantities (radians).
    src_ra_rad = np.radians(src_ra)
    src_dec_rad = np.radians(src_dec)
    cos_src_dec = np.cos(src_dec_rad)
    sin_src_dec = np.sin(src_dec_rad)
    comp_shear = g1 + 1j * g2
    g_mag = np.mean(g1 ** 2 + g2 ** 2) * np.ones_like(g1)
    weight_sum = np.sum(weights)

    pix_ra_rad = np.radians(pix_ra)
    pix_dec_rad = np.radians(pix_dec)
    cos_pix_dec = np.cos(pix_dec_rad)
    sin_pix_dec = np.sin(pix_dec_rad)

    # 3. Per-pixel: vectorize over sources.
    progress_step = max(1, n_pix // 10)
    for k in range(n_pix):
        # Great-circle distance via haversine.
        dra = src_ra_rad - pix_ra_rad[k]
        sin_dd_half = np.sin((src_dec_rad - pix_dec_rad[k]) / 2.0)
        sin_dra_half = np.sin(dra / 2.0)
        a = (sin_dd_half ** 2
             + cos_pix_dec[k] * cos_src_dec * sin_dra_half ** 2)
        np.clip(a, 0.0, 1.0, out=a)
        d_rad = 2.0 * np.arcsin(np.sqrt(a))
        d_deg = np.degrees(d_rad)

        # Position angle at pixel pointing to source (E from N).
        sin_dra = np.sin(dra)
        cos_dra = np.cos(dra)
        num = sin_dra * cos_src_dec
        denom = (cos_pix_dec[k] * sin_src_dec
                 - sin_pix_dec[k] * cos_src_dec * cos_dra)
        pa = np.arctan2(num, denom)

        # Spin-2 rotation. Convention matches `compute_mass_map`:
        # using its (x=west, y=north) tangent axes, `theta = pa + pi/2`
        # and `-shear * exp(-2i*theta) = shear * exp(-2i*pa)`.
        rotated = comp_shear * np.exp(-2j * pa)
        g_T = rotated.real
        g_X = rotated.imag
        qv = q_filter(d_deg, **filter_kwargs)

        Map_E[k] = np.sum(qv * g_T * weights) * filter_area / weight_sum
        Map_B[k] = np.sum(qv * g_X * weights) * filter_area / weight_sum
        Map_V[k] = (np.sum((qv ** 2) * g_mag * (weights ** 2))
                    * (filter_area ** 2) / (2.0 * weight_sum ** 2))

        if verbose and (k % progress_step == 0):
            import sys
            print(f"#   curved-sky map: {k}/{n_pix}", file=sys.stderr)

    return ipix, Map_E, Map_B, Map_V


def build_flat_wcs_grid(ra_bcg, dec_bcg, ra, dec,
                        n=161, half_extent_deg=0.16):
    """Project sources to flat-sky degrees relative to (ra_bcg, dec_bcg) and
    return both the source `(x, y)` and an NxN `(x_grid, y_grid)` mesh.
    """
    from astropy.wcs import WCS
    from astropy.wcs.utils import skycoord_to_pixel
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    wcs = WCS(naxis=2)
    wcs.wcs.crval = [ra_bcg, dec_bcg]
    wcs.wcs.crpix = [0, 0]
    wcs.wcs.cdelt = [-0.2 / 3600, 0.2 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.equinox = 2000
    wcs.wcs.cd = [[-0.2 / 3600, 0], [0, 0.2 / 3600]]

    bcg = SkyCoord(ra_bcg * u.deg, dec_bcg * u.deg)
    x0, y0 = skycoord_to_pixel(bcg, wcs, origin=0)
    src = SkyCoord(ra=ra, dec=dec, unit="degree")
    x_pix, y_pix = skycoord_to_pixel(src, wcs, origin=0)
    scale = 0.2 / 3600  # deg / pix
    x = (x_pix - x0) * scale
    y = (y_pix - y0) * scale

    s = np.linspace(-half_extent_deg, half_extent_deg, n)
    x_grid, y_grid = np.meshgrid(s, s, indexing="xy")
    return x, y, x_grid, y_grid
