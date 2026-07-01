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

"""SciPIC-exact broadband AB-magnitude synthesis from Flagship 2 catalog rows.

Reproduces the line-aware (`*_el_model3_ext`) broadband photometry stored in
`flagship_cosmos.fits` from the SED-component columns, ``abs_mag_r01``, and
the value-added data (COSMOS SED templates, extinction curves, filter
passbands).

Recipe (mirrors SciPIC's public excerpt ``ptallada/scipic_fluxes``):

  Per component i = 1, 2:
    T_i        = T[sed_cosmos_i] * (EXT[c_i] / EXT[0]) ^ (ebv_i / 0.2)
    rest_r01_i = ∫ T_i * F_r01 * λ dλ / (c * ∫ F_r01 / λ dλ)
    obs_band_i = ∫ T_i(λ/(1+z_obs)) / (1+z_obs) * F_band * λ dλ
                    / (c * ∫ F_band / λ dλ)
  Mix (both components share the same absolute-mag scaling):
    band_cont = (frac_1 * obs_band_1/rest_r01_1
                + frac_2 * obs_band_2/rest_r01_2)
              * 10^(-0.4 * (abs_mag_r01_phys + DM(z_true) + 48.6))
  Add emission lines (delta functions at observer-frame λ, filter-weighted):
    band_total = band_cont
               + Σ_lines mult * 10^logf * F_band(λ_obs) * λ_obs / DEN_band

Notes:
  * ``abs_mag_r01`` in the catalog is stored in h-free units
    (``M_phys - 5·log10(h)``); we add ``5·log10(h)`` back before use.
  * The 11-entry emission-line list is fixed: the 9 catalog
    ``logf_*_model3_ext`` columns plus the atomic-physics doublet
    counterparts ``[OIII] 4959 = (1/3)·[OIII] 5007`` and
    ``[NII] 6548 = (1/3)·[NII] 6584``.
  * SciPIC uses ``observed_redshift_gal`` for the SED redshifting and
    emission-line ``λ_obs``, and ``redshift`` (= ``true_redshift_gal``) for
    the luminosity distance; this function does the same.
  * No IGM applied. Valid to z ≲ 1.5 for lsst_u/g; other bands are IGM-free
    at all z ≤ 3.
"""

from __future__ import annotations

import glob
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

# `np.trapezoid` is the modern name (NumPy >= 2.0); `np.trapz` is the
# legacy alias (removed in later 2.x). Pick whichever exists.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Speed of light in Å/s (for the photon-counting AB denominator).
_C_AA = 2.99792458e18

# Standard cgs AB magnitude zero-point (mag = -2.5·log10(f_ν) - 48.6, with
# f_ν in erg/s/cm²/Hz). Used internally to convert the physical f_ν the
# recipe produces into the caller's `mag_zero` convention.
_AB_ZP_CGS = 48.6

# FS2 cosmology (Castander+2024 Sect. 3): Planck-like flat ΛCDM, h=0.67.
_COSMO = FlatLambdaCDM(H0=67.0, Om0=0.319, Tcmb0=2.7255)

# The catalog's ``abs_mag_r01`` is stored in h-free units
# (M_phys - 5·log10(h)); add this offset back to recover physical M.
_M_R01_H_OFFSET = 5.0 * np.log10(_COSMO.h)   # ≈ -0.87 for h=0.67

# Fixed reference band for per-component normalization: SciPIC uses
# ``sdss_r01`` (SDSS r shifted to z=0.1) since the catalog's
# ``abs_mag_r01`` column is defined against it.
_REF_BAND = "sdss_r01"

# Fixed emission-line list: (catalog column, rest λ Å, flux multiplier).
# The catalog stores only the STRONG line of each fine-structure doublet
# for [OIII] and [NII]; SciPIC's photometry pipeline internally adds the
# weaker doublet companion at the atomic-physics ratio of 1:3
# ([OIII] 4959 = (1/3)·[OIII] 5007; [NII] 6548 = (1/3)·[NII] 6584).
_LINES = (
    ("logf_o2_model3_ext",     3727.4, 1.0),
    ("logf_hdelta_model3_ext", 4101.7, 1.0),
    ("logf_hgamma_model3_ext", 4340.5, 1.0),
    ("logf_hbeta_model3_ext",  4861.3, 1.0),
    ("logf_o3_model3_ext",     4958.9, 1.0 / 3.0),   # [OIII] 4959
    ("logf_o3_model3_ext",     5006.8, 1.0),
    ("logf_halpha_model3_ext", 6562.8, 1.0),
    ("logf_n2_model3_ext",     6548.0, 1.0 / 3.0),   # [NII] 6548
    ("logf_n2_model3_ext",     6583.5, 1.0),
    ("logf_s2_model3_ext",     6723.5, 1.0),
    ("logf_s3_model3_ext",     9300.0, 1.0),
)


def _default_vad_dir():
    env = os.environ.get("CATSIM_VAD_DIR")
    if env:
        return env
    return (
        "/gpfs/mnt/gpfs02/astro/astro_desc/data/simulation/input_catalog/"
        "catsim_flagship_scripts/flagship_data"
    )


def _load_csv(path):
    return np.loadtxt(path, delimiter=",")


@lru_cache(maxsize=8)
def _load_templates(vad_dir):
    """Return (WL_REST, TPL) — rest-frame wavelengths and 31 COSMOS
    SED template f_λ values on that grid."""
    seds = sorted(
        glob.glob(str(Path(vad_dir) / "galaxy_seds" / "*.csv")),
        key=lambda p: int(Path(p).name.split("_")[0]),
    )
    if not seds:
        raise FileNotFoundError(
            f"no SED templates found under {vad_dir}/galaxy_seds/"
        )
    wl = _load_csv(seds[0])[:, 0]
    tpl = np.vstack([_load_csv(p)[:, 1] for p in seds])
    return wl, tpl


@lru_cache(maxsize=8)
def _load_ext_ratios(vad_dir):
    """Return ``EXT_RATIO[i] = ext_i(λ) / ext_0(λ)``, interpolated to
    the template rest-wavelength grid. Used as
    ``A(λ) = EXT_RATIO[idx] ** (E(B-V) / 0.2)``.
    """
    ext_files = sorted(
        glob.glob(str(Path(vad_dir) / "galaxy_extincts" / "*.csv")),
        key=lambda p: int(Path(p).name.split("_")[0]),
    )
    if not ext_files:
        raise FileNotFoundError(
            f"no extinction curves under {vad_dir}/galaxy_extincts/"
        )
    wl_rest, _ = _load_templates(vad_dir)
    raw = [_load_csv(p) for p in ext_files]
    ref = raw[0]
    return np.vstack([
        np.interp(
            wl_rest, e[:, 0],
            e[:, 1] / np.interp(e[:, 0], ref[:, 0], ref[:, 1]),
            left=1.0, right=1.0,
        )
        for e in raw
    ])


@lru_cache(maxsize=256)
def _load_filter(vad_dir, name):
    """Return ``(fw, ft, den)`` for a given filter name.

    ``fw`` and ``ft`` are 1D arrays (wavelength Å, transmission); ``den``
    is the photon-counting AB denominator ``∫ T · c / λ dλ``.
    """
    p = Path(vad_dir) / "filters" / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"filter file not found: {p}")
    a = _load_csv(p)
    fw = a[:, 0]
    ft = a[:, 1]
    den = float(_trapz(ft * _C_AA / fw, fw))
    return fw, ft, den


def _rest_band(tpl, wl_rest, fw, ft, den):
    flam = np.interp(fw, wl_rest, tpl, left=0.0, right=0.0)
    return float(_trapz(flam * ft * fw, fw)) / den


def compute_magnitudes(bands, row, vad_dir=None):
    """Compute SciPIC-exact broadband AB magnitudes for one catalog row.

    Parameters
    ----------
    bands : iterable of str
        Filter names to compute (e.g. ``['lsst_u', 'lsst_g',
        'euclid_vis']``). Each name must correspond to a CSV in
        ``<vad_dir>/filters/{name}.csv``.
    row : numpy.void or dict-like
        A single row from ``flagship_cosmos.fits`` (accessible by column
        name). Required columns: ``observed_redshift_gal``, ``redshift``
        (renamed from ``true_redshift_gal`` by ``prepare_flagship.py``),
        ``abs_mag_r01``, ``sed_cosmos_{1,2}``, ``ext_curve_cosmos_{1,2}``,
        ``ebv_cosmos_{1,2}``, ``frac_cosmos_{1,2}``, and the 9
        ``logf_*_model3_ext`` emission-line columns.
    vad_dir : str or None, optional
        Path to the value-added-data directory containing ``filters/``,
        ``galaxy_seds/`` and ``galaxy_extincts/``. Defaults to
        ``$CATSIM_VAD_DIR`` if set, otherwise the standard PIC location.

    Returns
    -------
    dict[str, float]
        Mapping ``band → AB magnitude`` (physical, cgs zero-point 48.6)
        for each requested band. Value is ``np.nan`` if the synthesis is
        undefined (e.g. both templates return zero flux in the reference
        band).
    """
    if vad_dir is None:
        vad_dir = _default_vad_dir()
    vad_dir = str(vad_dir)

    z_obs = float(row["observed_redshift_gal"])
    z_true = float(row["redshift"])
    mag_r01 = float(row["abs_mag_r01"]) + _M_R01_H_OFFSET

    ebv1 = float(row["ebv_cosmos_1"])
    ebv2 = float(row["ebv_cosmos_2"])
    frac1 = float(row["frac_cosmos_1"])
    frac2 = float(row["frac_cosmos_2"])
    sed1 = int(row["sed_cosmos_1"])
    sed2 = int(row["sed_cosmos_2"])
    ext1 = int(row["ext_curve_cosmos_1"])
    ext2 = int(row["ext_curve_cosmos_2"])

    wl_rest, tpl = _load_templates(vad_dir)
    ext_ratio = _load_ext_ratios(vad_dir)

    att1 = ext_ratio[ext1] ** (ebv1 / 0.2) if ebv1 > 0 else 1.0
    att2 = ext_ratio[ext2] ** (ebv2 / 0.2) if ebv2 > 0 else 1.0
    tpl1 = tpl[sed1] * att1
    tpl2 = tpl[sed2] * att2

    ref_fw, ref_ft, ref_den = _load_filter(vad_dir, _REF_BAND)
    rest_r01_1 = _rest_band(tpl1, wl_rest, ref_fw, ref_ft, ref_den)
    rest_r01_2 = _rest_band(tpl2, wl_rest, ref_fw, ref_ft, ref_den)
    if rest_r01_1 <= 0 or rest_r01_2 <= 0:
        return {b: float("nan") for b in bands}

    d_l_pc = _COSMO.luminosity_distance(z_true).to_value("pc")
    dm = 5.0 * (np.log10(d_l_pc) - 1.0)
    # Physical-cgs f_ν scale for the mixed SED at the observed distance.
    scale = 10.0 ** (-0.4 * (mag_r01 + dm + _AB_ZP_CGS))

    out = {}
    for band in bands:
        fw, ft, den = _load_filter(vad_dir, band)

        flam1 = np.interp(
            fw / (1.0 + z_obs), wl_rest, tpl1, left=0.0, right=0.0,
        ) / (1.0 + z_obs)
        flam2 = np.interp(
            fw / (1.0 + z_obs), wl_rest, tpl2, left=0.0, right=0.0,
        ) / (1.0 + z_obs)
        obs_band_1 = float(_trapz(flam1 * ft * fw, fw)) / den
        obs_band_2 = float(_trapz(flam2 * ft * fw, fw)) / den

        f_nu = (
            frac1 * obs_band_1 / rest_r01_1
            + frac2 * obs_band_2 / rest_r01_2
        ) * scale

        # Emission lines: observer-frame deltas × filter T × λ / DEN.
        for col, lam0, mult in _LINES:
            lam = lam0 * (1.0 + z_obs)
            if fw[0] <= lam <= fw[-1]:
                t_lam = float(np.interp(lam, fw, ft))
                f_nu += mult * 10.0 ** float(row[col]) * t_lam * lam / den

        out[band] = (
            -2.5 * np.log10(f_nu) - _AB_ZP_CGS if f_nu > 0 else float("nan")
        )
    return out
