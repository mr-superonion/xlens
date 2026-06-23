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

"""Tangential / cross shear estimation utilities.

Functions
---------
calibrate_shapes(table, c0=50.0, weights=None, normalize=True)
    Combine FPFS moments across `griz` -> (e1, e2, response).
fast_bootstrap_mean(data, ci_level=0.95)
    SciPy-bootstrap confidence interval of the mean.
anacal_get_tang_cross(cluster, sky_dist, bins, R, ci_level=0.95,
                      r_correction_offset=0.0, verbose=False)
    Bin AnaCal complex tangential/cross shear in radius and return the
    response-corrected means + bootstrap CIs.
"""
import numpy as np

# Default FPFS band weights (kept in sync with the griz `band_weights`
# block in `configs/measure_pipeline_*bands.yaml` so step3's recomputed
# combine matches what mergePatches writes to the merged catalog).
_DEFAULT_FPFS_WEIGHTS = {
    "g": 0.5287963721184084,
    "r": 0.33049707056512084,
    "i": 0.1362332926465851,
    "z": 0.004473264669885652,
}


def calibrate_shapes(table, c0=50.0, weights=None, normalize=True):
    """4-band FPFS combination -> (e1, e2, response).

    Parameters
    ----------
    table : astropy.table.Table-like
        Must carry `{band}_fpfs1_m00`, `_m20`, `_m22c`, `_m22s`,
        `_dm22c_dg1`, `_dm22s_dg2`, `_dm00_dg{1,2}`, plus `wsel`,
        `dwsel_dg1`, `dwsel_dg2`.
    c0 : float
        Soft-bias regulariser on the m00 denominator.
    weights : dict, optional
        Per-band weights; defaults to AnaCal griz weights.
    normalize : bool
        If True, rescale weights to sum to 1.
    """
    if weights is None:
        weights = dict(_DEFAULT_FPFS_WEIGHTS)
    if normalize:
        s = sum(weights.values())
        weights = {k: v / s for k, v in weights.items()}

    n = len(table)
    m22c = np.zeros(n)
    m22s = np.zeros(n)
    m00 = np.zeros(n)
    dm22c_dg1 = np.zeros(n)
    dm22s_dg2 = np.zeros(n)
    dm00_dg1 = np.zeros(n)
    dm00_dg2 = np.zeros(n)
    for band, w in weights.items():
        m22c = m22c + table[f"{band}_fpfs1_m22c"] * w
        m22s = m22s + table[f"{band}_fpfs1_m22s"] * w
        m00 = m00 + table[f"{band}_fpfs1_m00"] * w
        dm22c_dg1 = dm22c_dg1 + table[f"{band}_fpfs1_dm22c_dg1"] * w
        dm22s_dg2 = dm22s_dg2 + table[f"{band}_fpfs1_dm22s_dg2"] * w
        dm00_dg1 = dm00_dg1 + table[f"{band}_fpfs1_dm00_dg1"] * w
        dm00_dg2 = dm00_dg2 + table[f"{band}_fpfs1_dm00_dg2"] * w

    denom = m00 + c0
    e1 = m22c / denom * table["wsel"]
    e2 = m22s / denom * table["wsel"]
    de1_dg1 = dm22c_dg1 / denom - m22c / denom ** 2 * dm00_dg1
    de2_dg2 = dm22s_dg2 / denom - m22s / denom ** 2 * dm00_dg2
    res = (
        de1_dg1 * table["wsel"] + table["dwsel_dg1"] * m22c / denom
        + de2_dg2 * table["wsel"] + table["dwsel_dg2"] * m22s / denom
    ) / 2.0
    return e1, e2, res


def fast_bootstrap_mean(data, ci_level=0.95):
    """SciPy-bootstrap CI of the sample mean."""
    from scipy.stats import bootstrap

    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Input data must be 1D")
    res = bootstrap(
        (data,), np.mean, confidence_level=ci_level,
    )
    return np.array(
        [res.confidence_interval.low, res.confidence_interval.high]
    )


def anacal_get_tang_cross(cluster, sky_dist, bins, R,
                          ci_level=0.95,
                          r_correction_offset=0.0,
                          verbose=False):
    """Radial-bin AnaCal complex shear into tangential/cross profiles.

    Parameters
    ----------
    cluster : complex ndarray
        Complex shear rotated to the cluster frame, `e_T + 1j e_X`.
    sky_dist : ndarray
        Per-source projected distance (e.g. Mpc).
    bins : ndarray
        Bin edges in the same units as `sky_dist`.
    R : ndarray
        Per-source shear response.
    ci_level : float
        Bootstrap CI level for the per-bin means.
    r_correction_offset : float
        Subtracted from `mean(R)` to define the response correction
        applied to the tangential / cross means.

    Returns
    -------
    tang_avg, cross_avg : ndarray
        Response-corrected tangential and cross means per bin.
    tang_err, cross_err : ndarray (nb, 2)
        Bootstrap CIs for tangential and cross.
    bin_rs : list of ndarray
        Per-bin source-index arrays.
    res_avg, res_err : ndarray
        Per-bin mean response and CI.
    """
    nb = len(bins) - 1
    res_avg = np.zeros(nb)
    res_err = np.zeros((nb, 2))
    tang_avg = np.zeros(nb)
    cross_avg = np.zeros_like(tang_avg)
    tang_err = np.zeros((nb, 2))
    cross_err = np.zeros_like(tang_err)
    bin_rs = []
    R_correction2 = float(np.mean(R)) - r_correction_offset

    for i in range(nb):
        m = (sky_dist > bins[i]) & (sky_dist < bins[i + 1])
        bin_rs.append(np.where(m)[0])
        if np.sum(m) < 1:
            continue
        sample = cluster[m]
        sample_t = sample.real
        sample_x = sample.imag
        sr = np.asarray(R)[m]
        if verbose:
            print(np.sum(m))
        R_correction = float(np.mean(sr))
        tang_avg[i] = np.mean(sample_t) / R_correction2
        cross_avg[i] = np.mean(sample_x) / R_correction2
        res_err[i] = fast_bootstrap_mean(sr, ci_level=ci_level)
        tang_err[i] = fast_bootstrap_mean(sample_t, ci_level=ci_level) / R_correction2
        cross_err[i] = fast_bootstrap_mean(sample_x, ci_level=ci_level) / R_correction2
        res_avg[i] = R_correction
    return tang_avg, cross_avg, tang_err, cross_err, bin_rs, res_avg, res_err
