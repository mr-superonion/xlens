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

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

# Smooth magnitude truncation. The standard magnitude holds for sources
# brighter than MAG_KNEE; fainter than that it smoothly rolls off to the
# ceiling MAG_CAP, reaching MAG_CAP for zero/negative flux. Real detections
# (depth ~27 << 30) are unaffected; only non-detections are smoothed.
# Zeropoint-independent constants.
MAG_KNEE = 30.0
MAG_CAP = 40.0
# d(mag)/d(ln flux) = 2.5/ln(10) ~= 1.0857: magnitudes per fractional flux
# error (sigma_mag ~= MAG_ERR_FAC * sigma_f / f).
MAG_ERR_FAC = 2.5 / np.log(10.0)


def flux_to_mag(
    flux: NDArray,
    mag_zero: float,
    flux_err: NDArray | None = None,
    a_ext: NDArray | None = None,
    m_knee: float = MAG_KNEE,
    m_cap: float = MAG_CAP,
) -> tuple[NDArray, NDArray | None]:
    """Per-band flux -> (mag, mag_err), smoothly truncated at the faint end.

    For a source brighter than ``m_knee`` the standard magnitude
    ``mag = mag_zero - 2.5*log10(flux)`` (minus per-band extinction
    ``a_ext`` when given) is returned unchanged. Fainter than ``m_knee``
    the magnitude is blended by a C1 smoothstep toward the ceiling
    ``m_cap``, reaching ``m_cap`` exactly for ``flux <= flux(m_cap)``
    -- including zero and negative flux (no ``log10`` of non-positive
    flux, no discontinuity at ``flux = 0``).

    When ``flux_err`` is given, the error is propagated from the smoothly
    truncated magnitude: ``mag_err = MAG_ERR_FAC * flux_err / flux_eff``
    with ``flux_eff = 10**((mag_zero - mag)/2.5)`` the flux implied by
    the truncated ``mag``. For bright sources ``flux_eff == flux`` so
    this is the standard ``MAG_ERR_FAC * flux_err / flux``; as ``mag``
    saturates, ``flux_eff`` -> the fixed ``m_cap`` floor flux, so the
    error inflates smoothly and stays bounded and well-defined for
    ``flux <= 0``. Returns ``mag_err = None`` when ``flux_err`` is
    ``None``.
    """
    n = len(flux)
    # standard magnitude where flux>0; a finite sentinel above the cap
    # elsewhere so the smoothstep blend saturates to m_cap for
    # zero/negative flux.
    m_raw = np.full(n, m_cap + 1.0, dtype=np.float64)
    pos = flux > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        m_raw[pos] = mag_zero - 2.5 * np.log10(flux[pos])
        if a_ext is not None:
            m_raw[pos] = m_raw[pos] - a_ext[pos]
    t = np.clip((m_raw - m_knee) / (m_cap - m_knee), 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)  # smoothstep (C1)
    mag = (1.0 - s) * m_raw + s * m_cap
    if flux_err is None:
        return mag, None
    flux_eff = 10.0 ** ((mag_zero - mag) / 2.5)
    mag_err = MAG_ERR_FAC * flux_err / flux_eff
    return mag, mag_err


def dmag_dflux(
    flux: NDArray,
    mag_zero: float,
    a_ext: NDArray | None = None,
    m_knee: float = MAG_KNEE,
    m_cap: float = MAG_CAP,
) -> NDArray:
    """Analytic d(mag)/d(flux) of the smooth-truncated ``flux_to_mag``.

    Differentiates the smoothstep-truncated magnitude (not the plain
    log). With ``m_raw = mag_zero - 2.5*log10(flux) - a_ext`` and the
    smoothstep blend ``s(t)``,
    ``t = clip((m_raw - m_knee)/(m_cap - m_knee), 0, 1)``, the chain
    rule gives::

        dmag_dflux = (dm_raw/dflux) * [(1 - s) + (m_cap - m_raw) * 6t(1-t)/W]

    with ``W = m_cap - m_knee`` and
    ``dm_raw/dflux = -MAG_ERR_FAC/flux``. Brighter than ``m_knee``
    (``t=0``) the bracket is 1 -> the standard
    ``-MAG_ERR_FAC/flux``; in the saturated band (``t=1``) and for
    ``flux <= 0`` (mag pinned at ``m_cap``) the derivative is exactly
    0. Continuous everywhere (the smoothstep has ``s'(0)=s'(1)=0``).
    """
    n = len(flux)
    m_raw = np.full(n, m_cap + 1.0, dtype=np.float64)
    dmraw_dflux = np.zeros(n, dtype=np.float64)
    pos = flux > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        m_raw[pos] = mag_zero - 2.5 * np.log10(flux[pos])
        if a_ext is not None:
            m_raw[pos] = m_raw[pos] - a_ext[pos]
        dmraw_dflux[pos] = -MAG_ERR_FAC / flux[pos]
    w = m_cap - m_knee
    t = np.clip((m_raw - m_knee) / w, 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)  # smoothstep (C1)
    # Bracket: (1 - s) + (m_cap - m_raw) * s'(t)/W, with s'(t) = 6 t (1 - t).
    # s'(t) is exactly 0 in both clipped regions, so this holds for all flux>0;
    # flux<=0 entries carry dmraw_dflux = 0 (m_raw is the finite sentinel).
    bracket = (1.0 - s) + (m_cap - m_raw) * (6.0 * t * (1.0 - t)) / w
    return dmraw_dflux * bracket


def mag_shear_response(
    flux: NDArray,
    dflux_dg1: NDArray,
    dflux_dg2: NDArray,
    mag_zero: float,
    flux_err: NDArray | None = None,
    a_ext: NDArray | None = None,
    m_knee: float = MAG_KNEE,
    m_cap: float = MAG_CAP,
) -> tuple[
    NDArray,
    NDArray | None,
    NDArray,
    NDArray,
    NDArray | None,
    NDArray | None,
]:
    """Smooth-truncated magnitude and its shear response.

    Returns ``(mag, sigma_m, dmag_dg1, dmag_dg2, dsigma_m_dg1, dsigma_m_dg2)``.

    ``mag`` and ``sigma_m`` (the magnitude error) come from
    ``flux_to_mag``. The magnitude shear response chains the analytic
    ``dmag_dflux`` with the per-object flux response::

        dmag_dg{c} = dmag_dflux * dflux_dg{c}

    and the magnitude-error response follows from ``sigma_m = MAG_ERR_FAC *
    flux_err / flux_eff`` (with ``flux_err`` shear-independent)::

        dsigma_m_dg{c} = (ln10 / 2.5) * sigma_m * dmag_dg{c}
                       = sigma_m * dmag_dg{c} / MAG_ERR_FAC

    ``sigma_m``, ``dsigma_m_dg1`` and ``dsigma_m_dg2`` are ``None`` when
    ``flux_err`` is ``None``.
    """
    mag, sigma_m = flux_to_mag(
        flux, mag_zero, flux_err=flux_err, a_ext=a_ext, m_knee=m_knee, m_cap=m_cap
    )
    dm_df = dmag_dflux(flux, mag_zero, a_ext=a_ext, m_knee=m_knee, m_cap=m_cap)
    dmag_dg1 = dm_df * dflux_dg1
    dmag_dg2 = dm_df * dflux_dg2
    if sigma_m is None:
        return mag, None, dmag_dg1, dmag_dg2, None, None
    dsigma_m_dg1 = sigma_m * dmag_dg1 / MAG_ERR_FAC
    dsigma_m_dg2 = sigma_m * dmag_dg2 / MAG_ERR_FAC
    return mag, sigma_m, dmag_dg1, dmag_dg2, dsigma_m_dg1, dsigma_m_dg2


def add_magnitude_columns(
    catalog: NDArray,
    mag_zero: float,
    families: tuple[str, ...] = ("fpfs1", "gauss2"),
) -> NDArray:
    """Append per-band AB magnitude + shear response for each flux family.

    For every band ``b`` and family ``fam`` in ``families`` for which the
    catalog carries a flux ``{b}_flux_{fam}`` and its shear response
    ``{b}_dflux_{fam}_dg1`` / ``{b}_dflux_{fam}_dg2``, append (on the fixed
    ``mag_zero`` zeropoint, via the smooth-truncated :func:`flux_to_mag`)::

        {b}_mag_{fam}          {b}_dmag_{fam}_dg1        {b}_dmag_{fam}_dg2
        {b}_mag_{fam}_err      {b}_dmag_{fam}_err_dg1    {b}_dmag_{fam}_err_dg2

    where ``dmag_{fam}_dg{c} = dmag_dflux * dflux_{fam}_dg{c}`` and
    ``dmag_{fam}_err_dg{c} = (ln10/2.5) * mag_err * dmag_{fam}_dg{c}``.

    The three ``mag_err`` columns are added only when ``{b}_flux_{fam}_err``
    is present. The ``_err`` suffix mirrors the flux convention
    (``{b}_flux_{fam}_err``). No extinction is applied here (extinction is
    a downstream, photo-z-time correction). Bands are discovered from the
    column names, so this works on the survey-prefixed schema
    (``lsst_i_flux_fpfs1`` -> ``lsst_i_mag_fpfs1``).
    """
    names = catalog.dtype.names
    if names is None:
        return catalog
    nameset = set(names)
    new_names: list[str] = []
    new_cols: list[NDArray] = []
    for fam in families:
        suffix = f"_flux_{fam}"
        for col in names:
            if not col.endswith(suffix):
                continue
            b = col[: -len(suffix)]
            d1 = f"{b}_dflux_{fam}_dg1"
            d2 = f"{b}_dflux_{fam}_dg2"
            if d1 not in nameset or d2 not in nameset:
                continue
            flux = np.asarray(catalog[col], dtype=np.float64)
            dflux_dg1 = np.asarray(catalog[d1], dtype=np.float64)
            dflux_dg2 = np.asarray(catalog[d2], dtype=np.float64)
            ecol = f"{b}_flux_{fam}_err"
            flux_err = np.asarray(catalog[ecol], dtype=np.float64) if ecol in nameset else None
            mag, sigma_mag, dmag_dg1, dmag_dg2, dsig_dg1, dsig_dg2 = mag_shear_response(
                flux, dflux_dg1, dflux_dg2, mag_zero, flux_err=flux_err
            )
            new_names += [f"{b}_mag_{fam}", f"{b}_dmag_{fam}_dg1", f"{b}_dmag_{fam}_dg2"]
            new_cols += [mag, dmag_dg1, dmag_dg2]
            if sigma_mag is not None:
                new_names += [
                    f"{b}_mag_{fam}_err",
                    f"{b}_dmag_{fam}_err_dg1",
                    f"{b}_dmag_{fam}_err_dg2",
                ]
                new_cols += [sigma_mag, dsig_dg1, dsig_dg2]
    if not new_names:
        return catalog
    return np.asarray(rfn.append_fields(catalog, new_names, new_cols, usemask=False))


def _linear_modes_to_derivs(xx: dict[str, NDArray]) -> dict[str, NDArray]:
    d: dict[str, NDArray] = {}
    d["dm00_dg1"] = -np.sqrt(2.0) * xx["m22c"]
    d["dm00_dg2"] = -np.sqrt(2.0) * xx["m22s"]
    d["dm20_dg1"] = -np.sqrt(6.0) * xx["m42c"]
    d["dm20_dg2"] = -np.sqrt(6.0) * xx["m42s"]
    d["dm22c_dg1"] = (1.0 / np.sqrt(2.0)) * (xx["m00"] - xx["m40"]) - np.sqrt(3.0) * xx["m44c"]
    d["dm22c_dg2"] = -np.sqrt(3.0) * xx["m44s"]
    d["dm22s_dg1"] = -np.sqrt(3.0) * xx["m44s"]
    d["dm22s_dg2"] = (1.0 / np.sqrt(2.0)) * (xx["m00"] - xx["m40"]) + np.sqrt(3.0) * xx["m44c"]
    return d


def _moments_to_ell(
    nobj: int,
    C0: float,
    prefix: str,
    m00: NDArray,
    m20: NDArray,
    m22c: NDArray,
    m22s: NDArray,
    dm00_dg1: NDArray,
    dm00_dg2: NDArray,
    dm20_dg1: NDArray,
    dm20_dg2: NDArray,
    dm22c_dg1: NDArray,
    dm22c_dg2: NDArray,
    dm22s_dg1: NDArray,
    dm22s_dg2: NDArray,
) -> NDArray:
    p = prefix
    denom = m00 + C0
    denom_sq = denom * denom

    e1 = m22c / denom
    e2 = m22s / denom

    de1_dg1 = dm22c_dg1 / denom - dm00_dg1 * m22c / denom_sq
    de1_dg2 = dm22c_dg2 / denom - dm00_dg2 * m22c / denom_sq
    de2_dg1 = dm22s_dg1 / denom - dm00_dg1 * m22s / denom_sq
    de2_dg2 = dm22s_dg2 / denom - dm00_dg2 * m22s / denom_sq

    # m0 = m00, m2 = m00 + m20
    m0 = m00
    m2 = m00 + m20
    dm0_dg1 = dm00_dg1
    dm0_dg2 = dm00_dg2
    dm2_dg1 = dm00_dg1 + dm20_dg1
    dm2_dg2 = dm00_dg2 + dm20_dg2

    out = np.zeros(
        nobj,
        dtype=np.dtype(
            [
                (f"{p}e1", np.float64),
                (f"{p}de1_dg1", np.float64),
                (f"{p}de1_dg2", np.float64),
                (f"{p}e2", np.float64),
                (f"{p}de2_dg1", np.float64),
                (f"{p}de2_dg2", np.float64),
                (f"{p}m0", np.float64),
                (f"{p}dm0_dg1", np.float64),
                (f"{p}dm0_dg2", np.float64),
                (f"{p}m2", np.float64),
                (f"{p}dm2_dg1", np.float64),
                (f"{p}dm2_dg2", np.float64),
            ]
        ),
    )
    out[f"{p}e1"] = e1
    out[f"{p}de1_dg1"] = de1_dg1
    out[f"{p}de1_dg2"] = de1_dg2
    out[f"{p}e2"] = e2
    out[f"{p}de2_dg1"] = de2_dg1
    out[f"{p}de2_dg2"] = de2_dg2
    out[f"{p}m0"] = m0
    out[f"{p}dm0_dg1"] = dm0_dg1
    out[f"{p}dm0_dg2"] = dm0_dg2
    out[f"{p}m2"] = m2
    out[f"{p}dm2_dg1"] = dm2_dg1
    out[f"{p}dm2_dg2"] = dm2_dg2
    return out


def shapelets_linear2ell(
    data: NDArray,
    C0: float,
    prefix: str = "fpfs1_",
) -> NDArray:
    p = prefix
    moment_names = [
        "m00",
        "m20",
        "m22c",
        "m22s",
        "m40",
        "m42c",
        "m42s",
        "m44c",
        "m44s",
    ]
    noise_names = [
        "n00",
        "n20",
        "n22c",
        "n22s",
        "n40",
        "n42c",
        "n42s",
        "n44c",
        "n44s",
    ]
    xx = {mn: data[f"{p}{mn}"] - 2.0 * data[f"{p}{nn}"] for mn, nn in zip(moment_names, noise_names)}
    d = _linear_modes_to_derivs(xx)
    return _moments_to_ell(
        len(data),
        C0,
        prefix,
        data[f"{p}m00"],
        data[f"{p}m20"],
        data[f"{p}m22c"],
        data[f"{p}m22s"],
        d["dm00_dg1"],
        d["dm00_dg2"],
        d["dm20_dg1"],
        d["dm20_dg2"],
        d["dm22c_dg1"],
        d["dm22c_dg2"],
        d["dm22s_dg1"],
        d["dm22s_dg2"],
    )


def _multiband_moments2ell(
    nobj: int,
    bands: list[str],
    C0: float,
    prefix: str,
    w_list: list[float],
    moments: dict[str, dict[str, NDArray]],
    dmom_dg: dict[str, dict[str, NDArray]],
) -> NDArray:
    """Combine per-band shapelet moments into a single shape catalog
    using user-supplied constant per-band weights ``w_list`` (which
    must already be normalized to sum to 1).  Because the weights are
    constant in shear, ``dw/dg`` is zero and only the per-band moment
    derivatives contribute to the combined response.
    """
    deriv_names = [
        "dm00_dg1",
        "dm00_dg2",
        "dm20_dg1",
        "dm20_dg2",
        "dm22c_dg1",
        "dm22c_dg2",
        "dm22s_dg1",
        "dm22s_dg2",
    ]

    mc: dict[str, NDArray] = {}
    for mn in ["m00", "m20", "m22c", "m22s"]:
        s = np.zeros(nobj)
        for ib, b in enumerate(bands):
            s += w_list[ib] * moments[b][mn]
        mc[mn] = s

    dc: dict[str, NDArray] = {}
    for dg_name in deriv_names:
        s = np.zeros(nobj)
        for ib, b in enumerate(bands):
            s += w_list[ib] * dmom_dg[b][dg_name]
        dc[dg_name] = s

    return _moments_to_ell(
        nobj,
        C0,
        prefix,
        mc["m00"],
        mc["m20"],
        mc["m22c"],
        mc["m22s"],
        dc["dm00_dg1"],
        dc["dm00_dg2"],
        dc["dm20_dg1"],
        dc["dm20_dg2"],
        dc["dm22c_dg1"],
        dc["dm22c_dg2"],
        dc["dm22s_dg1"],
        dc["dm22s_dg2"],
    )


def _normalize_band_weights(
    bands: list[str],
    weights: list[float] | None,
) -> list[float]:
    """Return per-band constant weights normalized to sum to 1.  If
    ``weights`` is ``None`` an equal 1/N split is used.
    """
    n = len(bands)
    if n == 0:
        raise ValueError("bands must be non-empty")
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} does not match " f"bands length {n}")
    arr = np.asarray(weights, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError("weights must be non-negative")
    total = float(arr.sum())
    if not (total > 0):
        raise ValueError("weights must have positive sum")
    return (arr / total).tolist()


def multiband_shapelets_linear2ell(
    cat: NDArray,
    bands: list[str],
    C0: float,
    prefix: str = "fpfs1_",
    weights: list[float] | None = None,
) -> NDArray:
    """Combine per-band linear-mode shapelets into a single shape catalog.

    Per-band weights are user-supplied constants (no per-object error
    needed); they are normalized to sum to 1.  Because the weights are
    constant in shear, ``dw/dg1`` and ``dw/dg2`` are exactly zero.
    """
    p = prefix
    nobj = len(cat)

    moment_names = [
        "m00",
        "m20",
        "m22c",
        "m22s",
        "m40",
        "m42c",
        "m42s",
        "m44c",
        "m44s",
    ]
    noise_names = [
        "n00",
        "n20",
        "n22c",
        "n22s",
        "n40",
        "n42c",
        "n42s",
        "n44c",
        "n44s",
    ]

    w_list = _normalize_band_weights(bands, weights)

    moments: dict[str, dict[str, NDArray]] = {}
    noises: dict[str, dict[str, NDArray]] = {}
    for b in bands:
        moments[b] = {mn: np.asarray(cat[f"{b}_{p}{mn}"], dtype=np.float64) for mn in moment_names}
        noises[b] = {nn: np.asarray(cat[f"{b}_{p}{nn}"], dtype=np.float64) for nn in noise_names}

    # --- per-band dm_dg using (m - 2n) ------------------------------------
    dmom_dg: dict[str, dict[str, NDArray]] = {}
    for b in bands:
        m = moments[b]
        n = noises[b]
        xx = {mn: m[mn] - 2.0 * n[nn] for mn, nn in zip(moment_names, noise_names)}
        dmom_dg[b] = _linear_modes_to_derivs(xx)

    return _multiband_moments2ell(
        nobj,
        bands,
        C0,
        prefix,
        w_list,
        moments,
        dmom_dg,
    )


def multiband_shapelets2ell(
    cat: NDArray,
    bands: list[str],
    C0: float,
    prefix: str = "fpfs1_",
    weights: list[float] | None = None,
) -> NDArray:
    """Combine per-band shapelets (with explicit dm/dg columns) into a
    single shape catalog using user-supplied constant per-band weights
    (normalized to sum to 1).  ``dw/dg`` is zero by construction.
    """
    p = prefix
    nobj = len(cat)

    moment_names = ["m00", "m20", "m22c", "m22s"]
    deriv_names = [
        "dm00_dg1",
        "dm00_dg2",
        "dm20_dg1",
        "dm20_dg2",
        "dm22c_dg1",
        "dm22c_dg2",
        "dm22s_dg1",
        "dm22s_dg2",
    ]

    w_list = _normalize_band_weights(bands, weights)

    moments: dict[str, dict[str, NDArray]] = {}
    dmom_dg: dict[str, dict[str, NDArray]] = {}
    for b in bands:
        moments[b] = {mn: np.asarray(cat[f"{b}_{p}{mn}"], dtype=np.float64) for mn in moment_names}
        dmom_dg[b] = {dn: np.asarray(cat[f"{b}_{p}{dn}"], dtype=np.float64) for dn in deriv_names}

    return _multiband_moments2ell(
        nobj,
        bands,
        C0,
        prefix,
        w_list,
        moments,
        dmom_dg,
    )


def _resolve_cut(
    flux_min: float | dict,
    bands: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """Return per-band flux_min as dict{band: value}."""
    if isinstance(flux_min, dict):
        return {b: float(flux_min[b]) for b in bands}
    else:
        f = float(flux_min)
        return {b: f for b in bands}


def _resolve_cut_name(flux_name: str) -> str:
    """Normalize flux_name to column suffix ('' or f'_{name}')."""
    if len(flux_name) > 0:
        if flux_name[0] != "_":
            fn = "_" + flux_name
        else:
            fn = flux_name
    else:
        fn = ""
    return fn
