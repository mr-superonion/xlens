import numpy as np
from numpy.typing import NDArray


def _linear_modes_to_derivs(xx: dict[str, NDArray]) -> dict[str, NDArray]:
    d: dict[str, NDArray] = {}
    d["dm00_dg1"] = -np.sqrt(2.0) * xx["m22c"]
    d["dm00_dg2"] = -np.sqrt(2.0) * xx["m22s"]
    d["dm20_dg1"] = -np.sqrt(6.0) / 2.0 * xx["m42c"]
    d["dm20_dg2"] = -np.sqrt(6.0) / 2.0 * xx["m42s"]
    d["dm22c_dg1"] = (
        (1.0 / np.sqrt(2.0)) * (xx["m00"] - xx["m40"])
        - np.sqrt(3.0) * xx["m44c"]
    )
    d["dm22c_dg2"] = -np.sqrt(3.0) * xx["m44s"]
    d["dm22s_dg1"] = -np.sqrt(3.0) * xx["m44s"]
    d["dm22s_dg2"] = (
        (1.0 / np.sqrt(2.0)) * (xx["m00"] - xx["m40"])
        + np.sqrt(3.0) * xx["m44c"]
    )
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
        dtype=np.dtype([
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
        ]),
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
        "m00", "m20", "m22c", "m22s",
        "m40", "m42c", "m42s", "m44c", "m44s",
    ]
    noise_names = [
        "n00", "n20", "n22c", "n22s",
        "n40", "n42c", "n42s", "n44c", "n44s",
    ]
    xx = {
        mn: data[f"{p}{mn}"] - 2.0 * data[f"{p}{nn}"]
        for mn, nn in zip(moment_names, noise_names)
    }
    d = _linear_modes_to_derivs(xx)
    return _moments_to_ell(
        len(data), C0, prefix,
        data[f"{p}m00"], data[f"{p}m20"],
        data[f"{p}m22c"], data[f"{p}m22s"],
        d["dm00_dg1"], d["dm00_dg2"],
        d["dm20_dg1"], d["dm20_dg2"],
        d["dm22c_dg1"], d["dm22c_dg2"],
        d["dm22s_dg1"], d["dm22s_dg2"],
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
        "dm00_dg1", "dm00_dg2",
        "dm20_dg1", "dm20_dg2",
        "dm22c_dg1", "dm22c_dg2",
        "dm22s_dg1", "dm22s_dg2",
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
        nobj, C0, prefix,
        mc["m00"], mc["m20"], mc["m22c"], mc["m22s"],
        dc["dm00_dg1"], dc["dm00_dg2"],
        dc["dm20_dg1"], dc["dm20_dg2"],
        dc["dm22c_dg1"], dc["dm22c_dg2"],
        dc["dm22s_dg1"], dc["dm22s_dg2"],
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
        raise ValueError(
            f"weights length {len(weights)} does not match "
            f"bands length {n}"
        )
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
        "m00", "m20", "m22c", "m22s",
        "m40", "m42c", "m42s", "m44c", "m44s",
    ]
    noise_names = [
        "n00", "n20", "n22c", "n22s",
        "n40", "n42c", "n42s", "n44c", "n44s",
    ]

    w_list = _normalize_band_weights(bands, weights)

    moments: dict[str, dict[str, NDArray]] = {}
    noises: dict[str, dict[str, NDArray]] = {}
    for b in bands:
        moments[b] = {
            mn: np.asarray(cat[f"{b}_{p}{mn}"], dtype=np.float64)
            for mn in moment_names
        }
        noises[b] = {
            nn: np.asarray(cat[f"{b}_{p}{nn}"], dtype=np.float64)
            for nn in noise_names
        }

    # --- per-band dm_dg using (m - 2n) ------------------------------------
    dmom_dg: dict[str, dict[str, NDArray]] = {}
    for b in bands:
        m = moments[b]
        n = noises[b]
        xx = {
            mn: m[mn] - 2.0 * n[nn]
            for mn, nn in zip(moment_names, noise_names)
        }
        dmom_dg[b] = _linear_modes_to_derivs(xx)

    return _multiband_moments2ell(
        nobj, bands, C0, prefix, w_list, moments, dmom_dg,
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
        "dm00_dg1", "dm00_dg2",
        "dm20_dg1", "dm20_dg2",
        "dm22c_dg1", "dm22c_dg2",
        "dm22s_dg1", "dm22s_dg2",
    ]

    w_list = _normalize_band_weights(bands, weights)

    moments: dict[str, dict[str, NDArray]] = {}
    dmom_dg: dict[str, dict[str, NDArray]] = {}
    for b in bands:
        moments[b] = {
            mn: np.asarray(cat[f"{b}_{p}{mn}"], dtype=np.float64)
            for mn in moment_names
        }
        dmom_dg[b] = {
            dn: np.asarray(cat[f"{b}_{p}{dn}"], dtype=np.float64)
            for dn in deriv_names
        }

    return _multiband_moments2ell(
        nobj, bands, C0, prefix, w_list, moments, dmom_dg,
    )


def _resolve_cut(
    flux_min: float | dict,
    bands: str,
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
