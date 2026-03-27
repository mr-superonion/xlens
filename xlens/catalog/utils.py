import numpy as np
from numpy.typing import NDArray


def _linear_modes_to_derivs(xx: dict[str, NDArray]) -> dict[str, NDArray]:
    d: dict[str, NDArray] = {}
    d["dm00_dg1"] = -np.sqrt(2.0) * xx["m22c"]
    d["dm00_dg2"] = -np.sqrt(2.0) * xx["m22s"]
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
    m22c: NDArray,
    m22s: NDArray,
    dm00_dg1: NDArray,
    dm00_dg2: NDArray,
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

    out = np.zeros(
        nobj,
        dtype=np.dtype([
            (f"{p}e1", np.float64),
            (f"{p}de1_dg1", np.float64),
            (f"{p}de1_dg2", np.float64),
            (f"{p}e2", np.float64),
            (f"{p}de2_dg1", np.float64),
            (f"{p}de2_dg2", np.float64),
        ]),
    )
    out[f"{p}e1"] = e1
    out[f"{p}de1_dg1"] = de1_dg1
    out[f"{p}de1_dg2"] = de1_dg2
    out[f"{p}e2"] = e2
    out[f"{p}de2_dg1"] = de2_dg1
    out[f"{p}de2_dg2"] = de2_dg2
    return out


def shapelets_linear2ell(
    data: NDArray,
    C0: float,
    prefix: str = "fpfs1_",
) -> NDArray:
    p = prefix
    moment_names = ["m00", "m22c", "m22s", "m40", "m44c", "m44s"]
    noise_names = ["n00", "n22c", "n22s", "n40", "n44c", "n44s"]
    xx = {
        mn: data[f"{p}{mn}"] - 2.0 * data[f"{p}{nn}"]
        for mn, nn in zip(moment_names, noise_names)
    }
    d = _linear_modes_to_derivs(xx)
    return _moments_to_ell(
        len(data), C0, prefix,
        data[f"{p}m00"], data[f"{p}m22c"], data[f"{p}m22s"],
        d["dm00_dg1"], d["dm00_dg2"],
        d["dm22c_dg1"], d["dm22c_dg2"],
        d["dm22s_dg1"], d["dm22s_dg2"],
    )


def _multiband_moments2ell(
    nobj: int,
    bands: list[str],
    C0: float,
    prefix: str,
    w_list: list[NDArray],
    dw_dg1_list: list[NDArray],
    dw_dg2_list: list[NDArray],
    moments: dict[str, dict[str, NDArray]],
    pb_derivs: dict[str, dict[str, NDArray]],
) -> NDArray:
    W = np.sum(w_list, axis=0)
    dW_dg1 = np.sum(dw_dg1_list, axis=0)
    dW_dg2 = np.sum(dw_dg2_list, axis=0)

    # --- weight-average raw moments ---------------------------------------
    mc = {}
    for mn in ["m00", "m22c", "m22s"]:
        s = np.zeros(nobj)
        for ib, b in enumerate(bands):
            s += w_list[ib] * moments[b][mn]
        mc[mn] = s / W

    # --- combined moment derivatives with weight response -----------------
    deriv_names = [
        "dm00_dg1", "dm00_dg2",
        "dm22c_dg1", "dm22c_dg2",
        "dm22s_dg1", "dm22s_dg2",
    ]
    moment_for_deriv = {
        "dm00_dg1": "m00", "dm00_dg2": "m00",
        "dm22c_dg1": "m22c", "dm22c_dg2": "m22c",
        "dm22s_dg1": "m22s", "dm22s_dg2": "m22s",
    }
    dw_for_deriv = {
        "dm00_dg1": (dw_dg1_list, dW_dg1),
        "dm00_dg2": (dw_dg2_list, dW_dg2),
        "dm22c_dg1": (dw_dg1_list, dW_dg1),
        "dm22c_dg2": (dw_dg2_list, dW_dg2),
        "dm22s_dg1": (dw_dg1_list, dW_dg1),
        "dm22s_dg2": (dw_dg2_list, dW_dg2),
    }
    dc: dict[str, NDArray] = {}
    for dg_name in deriv_names:
        mn = moment_for_deriv[dg_name]
        dw_list, dW = dw_for_deriv[dg_name]
        num = np.zeros(nobj)
        for ib, b in enumerate(bands):
            num += (
                w_list[ib] * pb_derivs[b][dg_name]
                + dw_list[ib] * moments[b][mn]
            )
        dc[dg_name] = num / W - mc[mn] * dW / W

    return _moments_to_ell(
        nobj, C0, prefix,
        mc["m00"], mc["m22c"], mc["m22s"],
        dc["dm00_dg1"], dc["dm00_dg2"],
        dc["dm22c_dg1"], dc["dm22c_dg2"],
        dc["dm22s_dg1"], dc["dm22s_dg2"],
    )


def multiband_shapelets_linear2ell(
    cat: NDArray,
    bands: list[str],
    C0: float,
    prefix: str = "fpfs1_",
) -> NDArray:
    p = prefix
    nobj = len(cat)

    moment_names = [
        "m00", "m22c", "m22s",
        "m40", "m44c", "m44s",
    ]
    noise_names = [
        "n00", "n22c", "n22s",
        "n40", "n44c", "n44s",
    ]

    # --- per-band weights and their shear derivatives ---------------------
    w_list = []
    dw_dg1_list = []
    dw_dg2_list = []
    moments: dict[str, dict[str, NDArray]] = {}
    noises: dict[str, dict[str, NDArray]] = {}

    for b in bands:
        flux = cat[f"{b}_flux_gauss2"]
        err_sq = cat[f"{b}_flux_gauss2_err"] ** 2
        w_list.append(flux * flux / err_sq)
        dw_dg1_list.append(
            2.0 * flux * cat[f"{b}_dflux_gauss2_dg1"] / err_sq
        )
        dw_dg2_list.append(
            2.0 * flux * cat[f"{b}_dflux_gauss2_dg2"] / err_sq
        )
        moments[b] = {
            mn: np.asarray(cat[f"{b}_{p}{mn}"], dtype=np.float64)
            for mn in moment_names
        }
        noises[b] = {
            nn: np.asarray(cat[f"{b}_{p}{nn}"], dtype=np.float64)
            for nn in noise_names
        }

    # --- per-band dm_dg using (m - 2n) ------------------------------------
    pb_derivs: dict[str, dict[str, NDArray]] = {}
    for b in bands:
        m = moments[b]
        n = noises[b]
        xx = {
            mn: m[mn] - 2.0 * n[nn]
            for mn, nn in zip(moment_names, noise_names)
        }
        pb_derivs[b] = _linear_modes_to_derivs(xx)

    return _multiband_moments2ell(
        nobj, bands, C0, prefix,
        w_list, dw_dg1_list, dw_dg2_list,
        moments, pb_derivs,
    )


def multiband_shapelets2ell(
    cat: NDArray,
    bands: list[str],
    C0: float,
    prefix: str = "fpfs1_",
) -> NDArray:
    p = prefix
    nobj = len(cat)

    moment_names = ["m00", "m22c", "m22s"]
    deriv_names = [
        "dm00_dg1", "dm00_dg2",
        "dm22c_dg1", "dm22c_dg2",
        "dm22s_dg1", "dm22s_dg2",
    ]

    # --- per-band weights and their shear derivatives ---------------------
    w_list = []
    dw_dg1_list = []
    dw_dg2_list = []
    moments: dict[str, dict[str, NDArray]] = {}
    pb_derivs: dict[str, dict[str, NDArray]] = {}

    for b in bands:
        flux = cat[f"{b}_flux_gauss2"]
        err_sq = cat[f"{b}_flux_gauss2_err"] ** 2
        w_list.append(flux * flux / err_sq)
        dw_dg1_list.append(
            2.0 * flux * cat[f"{b}_dflux_gauss2_dg1"] / err_sq
        )
        dw_dg2_list.append(
            2.0 * flux * cat[f"{b}_dflux_gauss2_dg2"] / err_sq
        )
        moments[b] = {
            mn: np.asarray(cat[f"{b}_{p}{mn}"], dtype=np.float64)
            for mn in moment_names
        }
        pb_derivs[b] = {
            dn: np.asarray(cat[f"{b}_{p}{dn}"], dtype=np.float64)
            for dn in deriv_names
        }

    return _multiband_moments2ell(
        nobj, bands, C0, prefix,
        w_list, dw_dg1_list, dw_dg2_list,
        moments, pb_derivs,
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
