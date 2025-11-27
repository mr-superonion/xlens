import numpy as np
from .utils import _resolve_flux_min, _resolve_flux_name
from .model import w_model, w_model_derivs

MAG_MIN = 18.0


def get_esq(src: np.ndarray, comp: int = 1, dg: float = 0.0) -> np.ndarray:
    """Return |e|^2 evaluated at shear g_comp = dg to first order."""
    if comp not in (1, 2):
        raise ValueError(f"comp must be 1 or 2, got {comp!r}")

    e = src[f"fpfs_e{comp}"]
    de = src[f"fpfs_de{comp}_dg{comp}"]

    comp2 = 3 - comp        # 1 to 2; 2 to 1
    e2 = src[f"fpfs_e{comp2}"]
    de2 = src[f"fpfs_de{comp2}_dg{comp}"]

    esq0 = e * e + e2 * e2
    return esq0 + 2.0 * dg * (e * de + e2 * de2)


def _bin_count(idx, weights, minlength):
    return np.bincount(idx, weights=weights, minlength=minlength)[1:-1]


def measure_shear(
    *,
    src: np.ndarray,
    z_estimator,
    zbounds: list[float],
    flux_min: float | dict = 40.0,
    emax: float = 0.3,
    z_width95_max: float = 2.75,
    dg: float = 0.02,
    target: str = "g1",
    do_correction: bool = True,
    mag_zero: float = 30.0,
    flux_name: str = "gauss2",
    bands: str = "grizy",
    ref_band: str = "i",
):
    """
    Measure shear components in redshift bins, using a supplied z-estimator.

    Parameters
    ----------
    z_estimator : object
        Must provide method
          get_zsel(
              src, *, mag_zero, flux_name, bands, ref_band, comp, dg
          ) -> (zmode, width95)
    """
    fn = _resolve_flux_name(flux_name)
    esq0 = get_esq(src)

    fmax = 10.0 ** ((mag_zero - MAG_MIN) / 2.5)
    # band-independent fields
    e1_all = src["fpfs_e1"]
    e2_all = src["fpfs_e2"]
    de1_dg1 = src["fpfs_de1_dg1"]
    de2_dg2 = src["fpfs_de2_dg2"]
    pars = np.array([5.1161, -0.8258, 14.7111, 0.0242, -2.0378, -0.5194])
    wmod = w_model(
        src["i_flux_gauss2"],
        src["fpfs_m0"],
        src["fpfs_m2"],
        mag_zero,
        *pars,
    )
    wopt = src["wsel"] * wmod
    dwmod = w_model_derivs(
        src["i_flux_gauss2"],
        src["fpfs_m0"],
        src["fpfs_m2"],
        mag_zero,
        *pars,
    )
    dwmod_dg1 = (
        dwmod["dw_dflux"] * src["i_dflux_gauss2_dg1"] +
        dwmod["dw_dm0"] * src["fpfs_dm0_dg1"] +
        dwmod["dw_dm2"] * src["fpfs_dm2_dg1"]
    )
    dwmod_dg2 = (
        dwmod["dw_dflux"] * src["i_dflux_gauss2_dg2"] +
        dwmod["dw_dm0"] * src["fpfs_dm0_dg2"] +
        dwmod["dw_dm2"] * src["fpfs_dm2_dg2"]
    )

    dw_dg1 = src["dwsel_dg1"] * wmod + dwmod_dg1 * src["wsel"]
    dw_dg2 = src["dwsel_dg2"] * wmod + dwmod_dg2 * src["wsel"]

    # per-band flux minima and base fluxes
    fm = _resolve_flux_min(flux_min, bands=bands)
    flux = {b: src[f"{b}_flux{fn}"] for b in bands}

    # base selection: all bands above flux_min and |e|^2 < emax^2
    mask = np.ones(src.shape[0], dtype=bool)
    for b in bands:
        mask &= flux[b] > fm[b]
        if b == ref_band:
            mask &= (flux[b] < fmax)
    mask &= esq0 < emax * emax

    # photo-z + width cut at base shear
    zmode, width95 = z_estimator.get_zsel(
        src[mask],
        mag_zero=mag_zero,
        flux_name=flux_name,
        bands=bands,
        ref_band=ref_band,
        comp=1,      # base comp for selection mask; not used in esq0
        dg=0.0,
    )
    mtmp = width95 < z_width95_max
    mask[mask] &= mtmp
    zmode = zmode[mtmp]
    del mtmp, width95
    minlen = len(zbounds) + 1

    def sel_term(comp: int) -> np.ndarray:
        """Selection response term for component comp (1 or 2)."""
        e_comp = src[f"fpfs_e{comp}"]

        def one_side(sign: float) -> np.ndarray:
            """Compute binned ⟨w_sel e⟩ for shear +sign*dg."""
            dg_eff = sign * dg
            esq_side = get_esq(src, comp=comp, dg=dg_eff)

            mask_side = esq_side < emax * emax
            for b in bands:
                df = src[f"{b}_dflux{fn}_dg{comp}"]
                mask_side &= (flux[b] + dg_eff * df > fm[b])
                if b == ref_band:
                    mask_side &= (flux[b] + dg_eff * df < fmax)

            if do_correction:
                z_side, w_side = z_estimator.get_zsel(
                    src[mask_side],
                    mag_zero=mag_zero,
                    flux_name=flux_name,
                    bands=bands,
                    ref_band=ref_band,
                    comp=comp,
                    dg=dg_eff,
                )
            else:
                z_side, w_side = z_estimator.get_zsel(
                    src[mask_side],
                    mag_zero=mag_zero,
                    flux_name=flux_name,
                    bands=bands,
                    ref_band=ref_band,
                    comp=comp,
                    dg=0.0,
                )

            mtmp_local = w_side < z_width95_max
            mask_side[mask_side] &= mtmp_local
            z_side = z_side[mtmp_local]
            del mtmp_local, w_side

            idx_side = np.digitize(z_side, zbounds, right=False)
            ell_side = _bin_count(
                idx_side,
                wopt[mask_side] * e_comp[mask_side],
                minlength=minlen,
            )
            del esq_side, mask_side, idx_side
            return ell_side

        ellp = one_side(+1.0)
        ellm = one_side(-1.0)
        return (ellp - ellm) / (2.0 * dg)

    idx0 = np.digitize(zmode, zbounds, right=False)
    if target == "g1":
        e1 = _bin_count(idx0, wopt[mask] * e1_all[mask], minlength=minlen)
        r1 = _bin_count(
            idx0,
            dw_dg1[mask] * e1_all[mask] + wopt[mask] * de1_dg1[mask],
            minlength=minlen,
        )
        r1_sel = sel_term(1)
        return {"e": e1, "r": r1, "r_sel": r1_sel}
    elif target == "g2":
        e2 = _bin_count(idx0, wopt[mask] * e2_all[mask], minlength=minlen)
        r2 = _bin_count(
            idx0,
            dw_dg2[mask] * e2_all[mask] + wopt[mask] * de2_dg2[mask],
            minlength=minlen,
        )
        r2_sel = sel_term(2)
        return {"e": e2, "r": r2, "r_sel": r2_sel}
    elif target == "g1g2":
        e1 = _bin_count(idx0, wopt[mask] * e1_all[mask], minlength=minlen)
        r1 = _bin_count(
            idx0,
            dw_dg1[mask] * e1_all[mask] + wopt[mask] * de1_dg1[mask],
            minlength=minlen,
        )
        r1_sel = sel_term(1)
        e2 = _bin_count(idx0, wopt[mask] * e2_all[mask], minlength=minlen)
        r2 = _bin_count(
            idx0,
            dw_dg2[mask] * e2_all[mask] + wopt[mask] * de2_dg2[mask],
            minlength=minlen,
        )
        r2_sel = sel_term(2)
        return {
            "e1": e1, "r1": r1, "r1_sel": r1_sel,
            "e2": e2, "r2": r2, "r2_sel": r2_sel,
        }
    else:
        raise ValueError(
            f"target must be 'g1', 'g2', or 'g1g2', got {target!r}"
        )
