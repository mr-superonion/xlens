import numpy as np

# from .model import w_model, w_model_derivs
from .utils import _resolve_flux_min, _resolve_flux_name


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


def _bin_count(*, idx, weights=None, minlength=0):
    return np.bincount(idx, weights=weights, minlength=minlength)[1:-1]


class ShearEstimator(object):
    def __init__(
        self,
        *,
        flux_min: float | dict = 40.0,
        emax: float = 0.3,
        mag_zero: float = 30.0,
        flux_name: str = "gauss2",
        bands: str = "grizy",
        ref_band: str = "i",
        z_estimator=None,
        zbounds: list[float] = [0.0, 100.0],
        z_width95_max: float = 2.75,
        dg: float = 0.02,
    ):
        self.fn = _resolve_flux_name(flux_name)
        self.fm = _resolve_flux_min(flux_min, bands=bands)
        self.dg = dg
        self.bands = bands
        self.emax2 = emax * emax
        self.z_estimator = z_estimator
        self.flux_name = flux_name
        self.ref_band = ref_band
        self.mag_zero = mag_zero
        self.zbounds = zbounds
        self.z_width95_max = z_width95_max

    def _measure(self, src, comp: int, sign: float):
        """Compute binned <w_sel e> for shear +sign*dg."""
        fn = self.fn
        e_comp = src[f"fpfs_e{comp}"]
        de_dg = src[f"fpfs_de{comp}_dg{comp}"]

        wsel = src["wsel"]
        dw_dg = src[f"dwsel_dg{comp}"]

        dg_eff = sign * self.dg
        esq_side = get_esq(src, comp=comp, dg=dg_eff)
        mask_side = esq_side < self.emax2
        for b in self.bands:
            df = src[f"{b}_dflux{fn}_dg{comp}"]
            mask_side &= (src[f"{b}_flux{fn}"] + dg_eff * df > self.fm[b])

        if self.z_estimator is not None:
            z_side, w_side = self.z_estimator.get_zsel(
                src[mask_side],
                mag_zero=self.mag_zero,
                flux_name=self.flux_name,
                bands=self.bands,
                ref_band=self.ref_band,
                comp=comp,
                dg=dg_eff,
                include_mag_err=False,
            )
            mtmp_local = w_side < self.z_width95_max
            mask_side[mask_side] &= mtmp_local
            z_side = z_side[mtmp_local]
            del mtmp_local, w_side
            idx_side = np.digitize(z_side, self.zbounds, right=False)
            minlen = len(self.zbounds) + 1
        else:
            idx_side = np.ones(np.sum(mask_side.astype(int)))
            minlen = 3

        we = wsel[mask_side] * e_comp[mask_side]
        response = (
            dw_dg[mask_side] * e_comp[mask_side]
            + wsel[mask_side] * de_dg[mask_side]
        )
        ell_side = _bin_count(
            idx=idx_side,
            weights=we,
            minlength=minlen,
        )
        response_side = _bin_count(
            idx=idx_side,
            weights=response,
            minlength=minlen,
        )
        num_side = _bin_count(
            idx=idx_side,
            weights=None,
            minlength=minlen,
        )
        return ell_side, response_side, num_side

    def get_sel_response(self, src, comp: int) -> np.ndarray:
        """Selection response term for component comp (1 or 2)."""
        ellp, _, _ = self._measure(src, comp, +1.0)
        ellm, _, _ = self._measure(src, comp, -1.0)
        return (ellp - ellm) / (2.0 * self.dg)

    def measure_shear(
        self,
        src: np.ndarray,
        target: str,
    ):
        """
        Measure shear components in redshift bins, using a supplied z-estimator.
        """
        if target == "g1":
            e1, r1, num1 = self._measure(src, comp=1, sign=0)
            r1_sel = self.get_sel_response(src, comp=1)
            return {"e": e1, "r": r1, "r_sel": r1_sel, "num": num1}
        elif target == "g2":
            e2, r2, num2 = self._measure(src, comp=2, sign=0)
            r2_sel = self.get_sel_response(src, comp=2)
            return {"e": e2, "r": r2, "r_sel": r2_sel, "num": num2}
        elif target == "g1g2":
            e1, r1, num1 = self._measure(src, comp=1, sign=0)
            e2, r2, num2 = self._measure(src, comp=2, sign=0)
            r1_sel = self.get_sel_response(src, 1)
            r2_sel = self.get_sel_response(src, 2)
            return {
                "e1": e1, "r1": r1, "r1_sel": r1_sel,
                "e2": e2, "r2": r2, "r2_sel": r2_sel,
                "num": num1
            }
        else:
            raise ValueError(
                f"target must be 'g1', 'g2', or 'g1g2', got {target!r}"
            )


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

    # band-independent fields
    e1_all = src["fpfs_e1"]
    e2_all = src["fpfs_e2"]
    de1_dg1 = src["fpfs_de1_dg1"]
    de2_dg2 = src["fpfs_de2_dg2"]
    wopt = src["wsel"]

    dw_dg1 = src["dwsel_dg1"]
    dw_dg2 = src["dwsel_dg2"]

    # per-band flux minima and base fluxes
    fm = _resolve_flux_min(flux_min, bands=bands)
    flux = {b: src[f"{b}_flux{fn}"] for b in bands}

    # No shear
    mask = np.ones(src.shape[0], dtype=bool)
    for b in bands:
        mask &= flux[b] > fm[b]
    mask &= esq0 < emax * emax
    # photo-z + width cut at base shear
    zmode, width95 = z_estimator.get_zsel(
        src[mask],
        mag_zero=mag_zero,
        flux_name=flux_name,
        bands=bands,
        ref_band=ref_band,
        comp=1,
        dg=0.0,
        include_mag_err=False,
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

            if do_correction:
                z_side, w_side = z_estimator.get_zsel(
                    src[mask_side],
                    mag_zero=mag_zero,
                    flux_name=flux_name,
                    bands=bands,
                    ref_band=ref_band,
                    comp=comp,
                    dg=dg_eff,
                    include_mag_err=False,
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
                    include_mag_err=False,
                )

            mtmp_local = w_side < z_width95_max
            mask_side[mask_side] &= mtmp_local
            z_side = z_side[mtmp_local]
            del mtmp_local, w_side

            idx_side = np.digitize(z_side, zbounds, right=False)
            ell_side = _bin_count(
                weights=wopt[mask_side] * e_comp[mask_side],
                idx=idx_side,
                minlength=minlen,
            )
            del esq_side, mask_side, idx_side
            return ell_side

        ellp = one_side(+1.0)
        ellm = one_side(-1.0)
        return (ellp - ellm) / (2.0 * dg)

    idx0 = np.digitize(zmode, zbounds, right=False)
    if target == "g1":
        e1 = _bin_count(
            weights=wopt[mask] * e1_all[mask],
            idx=idx0,
            minlength=minlen,
        )
        r1 = _bin_count(
            weights=dw_dg1[mask] * e1_all[mask] + wopt[mask] * de1_dg1[mask],
            idx=idx0,
            minlength=minlen,
        )
        r1_sel = sel_term(1)
        return {"e": e1, "r": r1, "r_sel": r1_sel}
    elif target == "g2":
        e2 = _bin_count(
            weights=wopt[mask] * e2_all[mask],
            idx=idx0,
            minlength=minlen,
        )
        r2 = _bin_count(
            weights=dw_dg2[mask] * e2_all[mask] + wopt[mask] * de2_dg2[mask],
            idx=idx0,
            minlength=minlen,
        )
        r2_sel = sel_term(2)
        return {"e": e2, "r": r2, "r_sel": r2_sel}
    elif target == "g1g2":
        e1 = _bin_count(
            weights=wopt[mask] * e1_all[mask],
            idx=idx0,
            minlength=minlen,
        )
        r1 = _bin_count(
            weights=dw_dg1[mask] * e1_all[mask] + wopt[mask] * de1_dg1[mask],
            idx=idx0,
            minlength=minlen,
        )
        r1_sel = sel_term(1)
        e2 = _bin_count(
            weights=wopt[mask] * e2_all[mask],
            idx=idx0,
            minlength=minlen,
        )
        r2 = _bin_count(
            weights=dw_dg2[mask] * e2_all[mask] + wopt[mask] * de2_dg2[mask],
            idx=idx0,
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
