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

# from .model import w_model, w_model_derivs
from .utils import _resolve_cut, _resolve_cut_name


def get_esq(
    src: np.ndarray,
    comp: int = 1,
    dg: float = 0.0,
    sn: str = "fpfs_",
) -> np.ndarray:
    """Return |e|^2 evaluated at shear g_comp = dg to first order."""

    e = src[f"{sn}e{comp}"]
    de = src[f"{sn}de{comp}_dg{comp}"]

    comp2 = 3 - comp  # 1 to 2; 2 to 1
    e2 = src[f"{sn}e{comp2}"]
    de2 = src[f"{sn}de{comp2}_dg{comp}"]

    esq0 = e * e + e2 * e2
    return esq0 + 2.0 * dg * (e * de + e2 * de2)


def get_trace(
    src: np.ndarray,
    comp: int = 1,
    dg: float = 0.0,
    sn: str = "fpfs_",
) -> np.ndarray:
    """Return trace evaluated at shear g_comp = dg to first order."""
    dm2 = src[f"{sn}dm2_dg{comp}"]
    m2 = src[f"{sn}m2"] + dg * dm2
    dm0 = src[f"{sn}dm0_dg{comp}"]
    m0 = src[f"{sn}m0"] + dg * dm0
    trace = m2 / m0
    return trace


def _bin_count(*, idx, weights=None, minlength=0):
    return np.bincount(idx, weights=weights, minlength=minlength)[1:-1]


class ShearEstimator(object):
    def __init__(
        self,
        *,
        mag_max: float | dict = 40.0,
        emax: float = 0.3,
        trace_min: float = 0.05,
        mag_zero: float = 30.0,
        flux_name: str = "gauss2",
        shape_name: str = "fpfs",
        bands: str = "grizy",
        ref_band: str = "i",
        z_estimator=None,
        zbounds: list[float] = [0.0, 100.0],
        z_width95_max: float = 2.75,
        dg: float = 0.02,
        z_point_name: str = "zmode",
    ):
        self.fn = _resolve_cut_name(flux_name)
        self.magx = _resolve_cut(mag_max, bands=bands)
        self.dg = dg
        self.bands = bands
        self.emax2 = emax * emax
        self.trace_min = trace_min
        self.z_estimator = z_estimator
        self.flux_name = flux_name
        self.ref_band = ref_band
        self.mag_zero = mag_zero
        self.zbounds = zbounds
        self.z_width95_max = z_width95_max
        self.z_point_name = z_point_name
        if len(shape_name) > 0:
            self.sn = shape_name + "_"
        else:
            self.sn = ""

    def _measure(
        self,
        src,
        comp: int,
        sign: float,
        extinction: np.ndarray | None = None,
    ):
        """Compute binned <w_sel e> for shear +sign*dg."""
        fn = self.fn
        e_comp = src[f"{self.sn}e{comp}"]
        de_dg = src[f"{self.sn}de{comp}_dg{comp}"]

        wsel = src["wsel"]
        dw_dg = src[f"dwsel_dg{comp}"]

        dg_eff = sign * self.dg
        esq_s = get_esq(src, comp=comp, dg=dg_eff, sn=self.sn)
        trace_s = get_trace(src, comp=comp, dg=dg_eff, sn=self.sn)
        mask_s = (esq_s < self.emax2) & (trace_s > self.trace_min)
        for b in self.bands:
            _f = src[f"{b}_flux{fn}"] + dg_eff * src[f"{b}_dflux{fn}_dg{comp}"]
            _m = np.full(len(src), 40.0, dtype=np.float64)
            _p = _f > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                _m[_p] = self.mag_zero - 2.5 * np.log10(_f[_p])
            if extinction is not None:
                _m = _m - extinction[f"a_{b}"]
            mask_s &= _m < self.magx[b]
        if extinction is None:
            ext = None
        else:
            ext = extinction[mask_s]

        if self.z_estimator is not None:
            z_s, w_s = self.z_estimator.get_zsel(
                src[mask_s],
                mag_zero=self.mag_zero,
                flux_name=self.flux_name,
                bands=self.bands,
                ref_band=self.ref_band,
                comp=comp,
                dg=dg_eff,
                include_mag_err=False,
                z_point_name=self.z_point_name,
                extinction=ext,
            )
            mtmp_local = w_s < self.z_width95_max
            mask_s[mask_s] &= mtmp_local
            z_s = z_s[mtmp_local]
            del mtmp_local, w_s
            idx_s = np.digitize(z_s, self.zbounds, right=False)
            minlen = len(self.zbounds) + 1
        else:
            idx_s = np.ones(np.sum(mask_s.astype(int)))
            minlen = 3

        we = wsel[mask_s] * e_comp[mask_s]
        response = wsel[mask_s] * de_dg[mask_s]
        response_det = dw_dg[mask_s] * e_comp[mask_s]
        ell_s = _bin_count(
            idx=idx_s,
            weights=we,
            minlength=minlen,
        )
        response_s = _bin_count(
            idx=idx_s,
            weights=response,
            minlength=minlen,
        )
        response_det_s = _bin_count(
            idx=idx_s,
            weights=response_det,
            minlength=minlen,
        )
        num_s = _bin_count(
            idx=idx_s,
            weights=None,
            minlength=minlen,
        )
        return ell_s, response_s, response_det_s, num_s

    def get_sel_response(
        self,
        src,
        comp: int,
        extinction: np.ndarray | None = None,
    ) -> np.ndarray:
        """Selection response term for component comp (1 or 2)."""
        ellp, _, _, _ = self._measure(src, comp, +1.0, extinction=extinction)
        ellm, _, _, _ = self._measure(src, comp, -1.0, extinction=extinction)
        return (ellp - ellm) / (2.0 * self.dg)

    def measure_shear(
        self,
        src: np.ndarray,
        target: str,
        extinction: np.ndarray | None = None,
    ):
        """Measure shear components in redshift bins via a z-estimator."""
        if target == "g1":
            e1, r1, r1_det, num1 = self._measure(
                src,
                comp=1,
                sign=0,
                extinction=extinction,
            )
            r1_sel = self.get_sel_response(
                src,
                comp=1,
                extinction=extinction,
            )
            return {"e": e1, "r": r1, "r_det": r1_det, "r_sel": r1_sel, "num": num1}
        elif target == "g2":
            e2, r2, r2_det, num2 = self._measure(
                src,
                comp=2,
                sign=0,
                extinction=extinction,
            )
            r2_sel = self.get_sel_response(
                src,
                comp=2,
                extinction=extinction,
            )
            return {"e": e2, "r": r2, "r_det": r2_det, "r_sel": r2_sel, "num": num2}
        elif target == "g1g2":
            e1, r1, r1_det, num1 = self._measure(
                src,
                comp=1,
                sign=0,
                extinction=extinction,
            )
            e2, r2, r2_det, num2 = self._measure(
                src,
                comp=2,
                sign=0,
                extinction=extinction,
            )
            r1_sel = self.get_sel_response(
                src,
                1,
                extinction=extinction,
            )
            r2_sel = self.get_sel_response(
                src,
                2,
                extinction=extinction,
            )
            return {
                "e1": e1,
                "r1": r1,
                "r1_det": r1_det,
                "r1_sel": r1_sel,
                "e2": e2,
                "r2": r2,
                "r2_det": r2_det,
                "r2_sel": r2_sel,
                "num": num1,
            }
        else:
            raise ValueError(f"target must be 'g1', 'g2', or 'g1g2', got {target!r}")


def measure_shear(
    *,
    src: np.ndarray,
    z_estimator,
    zbounds: list[float],
    flux_min: float | dict = 40.0,
    emax: float = 0.3,
    trace_min: float = 0.05,
    z_width95_max: float = 2.75,
    dg: float = 0.02,
    target: str = "g1",
    do_correction: bool = True,
    mag_zero: float = 30.0,
    flux_name: str = "gauss2",
    bands: str = "grizy",
    ref_band: str = "i",
    z_point_name: str = "zmode",
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
    fn = _resolve_cut_name(flux_name)
    esq0 = get_esq(src)
    trace0 = get_trace(src)

    # band-independent fields
    e1_all = src["fpfs_e1"]
    e2_all = src["fpfs_e2"]
    de1_dg1 = src["fpfs_de1_dg1"]
    de2_dg2 = src["fpfs_de2_dg2"]
    wopt = src["wsel"]

    dw_dg1 = src["dwsel_dg1"]
    dw_dg2 = src["dwsel_dg2"]

    # per-band flux minima and base fluxes
    fm = _resolve_cut(flux_min, bands=bands)
    flux = {b: src[f"{b}_flux{fn}"] for b in bands}

    # No shear
    mask = np.ones(src.shape[0], dtype=bool)
    for b in bands:
        mask &= flux[b] > fm[b]
    mask &= esq0 < emax * emax
    mask &= trace0 > trace_min
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
        z_point_name=z_point_name,
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
            trace_side = get_trace(src, comp=comp, dg=dg_eff)
            mask_side = (esq_side < emax * emax) & (trace_side > trace_min)
            for b in bands:
                df = src[f"{b}_dflux{fn}_dg{comp}"]
                mask_side &= flux[b] + dg_eff * df > fm[b]

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
                    z_point_name=z_point_name,
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
                    z_point_name=z_point_name,
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
            del esq_side, trace_side, mask_side, idx_side
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
            "e1": e1,
            "r1": r1,
            "r1_sel": r1_sel,
            "e2": e2,
            "r2": r2,
            "r2_sel": r2_sel,
        }
    else:
        raise ValueError(f"target must be 'g1', 'g2', or 'g1g2', got {target!r}")
