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
from ..constants import MAG_ZERO_AB
from .utils import _resolve_cut, _resolve_cut_name, flux_to_mag


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
    """Per-bin sums of ``weights`` (counts when None) for ``np.digitize`` indices.

    The under- and overflow bins are dropped, so ``len(zbounds)`` edges give
    ``len(zbounds) - 1`` interior bins.
    """
    return np.bincount(idx, weights=weights, minlength=minlength)[1:-1]


def _accumulate(weights, idx, minlength):
    """Sum ``weights`` per redshift bin, or over everything when unbinned.

    ``idx=None`` is the no-tomography case: a single number for the whole
    sample, returned as a 1-element array so callers see the same "one
    value per bin" shape either way, and computed with a plain sum instead
    of a bincount.
    """
    if idx is None:
        return np.array([np.sum(weights)], dtype=np.float64)
    return _bin_count(idx=idx, weights=weights, minlength=minlength)


def _count(mask, idx, minlength):
    """Number of selected sources per bin (or in total when ``idx`` is None)."""
    if idx is None:
        return np.array([np.count_nonzero(mask)], dtype=np.float64)
    return _bin_count(idx=idx, minlength=minlength).astype(np.float64)


def _check_binning(z_estimator, zbounds):
    """Validate the (z_estimator, zbounds) pair; -> True when binned."""
    if z_estimator is None:
        if zbounds is not None:
            raise ValueError(
                "zbounds given without a z_estimator: there is no redshift to "
                "bin on.  Pass a z_estimator for tomographic bins, or drop "
                "zbounds for a single bin over the whole sample."
            )
        return False
    if zbounds is None or len(zbounds) < 2:
        raise ValueError(
            "a z_estimator needs zbounds with at least two edges "
            f"(got {zbounds!r}); len(zbounds) - 1 interior bins are returned"
        )
    return True


def build_selection_mask(
    src,
    *,
    comp: int = 1,
    dg: float = 0.0,
    bands: tuple[str, ...] = ("lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"),
    mag_max: float | dict = 40.0,
    mag_zero: float = MAG_ZERO_AB,
    flux_name: str = "fpfs1",
    shape_name: str = "fpfs",
    emax: float = 0.3,
    trace_min: float | None = 0.05,
    extinction: np.ndarray | None = None,
    z_estimator=None,
    z_width95_max: float = 2.75,
    ref_band: str = "lsst_i",
    z_point_name: str = "zmode",
    include_mag_err: bool = False,
):
    """Build a selection mask at first order in shear g_comp = dg.

    Combines, in order:
      1. shape ``|e|^2 < emax^2``  (``get_esq``, evaluated at ``dg``)
      2. size  ``trace > trace_min`` (``get_trace``, evaluated at ``dg``);
         skipped entirely when ``trace_min`` is None -- note that 0.0 is a
         real cut, it drops sources whose monopole is not positive
      3. per-band mag cut: ``mag_b < mag_max[b]`` with
         ``flux_b = src[{b}_flux{fn}] + dg * src[{b}_dflux{fn}_dg{comp}]``
         and ``mag_b = mag_zero - 2.5*log10(flux_b)``
         (optionally minus the per-band extinction ``extinction[f'a_{b}']``).
      4. optional photo-z width cut via ``z_estimator.get_zsel``.

    Parameters
    ----------
    src : structured ndarray
        AnaCal catalog rows; must carry the columns referenced above.
    comp : {1, 2}
        Shear component for the dg perturbation.
    dg : float
        Shear value at which to evaluate the cut (``0.0`` for the
        unperturbed selection; ``+self.dg`` / ``-self.dg`` for the
        selection-response terms).
    bands, mag_max, mag_zero, flux_name, shape_name, emax, trace_min, ...
        Same meaning as on ``ShearEstimator``; ``mag_max`` may be a scalar
        (broadcast across bands) or a ``{band: value}`` dict.
    z_estimator : optional
        If given, the photo-z estimator is queried at the dg-perturbed
        magnitudes and rows with ``w_s >= z_width95_max`` are dropped.

    Returns
    -------
    mask : ndarray of bool, shape (N,)
        Source selection. Length matches ``len(src)``.
    z_s : ndarray or None
        Photo-z point estimate for the rows kept by the mask, or
        ``None`` when no ``z_estimator`` is supplied. Caller can use
        this for downstream redshift binning without recomputing it.
    """
    fn = _resolve_cut_name(flux_name)
    magx = _resolve_cut(mag_max, bands=bands)
    sn = shape_name + "_" if (shape_name and not shape_name.endswith("_")) else (shape_name or "")
    emax2 = emax * emax

    mask = get_esq(src, comp=comp, dg=dg, sn=sn) < emax2
    if trace_min is not None:
        mask &= get_trace(src, comp=comp, dg=dg, sn=sn) > trace_min
    for b in bands:
        flux_b = src[f"{b}_flux{fn}"] + dg * src[f"{b}_dflux{fn}_dg{comp}"]
        a_ext = None if extinction is None else extinction[f"a_{b}"]
        mag_b, _ = flux_to_mag(flux_b, mag_zero, a_ext=a_ext)
        mask &= mag_b < magx[b]

    z_s = None
    if z_estimator is not None:
        ext = None if extinction is None else extinction[mask]
        z_s, w_s = z_estimator.get_zsel(
            src[mask],
            mag_zero=mag_zero,
            flux_name=flux_name,
            bands=bands,
            ref_band=ref_band,
            comp=comp,
            dg=dg,
            include_mag_err=include_mag_err,
            z_point_name=z_point_name,
            extinction=ext,
        )
        keep_z = w_s < z_width95_max
        mask[mask] &= keep_z
        z_s = z_s[keep_z]

    return mask, z_s


class ShearEstimator(object):
    def __init__(
        self,
        *,
        mag_max: float | dict = 40.0,
        emax: float = 0.3,
        trace_min: float | None = 0.05,
        mag_zero: float = MAG_ZERO_AB,
        flux_name: str = "gauss2",
        shape_name: str = "fpfs",
        bands: tuple[str, ...] = ("lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"),
        ref_band: str = "lsst_i",
        z_estimator=None,
        zbounds: list[float] | None = None,
        z_width95_max: float = 2.75,
        dg: float = 0.02,
        z_point_name: str = "zmode",
    ):
        """Shear estimator with magnitude, shape and size cuts.

        With ``z_estimator`` and ``zbounds`` the outputs are per redshift
        bin (``len(zbounds) - 1`` interior bins).  With neither, there is
        no tomographic binning: every output is a 1-element array for the
        whole sample, no photo-z is evaluated and nothing is digitized.
        Giving one without the other is an error.  ``trace_min=None`` skips
        the size cut.
        """
        self.binned = _check_binning(z_estimator, zbounds)
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
        e_comp = src[f"{self.sn}e{comp}"]
        de_dg = src[f"{self.sn}de{comp}_dg{comp}"]

        wsel = src["wsel"]
        dw_dg = src[f"dwsel_dg{comp}"]

        dg_eff = sign * self.dg
        mask_s, z_s = build_selection_mask(
            src,
            comp=comp,
            dg=dg_eff,
            bands=self.bands,
            mag_max=self.magx,
            mag_zero=self.mag_zero,
            flux_name=self.flux_name,
            shape_name=self.sn,
            emax=np.sqrt(self.emax2),
            trace_min=self.trace_min,
            extinction=extinction,
            z_estimator=self.z_estimator,
            z_width95_max=self.z_width95_max,
            ref_band=self.ref_band,
            z_point_name=self.z_point_name,
            include_mag_err=False,
        )
        if z_s is not None:
            idx_s = np.digitize(z_s, self.zbounds, right=False)
            minlen = len(self.zbounds) + 1
        else:
            # no tomographic binning: plain sums over the selection
            idx_s, minlen = None, 0

        w_s = wsel[mask_s]
        e_s = e_comp[mask_s]
        ell_s = _accumulate(w_s * e_s, idx_s, minlen)
        response_s = _accumulate(w_s * de_dg[mask_s], idx_s, minlen)
        response_det_s = _accumulate(dw_dg[mask_s] * e_s, idx_s, minlen)
        num_s = _count(mask_s, idx_s, minlen)
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
    z_estimator=None,
    zbounds: list[float] | None = None,
    flux_min: float | dict = 40.0,
    emax: float = 0.3,
    trace_min: float | None = 0.05,
    z_width95_max: float = 2.75,
    dg: float = 0.02,
    target: str = "g1",
    do_correction: bool = True,
    mag_zero: float = MAG_ZERO_AB,
    flux_name: str = "gauss2",
    bands: tuple[str, ...] = ("lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"),
    ref_band: str = "lsst_i",
    z_point_name: str = "zmode",
):
    """
    Measure shear components, in redshift bins or over the whole sample.

    Parameters
    ----------
    z_estimator : object or None
        Must provide
          get_zsel(src, *, mag_zero, flux_name, bands, ref_band, comp, dg,
                   ...) -> (zpoint, width95).
        None means no tomographic binning: every output is a 1-element
        array for the whole sample, the estimator is never called (so no
        photo-z cost and no per-cut catalog copies), and nothing is
        digitized.
    zbounds : list of float or None
        Bin edges; ``len(zbounds) - 1`` interior bins are returned, sources
        outside the outer edges are dropped.  Required with ``z_estimator``,
        must be None without it.
    trace_min : float or None
        Size cut ``(m00 + m20) / m00 > trace_min``.  None skips the cut and
        never computes the trace; 0.0 is a real cut that drops sources whose
        monopole is not positive.

    Returns
    -------
    dict of arrays, one value per bin: ``e``, ``r``, ``r_sel``, ``n_gal``
    (``e1``/``r1``/``r1_sel``, ``e2``/... and ``n_gal`` for target "g1g2").
    """
    binned = _check_binning(z_estimator, zbounds)
    minlen = len(zbounds) + 1 if binned else 0
    fn = _resolve_cut_name(flux_name)
    emax2 = emax * emax
    wopt = src["wsel"]

    # per-band flux minima and base fluxes
    fm = _resolve_cut(flux_min, bands=bands)
    flux = {b: src[f"{b}_flux{fn}"] for b in bands}

    def select(comp: int, dg_eff: float) -> np.ndarray:
        """Flux, shape and size cuts at shear g_comp = dg_eff."""
        mask = get_esq(src, comp=comp, dg=dg_eff) < emax2
        if trace_min is not None:
            mask &= get_trace(src, comp=comp, dg=dg_eff) > trace_min
        for b in bands:
            if dg_eff == 0.0:
                mask &= flux[b] > fm[b]
            else:
                df = src[f"{b}_dflux{fn}_dg{comp}"]
                mask &= flux[b] + dg_eff * df > fm[b]
        return mask

    def bin_index(mask: np.ndarray, comp: int, dg_eff: float):
        """Apply the photo-z width cut in place; -> digitized bin index.

        Returns None when unbinned, in which case ``mask`` is untouched and
        the estimator is never consulted.
        """
        if not binned:
            return None
        z, width95 = z_estimator.get_zsel(
            src[mask],
            mag_zero=mag_zero,
            flux_name=flux_name,
            bands=bands,
            ref_band=ref_band,
            comp=comp,
            dg=dg_eff,
            include_mag_err=False,
            z_point_name=z_point_name,
        )
        keep = width95 < z_width95_max
        mask[mask] &= keep
        return np.digitize(z[keep], zbounds, right=False)

    # the unperturbed selection (comp is irrelevant at dg = 0)
    mask0 = select(1, 0.0)
    idx0 = bin_index(mask0, 1, 0.0)
    n_gal = _count(mask0, idx0, minlen)
    w0 = wopt[mask0]

    def sel_term(comp: int) -> np.ndarray:
        """Selection response for component comp (1 or 2) by finite difference."""
        e_comp = src[f"fpfs_e{comp}"]

        def one_side(sign: float) -> np.ndarray:
            dg_eff = sign * dg
            mask = select(comp, dg_eff)
            # without the correction the photo-z is evaluated at the
            # unperturbed magnitudes even though the cuts are perturbed
            idx = bin_index(mask, comp, dg_eff if do_correction else 0.0)
            return _accumulate(wopt[mask] * e_comp[mask], idx, minlen)

        return (one_side(+1.0) - one_side(-1.0)) / (2.0 * dg)

    def component(comp: int):
        """-> (e, r, r_sel) for one shear component."""
        e = src[f"fpfs_e{comp}"][mask0]
        de = src[f"fpfs_de{comp}_dg{comp}"][mask0]
        dw = src[f"dwsel_dg{comp}"][mask0]
        return (
            _accumulate(w0 * e, idx0, minlen),
            _accumulate(dw * e + w0 * de, idx0, minlen),
            sel_term(comp),
        )

    if target == "g1":
        e1, r1, r1_sel = component(1)
        return {"e": e1, "r": r1, "r_sel": r1_sel, "n_gal": n_gal}
    elif target == "g2":
        e2, r2, r2_sel = component(2)
        return {"e": e2, "r": r2, "r_sel": r2_sel, "n_gal": n_gal}
    elif target == "g1g2":
        e1, r1, r1_sel = component(1)
        e2, r2, r2_sel = component(2)
        return {
            "e1": e1,
            "r1": r1,
            "r1_sel": r1_sel,
            "e2": e2,
            "r2": r2,
            "r2_sel": r2_sel,
            "n_gal": n_gal,
        }
    else:
        raise ValueError(f"target must be 'g1', 'g2', or 'g1g2', got {target!r}")
