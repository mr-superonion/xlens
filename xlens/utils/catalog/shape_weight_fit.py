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

"""Fit the optimal shape weight ``w_opt = f1(S/N) f2(|e|^2)`` on a catalog.

The weight (:mod:`xlens.utils.catalog.shape_weight`) multiplies every
object's contribution to the shear estimator

    g = sum(W e) / sum(R),   W = wsel * w_opt,

and is chosen to minimise the shear noise per object, the figure of merit

    FoM = sigma(W e) / <R>

with ``R`` the FULL response of the weighted, selected sample:

* the shape response, ``W de/dg``;
* the detection weight's response, ``dwsel/dg * w_opt * e``;
* the optimal weight's own response, ``wsel * dw_opt/dg * e`` (chain rule
  through the S/N and |e|^2, as MergePipe propagates it);
* the selection response of the hard cuts, by finite differences: every
  cut quantity ``x`` is shifted to ``x +- dg dx/dg_j`` and
  ``(sum_+ W e_j - sum_- W e_j) / (2 dg)`` is added (first order in dg).

``amp`` cancels between ``W e`` and ``R``, so only ``alpha, snr0, beta,
esq_scale`` are fitted; ``amp`` is then set so that the mean ``w_opt``
over the selected sample is 1.  The figure of merit is evaluated on fit
and held-out groups (e.g. tracts), and against the unweighted estimator
(``W = wsel``), whose ratio squared is the gain in effective number.

The catalog must carry the UNWEIGHTED shapes: a merged catalog built
with ``MergePipeConfig.shape_weight_params`` empty (MergePipe otherwise
stores ``w_opt * e`` in the shape columns).

    sample = prepare_shape_weight_sample(catalog, cuts=[
        cut_from_columns(catalog, "fpfs1_s2n", ">", 5.0),
        trace_cut(catalog, ">", 0.05),
    ], groups=catalog["tract_id"])
    result = fit_shape_weight(sample)
    result["params"]          # [amp, alpha, snr0, beta, esq_scale]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .shape_weight import (
    DEFAULT_SHAPE_WEIGHT_PARAMS,
    ds2n_column_name,
    shape_column_prefix,
    shape_weight_f1,
    shape_weight_f2,
)

__all__ = [
    "Cut",
    "cut_from_columns",
    "trace_cut",
    "prepare_shape_weight_sample",
    "shape_weight_fom",
    "fit_shape_weight",
]

_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}


@dataclass
class Cut:
    """A hard cut ``value <op> threshold`` with the shear response of the
    cut quantity, ``dvalue/dg_1`` and ``dvalue/dg_2`` (zeros for a
    quantity that does not respond, e.g. a mask fraction)."""

    value: np.ndarray
    op: str
    threshold: float
    dvalue_dg1: np.ndarray | None = None
    dvalue_dg2: np.ndarray | None = None
    name: str = ""

    def passes(self, comp: int = 0, sign: int = 0, dg: float = 0.0) -> np.ndarray:
        """The cut on the quantity shifted by ``sign * dg`` along shear
        component ``comp`` (0: unshifted)."""
        x = np.asarray(self.value, dtype=np.float64)
        if comp and sign:
            dx = self.dvalue_dg1 if comp == 1 else self.dvalue_dg2
            if dx is not None:
                x = x + sign * dg * np.asarray(dx, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            return np.isfinite(x) & _OPS[self.op](x, self.threshold)


def _response_name(column: str, comp: int) -> str:
    """AnaCal/xlens naming of the shear response of a column:
    ``esq -> desq_dg1``; ``fpfs1_s2n -> fpfs1_ds2n_dg1``;
    ``hsc_i_mag_fpfs1 -> hsc_i_dmag_fpfs1_dg1``."""
    for obs in ("s2n", "mag", "flux"):
        head, sep, tail = column.rpartition(obs)
        if sep and (not head or head.endswith("_")):
            return f"{head}d{obs}{tail}_dg{comp}"
    return f"d{column}_dg{comp}"


def cut_from_columns(catalog, column: str, op: str, threshold: float,
                     response: bool = True) -> Cut:
    """A :class:`Cut` on one catalog column, with its response columns
    found by name (or no response with ``response=False``)."""
    names = getattr(catalog, "colnames", None) or catalog.dtype.names
    d = [None, None]
    if response:
        for j in (1, 2):
            rn = _response_name(column, j)
            if rn not in names:
                raise KeyError(f"response column {rn!r} of {column!r} not found; "
                               "pass response=False for a quantity that does not respond")
            d[j - 1] = np.asarray(catalog[rn], dtype=np.float64)
    return Cut(np.asarray(catalog[column], dtype=np.float64), op, threshold,
               d[0], d[1], name=f"{column} {op} {threshold:g}")


def trace_cut(catalog, op: str, threshold: float, shape_name: str = "fpfs1") -> Cut:
    """A :class:`Cut` on the size ratio ``(m00 + m20) / m00`` of a shapelet
    set, with its response by the quotient rule."""
    p = shape_column_prefix(shape_name)
    m0 = np.asarray(catalog[f"{p}m00"], dtype=np.float64)
    m2 = np.asarray(catalog[f"{p}m20"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (m0 + m2) / m0
        d = []
        for j in (1, 2):
            dm0 = np.asarray(catalog[f"{p}dm00_dg{j}"], dtype=np.float64)
            dm2 = np.asarray(catalog[f"{p}dm20_dg{j}"], dtype=np.float64)
            d.append((dm0 + dm2) / m0 - (m0 + m2) * dm0 / m0**2)
    return Cut(t, op, threshold, d[0], d[1], name=f"trace {op} {threshold:g}")


def prepare_shape_weight_sample(
    catalog,
    cuts: Sequence[Cut] = (),
    shape_name: str = "fpfs1",
    s2n_name: str | None = None,
    esq_name: str = "esq",
    groups=None,
    dg: float = 0.01,
) -> dict:
    """The arrays the figure of merit needs, restricted to the rows that
    pass the cuts nominally or under one of the four +-dg shifts.

    Returns a dict of float64 arrays (``e1, e2, j11, j22, w, dw1, dw2,
    snr, ds1, ds2, v, dv1, dv2``), the bool selections ``s0`` (nominal)
    and ``p1, m1, p2, m2`` (shifted along g1 / g2), the ``groups`` labels
    (or zeros) and ``dg``.  Shapes are ``{shape}_e{1,2}`` with their
    diagonal responses; S/N is ``s2n_name`` (default ``{shape}_s2n``);
    |e|^2 is ``esq_name`` with ``d{esq}_dg{j}``.
    """
    p = shape_column_prefix(shape_name)
    if s2n_name is None:
        s2n_name = f"{p}s2n"
    f = lambda c: np.asarray(catalog[c], dtype=np.float64)  # noqa: E731
    n = len(catalog)
    base = np.isfinite(f(f"{p}e1")) & np.isfinite(f(f"{p}e2")) & np.isfinite(f(s2n_name))
    sel = {}
    for key, comp, sign in (("s0", 0, 0), ("p1", 1, 1), ("m1", 1, -1), ("p2", 2, 1), ("m2", 2, -1)):
        s = base.copy()
        for c in cuts:
            s &= c.passes(comp, sign, dg)
        sel[key] = s
    keep = sel["s0"] | sel["p1"] | sel["m1"] | sel["p2"] | sel["m2"]
    out = dict(
        e1=f(f"{p}e1"), e2=f(f"{p}e2"),
        j11=f(f"{p}de1_dg1"), j22=f(f"{p}de2_dg2"),
        w=f("wsel"), dw1=f("dwsel_dg1"), dw2=f("dwsel_dg2"),
        snr=f(s2n_name), ds1=f(ds2n_column_name(s2n_name, 1)), ds2=f(ds2n_column_name(s2n_name, 2)),
        v=f(esq_name), dv1=f(f"d{esq_name}_dg1"), dv2=f(f"d{esq_name}_dg2"),
    )
    out = {k: v[keep] for k, v in out.items()}
    out.update({k: v[keep] for k, v in sel.items()})
    g = np.zeros(n, dtype=np.int64) if groups is None else np.asarray(groups)
    out["groups"] = g[keep]
    out["dg"] = float(dg)
    return out


def _subset(sample: dict, mask: np.ndarray) -> dict:
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in sample.items()}


def shape_weight_fom(params, sample: dict, flat: bool = False):
    """``(FoM, parts)`` for weight parameters ``[amp, alpha, snr0, beta,
    esq_scale]`` (amp is irrelevant) or, with ``flat``, for ``W = wsel``.

    ``parts`` holds ``n``, the mean full response ``R`` and the fraction of
    it from each term (``shape, det, weight, sel``), and the scatter ``sd``.
    """
    s0 = sample["s0"]
    if flat:
        fw = np.ones_like(sample["w"])
        dfw1 = dfw2 = np.zeros_like(fw)
    else:
        _, alpha, snr0, beta, esq_scale = (float(x) for x in params)
        f1, df1 = shape_weight_f1(sample["snr"], alpha, snr0, beta)
        f2, df2 = shape_weight_f2(sample["v"], esq_scale)
        fw = f1 * f2
        dfw1 = df1 * sample["ds1"] * f2 + f1 * df2 * sample["dv1"]
        dfw2 = df1 * sample["ds2"] * f2 + f1 * df2 * sample["dv2"]
    W = sample["w"] * fw
    We1, We2 = W * sample["e1"], W * sample["e2"]
    n = int(s0.sum())
    sd = np.sqrt(0.5 * (We1[s0].var() + We2[s0].var()))
    r_shape = 0.5 * np.sum(W[s0] * (sample["j11"][s0] + sample["j22"][s0]))
    r_det = 0.5 * np.sum((sample["dw1"] * fw * sample["e1"])[s0] + (sample["dw2"] * fw * sample["e2"])[s0])
    r_wgt = 0.5 * np.sum((sample["w"] * dfw1 * sample["e1"])[s0] + (sample["w"] * dfw2 * sample["e2"])[s0])
    dg = sample["dg"]
    r_sel = 0.5 * ((We1[sample["p1"]].sum() - We1[sample["m1"]].sum())
                   + (We2[sample["p2"]].sum() - We2[sample["m2"]].sum())) / (2.0 * dg)
    tot = r_shape + r_det + r_wgt + r_sel
    R = tot / n
    parts = dict(n=n, R=R, sd=sd, shape=r_shape / tot, det=r_det / tot,
                 weight=r_wgt / tot, sel=r_sel / tot, mean_w=float(fw[s0].mean()))
    return sd / R, parts


def fit_shape_weight(
    sample: dict,
    x0: Sequence[float] = DEFAULT_SHAPE_WEIGHT_PARAMS,
    fit_groups=None,
    holdout_fraction: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Fit ``alpha, snr0, beta, esq_scale`` by minimising the figure of
    merit on the fit groups (Nelder-Mead in ``alpha, log10 snr0, beta,
    log10 esq_scale``), and report it on the held-out groups.

    ``fit_groups``: the group labels to fit on; default a random
    ``1 - holdout_fraction`` of the groups (seeded).  Returns a dict with
    ``params`` ([amp, alpha, snr0, beta, esq_scale], amp normalising the
    mean w_opt to 1 on the fit sample), ``fom_fit`` / ``fom_holdout`` and
    their unweighted counterparts, ``gain_holdout`` (the gain in effective
    number, (FoM_flat / FoM)^2), the response breakdowns and the groups.
    """
    from scipy.optimize import minimize

    groups = sample["groups"]
    ug = np.unique(groups)
    if fit_groups is None:
        rng = np.random.default_rng(seed)
        nfit = max(1, int(round(len(ug) * (1.0 - holdout_fraction))))
        fit_groups = rng.choice(ug, nfit, replace=False) if len(ug) > 1 else ug
    infit = np.isin(groups, np.asarray(fit_groups))
    fit, hold = _subset(sample, infit), _subset(sample, ~infit)
    _, a0, s0, b0, v0 = (float(x) for x in x0)

    def loss(x):
        alpha, ls0, beta, lvs = x
        if not (0.1 < alpha < 20 and -1 < ls0 < 3.5 and -3 < beta < 3 and -4 < lvs < 2):
            return 1e3
        val, _ = shape_weight_fom([1.0, alpha, 10**ls0, beta, 10**lvs], fit)
        return val if np.isfinite(val) else 1e3

    res = minimize(loss, [a0, np.log10(s0), b0, np.log10(v0)], method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-7, maxiter=4000, maxfev=8000))
    alpha, ls0, beta, lvs = res.x
    p = [1.0, alpha, 10**ls0, beta, 10**lvs]
    _, parts_fit = shape_weight_fom(p, fit)
    p[0] = 1.0 / parts_fit["mean_w"]
    out = dict(params=p, success=bool(res.success), nfev=int(res.nfev),
               fit_groups=np.asarray(fit_groups))
    for lab, sub in (("fit", fit), ("holdout", hold)):
        if sub["s0"].sum() == 0:
            continue
        fom, parts = shape_weight_fom(p, sub)
        flat, parts_flat = shape_weight_fom(p, sub, flat=True)
        out[f"fom_{lab}"], out[f"fom_flat_{lab}"] = fom, flat
        out[f"gain_{lab}"] = (flat / fom) ** 2
        out[f"parts_{lab}"], out[f"parts_flat_{lab}"] = parts, parts_flat
    if verbose:
        state = "converged" if res.success else "NOT converged"
        print("optimal shape weight [amp, alpha, snr0, beta, esq_scale] = "
              + ", ".join(f"{x:.5g}" for x in p) + f"  ({state}, {res.nfev} evals)")
        for lab in ("fit", "holdout"):
            if f"fom_{lab}" in out:
                pt = out[f"parts_{lab}"]
                print(f"  {lab:8s} n {pt['n']:9d}  FoM {out[f'fom_{lab}']:.4f} (wsel alone "
                      f"{out[f'fom_flat_{lab}']:.4f})  gain in N_eff {out[f'gain_{lab}']:.3f}  "
                      f"<R> {pt['R']:.4f} = shape {pt['shape']:+.3f} det {pt['det']:+.3f} "
                      f"weight {pt['weight']:+.3f} sel {pt['sel']:+.3f}")
    return out
