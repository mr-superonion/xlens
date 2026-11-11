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

"""The optimal shape weight ``w_opt = f1(S/N) f2(|e|^2)`` and its
application to an FPFS shape catalog, ``e_opt = w_opt * e``.

One parameter vector ``[amp, alpha, snr0, beta, esq_scale]`` serves every
consumer: :class:`xlens.processor.merge.MergePipe` and
:class:`xlens.analysis.shear_diagnostics.ShearStatsPipe` on merged
catalogs (``esq`` / ``fpfs1_s2n`` columns), and
:func:`xlens.utils.catalog.measure_shear` and the ``bin/`` summary
scripts on per-image catalogs, where |e|^2 and its response are built
from the shape and the S/N comes from a per-band column.

``f2`` is a smooth Gaussian down-weighting of large |e|^2 and never
reaches zero: there is NO cut on |e|.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "shape_weight_f1",
    "shape_weight_f2",
    "parse_shape_weight_params",
    "optimal_shape_weight",
    "shape_weight_from_columns",
    "apply_optimal_shape_weight",
    "ds2n_column_name",
    "shape_column_prefix",
    "add_shape_weight_arguments",
    "shape_weight_params_from_args",
    "DEFAULT_SHAPE_WEIGHT_PARAMS",
    "DEFAULT_SHAPE_WEIGHT_S2N",
    "SHAPE_WEIGHT_PARAM_NAMES",
]

#: Parameter vector layout; the four-element form without ``amp`` means
#: amp 1.
SHAPE_WEIGHT_PARAM_NAMES = ("amp", "alpha", "snr0", "beta", "esq_scale")
#: The only parameter set recorded so far (the MergePipe test fixture);
#: a placeholder until a survey or simulation fit is stored.
DEFAULT_SHAPE_WEIGHT_PARAMS = (3.0, 1.6, 14.0, 0.06, 0.0875)
#: S/N column the per-image (unmerged) catalogs carry: the i-band forced
#: FPFS flux S/N written by AnaCal, with ``lsst_i_ds2n_fpfs1_dg{1,2}``.
DEFAULT_SHAPE_WEIGHT_S2N = "lsst_i_s2n_fpfs1"


def shape_column_prefix(shape_name: str | None) -> str:
    """``"fpfs" -> "fpfs_"``, ``"fpfs1_" -> "fpfs1_"``, ``"" -> ""``: the
    prefix of the ``e1`` / ``de1_dg1`` ... columns of a shape set."""
    if not shape_name:
        return ""
    return shape_name if shape_name.endswith("_") else shape_name + "_"


def shape_weight_f1(s, alpha, snr0, beta, amp=1.0):
    """``f1(s) = amp * (s/50)**beta / (1 + (snr0/s)**alpha)`` and
    ``df1/ds``."""
    s = np.maximum(np.asarray(s, dtype=np.float64), 1e-3)
    r = (snr0 / s) ** alpha
    f = amp * (s / 50.0) ** beta / (1.0 + r)
    return f, f * (beta / s + alpha * r / (s * (1.0 + r)))


def shape_weight_f2(v, esq_scale):
    """``f2(v) = exp(-v/esq_scale)`` and ``df2/dv``; a smooth down-weighting
    of large |e|^2 that never reaches zero -- there is no cut on |e|."""
    v = np.asarray(v, dtype=np.float64)
    ex = np.exp(-v / esq_scale)
    return ex, -ex / esq_scale


def parse_shape_weight_params(
    params: Sequence[float],
) -> tuple[float, float, float, float, float]:
    """``[amp, alpha, snr0, beta, esq_scale]`` (or the four-element form
    without amp) -> ``(amp, alpha, snr0, beta, esq_scale)``."""
    params = [float(p) for p in params]
    if len(params) == len(SHAPE_WEIGHT_PARAM_NAMES) - 1:
        params = [1.0] + params
    if len(params) != len(SHAPE_WEIGHT_PARAM_NAMES):
        names = list(SHAPE_WEIGHT_PARAM_NAMES)
        raise ValueError(
            f"shape weight parameters must be {names[1:]} or {names}, "
            f"got {params}"
        )
    amp, alpha, snr0, beta, esq_scale = params
    if esq_scale <= 0.0 or snr0 <= 0.0:
        raise ValueError("shape weight snr0 and esq_scale must be positive")
    return amp, alpha, snr0, beta, esq_scale


def optimal_shape_weight(s2n, esq, amp, alpha, snr0, beta, esq_scale):
    """``w_opt = f1(s2n) f2(esq)`` with its partial derivatives.

    Returns ``(w, dw_ds, dw_dv)`` so that the shear response of the
    weight is ``dw/dg = dw_ds * ds2n/dg + dw_dv * desq/dg``.
    """
    f1, df1 = shape_weight_f1(s2n, alpha, snr0, beta, amp)
    f2, df2 = shape_weight_f2(esq, esq_scale)
    return f1 * f2, df1 * f2, f1 * df2


def ds2n_column_name(s2n_name: str, comp: int) -> str:
    """Name of the shear response of the S/N column ``s2n_name``.

    AnaCal/xlens put ``d`` in front of the observable and ``_dg{comp}``
    at the end: ``lsst_i_s2n_fpfs1 -> lsst_i_ds2n_fpfs1_dg1`` and
    ``fpfs1_s2n -> fpfs1_ds2n_dg1``.
    """
    head, sep, tail = s2n_name.rpartition("s2n")
    if not sep:
        raise ValueError(f"{s2n_name!r} is not an s2n column name")
    return f"{head}ds2n{tail}_dg{comp}"


def shape_weight_from_columns(
    src,
    params: Sequence[float],
    shape_name: str = "fpfs",
    s2n_name: str = DEFAULT_SHAPE_WEIGHT_S2N,
    esq_name: str | None = None,
):
    """``w_opt`` and its shear responses, read off a catalog.

    Returns ``(w_opt, dw_opt_dg1, dw_opt_dg2)`` with ``dw_opt/dg_j =
    dw/ds * ds2n/dg_j + dw/dv * desq/dg_j``.  The S/N comes from
    ``s2n_name`` and its responses from :func:`ds2n_column_name`.  |e|^2
    comes from ``esq_name`` and ``d{esq_name}_dg{j}`` when given (the
    merged-catalog columns ``esq`` / ``desq_dg{j}``), otherwise it is
    built from the shape set ``{shape_name}_e{1,2}`` and the four
    ``{shape_name}_de{i}_dg{j}`` columns, as on a per-image catalog.

    ``src`` is an astropy Table or a numpy structured array.  The caller
    folds the result into the detection weight as ``w = wsel * w_opt``
    and ``dw/dg = dwsel/dg * w_opt + wsel * dw_opt/dg``.
    """
    amp, alpha, snr0, beta, esq_scale = parse_shape_weight_params(params)
    sn = shape_column_prefix(shape_name)
    s = np.asarray(src[s2n_name], dtype=np.float64)
    if esq_name is None:
        e1 = np.asarray(src[f"{sn}e1"], dtype=np.float64)
        e2 = np.asarray(src[f"{sn}e2"], dtype=np.float64)
        v = e1 * e1 + e2 * e2
        dv = [
            2.0 * (
                e1 * np.asarray(src[f"{sn}de1_dg{j}"], dtype=np.float64)
                + e2 * np.asarray(src[f"{sn}de2_dg{j}"], dtype=np.float64)
            )
            for j in (1, 2)
        ]
    else:
        v = np.asarray(src[esq_name], dtype=np.float64)
        dv = [
            np.asarray(src[f"d{esq_name}_dg{j}"], dtype=np.float64)
            for j in (1, 2)
        ]
    w, dw_ds, dw_dv = optimal_shape_weight(
        s, v, amp, alpha, snr0, beta, esq_scale
    )
    dw = [
        dw_ds * np.asarray(src[ds2n_column_name(s2n_name, j)], dtype=np.float64)
        + dw_dv * dv[j - 1]
        for j in (1, 2)
    ]
    return w, dw[0], dw[1]


def apply_optimal_shape_weight(
    catalog,
    params: Sequence[float],
    shape_name: str = "fpfs1",
    s2n_name: str | None = None,
    esq_name: str = "esq",
):
    """Apply ``w_opt`` to the shape columns of a merged catalog IN PLACE.

    ``{shape_name}_e{i} <- w_opt * e_i`` and, for every response column
    present, ``{shape_name}_de{i}_dg{j} <- w_opt * de_i/dg_j +
    dw_opt/dg_j * e_i``.  ``wsel`` and ``esq`` (the unweighted |e|^2 that
    f2 is defined on) are untouched and no column is added, so downstream
    code keeps using ``e = wsel * e`` and ``r = wsel * de/dg + dwsel/dg *
    e``.  ``s2n_name`` defaults to the band-combined ``{shape_name}_s2n``.
    Returns ``w_opt``.
    """
    sn = shape_column_prefix(shape_name)
    if s2n_name is None:
        s2n_name = f"{sn}s2n"
    w, dw1, dw2 = shape_weight_from_columns(
        catalog, params, shape_name=sn, s2n_name=s2n_name, esq_name=esq_name
    )
    names = getattr(catalog, "colnames", None) or catalog.dtype.names
    # copies: the responses need the unweighted shape after e is rewritten
    e = {i: np.array(catalog[f"{sn}e{i}"], dtype=np.float64) for i in (1, 2)}
    for j, dw in ((1, dw1), (2, dw2)):
        for i in (1, 2):
            col = f"{sn}de{i}_dg{j}"
            if col in names:
                de = np.asarray(catalog[col], dtype=np.float64)
                catalog[col] = w * de + dw * e[i]
    for i in (1, 2):
        catalog[f"{sn}e{i}"] = w * e[i]
    return w


# ---------------------------------------------------------------------
# Command-line plumbing shared by the bin/ summary scripts.
# ---------------------------------------------------------------------
def add_shape_weight_arguments(parser) -> None:
    """Add ``--use-optimal-shape``, ``--shape-weight-params`` and
    ``--shape-weight-s2n`` to an :class:`argparse.ArgumentParser`."""
    import argparse

    parser.add_argument(
        "--use-optimal-shape", action=argparse.BooleanOptionalAction,
        default=False,
        help="Weight each shape by the optimal weight w_opt = f1(S/N) "
             "f2(|e|^2) on top of wsel, with the full shear response "
             "(xlens.utils.catalog.shape_weight). The cuts still act on "
             "the unweighted shape. Partials go to a separate *_wopt/ "
             "directory so they never mix with unweighted ones.",
    )
    parser.add_argument(
        "--shape-weight-params", type=float,
        nargs=len(SHAPE_WEIGHT_PARAM_NAMES),
        default=list(DEFAULT_SHAPE_WEIGHT_PARAMS),
        metavar=tuple(n.upper() for n in SHAPE_WEIGHT_PARAM_NAMES),
        help="Optimal-weight parameters [amp, alpha, snr0, beta, "
             "esq_scale]: f1(s) = amp (s/50)^beta / (1 + (snr0/s)^alpha), "
             "f2(v) = exp(-v/esq_scale) with v = |e|^2 (no cut on |e|). "
             "Default: the MergePipe test-fixture values "
             f"{list(DEFAULT_SHAPE_WEIGHT_PARAMS)}.",
    )
    parser.add_argument(
        "--shape-weight-s2n", type=str, default=DEFAULT_SHAPE_WEIGHT_S2N,
        help="S/N column f1 reads (its ds2n_dg responses are derived "
             "from the name); default the i-band forced FPFS flux S/N.",
    )


def shape_weight_params_from_args(args) -> list[float] | None:
    """The validated parameter list when ``--use-optimal-shape`` is on,
    else None; prints what is in use."""
    if not args.use_optimal_shape:
        return None
    params = list(parse_shape_weight_params(args.shape_weight_params))
    print(f"[Info] optimal shape weight on: params {params}, "
          f"S/N column {args.shape_weight_s2n}")
    return params
