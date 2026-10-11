#!/usr/bin/env python3
"""Aggregate shear measurements using either FlexZBoost or BPZ redshifts."""

from __future__ import annotations

import argparse
import gc
import glob
import os
from typing import Iterable, List, Optional, Tuple

import common
import fitsio
import numpy as np

from xlens.utils.catalog import measure_shear as _measure_shear_binned
from xlens.utils.constants import MAG_ZERO_AB


def _shear_root(layout, target, shear):
    # Both process_mode0/ and process_mode40/ share this parent.
    return os.path.dirname(common.process_outdir(layout, target, shear, 40))


colnames = [
    "wsel",
    "dwsel_dg1",
    "dwsel_dg2",
    "fpfs_e1",
    "fpfs_de1_dg1",
    "fpfs_de1_dg2",
    "fpfs_e2",
    "fpfs_de2_dg1",
    "fpfs_de2_dg2",
    "fpfs_m0",
    "fpfs_dm0_dg1",
    "fpfs_dm0_dg2",
    "fpfs_m2",
    "fpfs_dm2_dg1",
    "fpfs_dm2_dg2",
    "lsst_i_flux_fpfs1",
    "lsst_i_dflux_fpfs1_dg1",
    "lsst_i_dflux_fpfs1_dg2",
    "lsst_i_fpfs1_e1",
    "lsst_i_fpfs1_de1_dg1",
    "lsst_i_fpfs1_e2",
    "lsst_i_fpfs1_de2_dg2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure + aggregate from catalogs over a given seed ID range "
            "using either FlexZBoost or BPZ redshift slices."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--summary", action=argparse.BooleanOptionalAction, default=False
    )
    # Directory layout and naming
    parser.add_argument(
        "--layout",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help="Layout used in path naming.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="g1",
        choices=["g1", "g2"],
        help="Which component to analyze (affects R and e used).",
    )
    parser.add_argument(
        "--shear",
        type=float,
        default=0.02,
        help="True shear amplitude |g| used in sims.",
    )
    # ID range
    parser.add_argument(
        "--min-id",
        type=int,
        required=True,
        help="Minimum sim_seed (inclusive).",
    )
    parser.add_argument(
        "--max-id",
        type=int,
        required=True,
        help="Maximum sim_seed (exclusive).",
    )
    parser.add_argument(
        "--mag-max",
        type=float,
        default=25.0,
        help="Flux cut applied to each band before selection.",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=0.5,
        help="Ellipticity magnitude cut upper bound.",
    )
    parser.add_argument(
        "--trace-min",
        type=float,
        default=None,
        help="Size cut: keep sources with trace = fpfs_m2 / fpfs_m0 = "
             "(m00 + m20) / m00 above this. Left out, NO size cut is applied "
             "at all, which is what the original partials were made with -- "
             "note 0.0 is not the same thing, since trace > 0 already drops "
             "sources with a non-positive monopole. "
             "xlens.utils.catalog.base.measure_shear uses 0.05.",
    )
    parser.add_argument(
        "--dg",
        type=float,
        default=0.02,
        help="Finite-difference step for selection response.",
    )
    # Geometry for density / area
    parser.add_argument(
        "--stamp-dim",
        type=int,
        default=1350,
        help="Usable image dimension (pixels) for density/area calc.",
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=0.2,
        help="Pixel scale (arcsec/pixel).",
    )
    # Bootstrap
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="# bootstrap resamples for m uncertainty",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Optional integer tag; reads catalogs from "
             "process_mode<N>-v<version>/ and writes partials to "
             "summary-flux-40-00-v<version>/.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional configuration name; reads catalogs from "
             "process_mode<N>_<tag>/ and writes partials to "
             "summary-flux-40-00_<tag>/. Combines with --version.",
    )
    parser.add_argument(
        "--summary-tag",
        type=str,
        default=None,
        help="Optional name for the CUTS, appended to the partials "
             "directory only (not to the catalog directory --tag selects), "
             "so several cut choices over the same catalogs can coexist: "
             "summary-flux-40-00[_<tag>]_<summary-tag>/.",
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("[warn] Ignoring unknown args:", unknown_args)
    return args


def cat_read(layout, target, shear, mode, sim_id, version=None, tag=None):
    outdir = common.process_outdir(layout, target, shear, mode,
                                   version=version, tag=tag)
    return fitsio.read(common.cat_path(outdir, sim_id), columns=colnames)


def measure_shear(src, flux_min=0.0, emax=0.3, trace_min=None, dg=0.02,
                  target="g1"):
    """Selection + response, including the selection response.

    A thin wrapper over ``xlens.utils.catalog.measure_shear`` with no
    tomographic binning (``z_estimator=None``): the cuts, their
    finite-difference selection responses and the estimator all come from
    the library, so this script and the redshift-binned analysis cannot
    drift apart.  What makes it this analysis is

      * ``flux_name="fpfs1"`` with ``bands=("lsst_i",)`` -- the flux cut is
        on the i-band forced FPFS flux alone, not on every band
      * ``trace_min=None`` by default -- no size cut, the setting the
        reference partials were made with (0.0 would already drop sources
        whose monopole is not positive)

    Returns a dict of 1-element arrays: e, r, r_sel, n_gal.
    """
    return _measure_shear_binned(
        src=src,
        z_estimator=None,
        zbounds=None,
        flux_min=flux_min,
        emax=emax,
        trace_min=trace_min,
        dg=dg,
        target=target,
        mag_zero=MAG_ZERO_AB,
        flux_name="fpfs1",
        bands=("lsst_i",),
        ref_band="lsst_i",
    )


def per_rank_work(
    ids_chunk: Iterable[int],
    layout: str,
    target: str,
    shear: float,
    flux_min: float,
    emax: float,
    dg: float,
    trace_min: Optional[float] = None,
    version: Optional[int] = None,
    tag: Optional[str] = None,
):
    e_pos_rows = []
    e_neg_rows = []
    r_pos_rows = []
    r_neg_rows = []
    n_pos_rows = []
    n_neg_rows = []
    for sim_id in ids_chunk:
        src_pos = cat_read(
            layout, target, shear, mode=40, sim_id=sim_id, version=version,
            tag=tag,
        )
        out_pos = measure_shear(
            src=src_pos,
            flux_min=flux_min,
            emax=emax,
            trace_min=trace_min,
            dg=dg,
            target=target,
        )
        del src_pos
        gc.collect()
        src_neg = cat_read(
            layout, target, shear, mode=0, sim_id=sim_id, version=version,
            tag=tag,
        )
        out_neg = measure_shear(
            src=src_neg,
            flux_min=flux_min,
            emax=emax,
            trace_min=trace_min,
            dg=dg,
            target=target,
        )
        del src_neg
        gc.collect()
        e_pos_rows.append(out_pos["e"])
        e_neg_rows.append(out_neg["e"])
        r_pos_rows.append(out_pos["r"] + out_pos["r_sel"])
        r_neg_rows.append(out_neg["r"] + out_neg["r_sel"])
        n_pos_rows.append(out_pos["n_gal"])
        n_neg_rows.append(out_neg["n_gal"])

    return (
        np.vstack(e_pos_rows),
        np.vstack(e_neg_rows),
        np.vstack(r_pos_rows),
        np.vstack(r_neg_rows),
        np.vstack(n_pos_rows),
        np.vstack(n_neg_rows),
    )


def summary_directory(
    layout: str, target: str, shear: float, version: Optional[int] = None,
    tag: Optional[str] = None, summary_tag: Optional[str] = None,
) -> str:
    leaf = "summary-flux-40-00"
    if version is not None:
        leaf = f"{leaf}-v{int(version)}"
    if tag:
        leaf = f"{leaf}_{tag}"
    if summary_tag:
        leaf = f"{leaf}_{summary_tag}"
    return os.path.join(_shear_root(layout, target, shear), leaf)


def save_rank_partial(
    layout: str,
    target: str,
    shear: float,
    seed_index: int,
    e_pos: np.ndarray,
    e_neg: np.ndarray,
    r_pos: np.ndarray,
    r_neg: np.ndarray,
    n_pos: np.ndarray,
    n_neg: np.ndarray,
    version: Optional[int] = None,
    tag: Optional[str] = None,
    summary_tag: Optional[str] = None,
) -> str:
    partdir = summary_directory(layout, target, shear, version=version, tag=tag,
                                summary_tag=summary_tag)
    os.makedirs(partdir, exist_ok=True)
    path = os.path.join(partdir, f"seed_{seed_index:05d}.npz")
    np.savez_compressed(
        path,
        E_pos=e_pos,
        E_neg=e_neg,
        R_pos=r_pos,
        R_neg=r_neg,
        N_pos=n_pos,
        N_neg=n_neg,
        ncut=1,
    )
    return path


def load_and_stack_all(
    layout: str,
    target: str,
    shear: float,
    ncut_expected: Optional[int] = None,
    version: Optional[int] = None,
    tag: Optional[str] = None,
    summary_tag: Optional[str] = None,
):
    partdir = summary_directory(layout, target, shear, version=version, tag=tag,
                                summary_tag=summary_tag)
    arrays_E_pos: List[np.ndarray] = []
    arrays_E_neg: List[np.ndarray] = []
    arrays_R_pos: List[np.ndarray] = []
    arrays_R_neg: List[np.ndarray] = []
    arrays_N_pos: List[np.ndarray] = []
    arrays_N_neg: List[np.ndarray] = []
    ncut_from_file: Optional[int] = None

    for path in sorted(glob.glob(os.path.join(partdir, "*.npz"))):
        with np.load(path) as data:
            arrays_E_pos.append(data["E_pos"])
            arrays_E_neg.append(data["E_neg"])
            arrays_R_pos.append(data["R_pos"])
            arrays_R_neg.append(data["R_neg"])
            arrays_N_pos.append(data["N_pos"])
            arrays_N_neg.append(data["N_neg"])
            if ncut_from_file is None:
                ncut_from_file = int(data["ncut"])

    def _stack(blocks: List[np.ndarray], ncut: int) -> np.ndarray:
        valid = [blk for blk in blocks if blk.size > 0]
        if not valid:
            return np.zeros((0, ncut), dtype=np.float64)
        return np.vstack(valid)

    ncut = ncut_expected if ncut_expected is not None else (ncut_from_file or 0)
    E_pos_all = _stack(arrays_E_pos, ncut)
    E_neg_all = _stack(arrays_E_neg, ncut)
    R_pos_all = _stack(arrays_R_pos, ncut)
    R_neg_all = _stack(arrays_R_neg, ncut)
    N_pos_all = _stack(arrays_N_pos, ncut)
    N_neg_all = _stack(arrays_N_neg, ncut)
    return E_pos_all, E_neg_all, R_pos_all, R_neg_all, N_pos_all, N_neg_all


def bootstrap_mc(
    rng: np.random.Generator,
    e_pos: np.ndarray,
    e_neg: np.ndarray,
    r_pos: np.ndarray,
    r_neg: np.ndarray,
    shear_value: float,
    nsamp: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    n_obj, ncut = e_pos.shape
    ms = np.zeros((nsamp, ncut))
    cs = np.zeros((nsamp, ncut))
    for idx in range(nsamp):
        choices = rng.integers(0, n_obj, size=n_obj, endpoint=False)
        denom = np.sum(r_pos[choices] + r_neg[choices], axis=0)

        num_m = np.sum(e_pos[choices] - e_neg[choices], axis=0)
        gamma = num_m / denom
        ms[idx] = gamma / shear_value - 1.0

        num_c = np.sum(e_pos[choices] + e_neg[choices], axis=0)
        cs[idx] = num_c / denom
    return ms, cs


def bootstrap_one(
    rng: np.random.Generator,
    e_pos: np.ndarray,
    r_pos: np.ndarray,
    nsamp: int = 10000,
) -> np.ndarray:
    n_obj, ncut = e_pos.shape
    gout = np.zeros((nsamp, ncut))
    denom = np.sum(r_pos, axis=0)
    for idx in range(nsamp):
        choices = rng.integers(0, n_obj, size=n_obj, endpoint=False)
        num_m = np.sum(e_pos[choices], axis=0)
        gout[idx] = num_m / denom
    return gout


def main() -> None:
    args = parse_args()
    if args.max_id <= args.min_id:
        raise SystemExit("--max-id must be > --min-id")

    flux_min = 10.0 ** ((MAG_ZERO_AB - args.mag_max) / 2.5)
    if not args.summary:
        my_ids = np.arange(args.min_id, args.max_id, dtype=int)
        if len(my_ids) > 0:
            e_pos, e_neg, r_pos, r_neg, n_pos, n_neg = per_rank_work(
                my_ids,
                args.layout,
                args.target,
                args.shear,
                flux_min,
                args.emax,
                args.dg,
                trace_min=args.trace_min,
                version=args.version,
                tag=args.tag,
            )
            save_rank_partial(
                args.layout,
                args.target,
                args.shear,
                int(my_ids[0]),
                e_pos,
                e_neg,
                r_pos,
                r_neg,
                n_pos,
                n_neg,
                version=args.version,
                tag=args.tag,
                summary_tag=args.summary_tag,
            )
    else:
        all_e_pos, all_e_neg, all_r_pos, all_r_neg, all_n_pos, all_n_neg = \
            load_and_stack_all(
                args.layout, args.target, args.shear, ncut_expected=1,
                version=args.version, tag=args.tag,
                summary_tag=args.summary_tag,
            )

        if all_e_pos.size == 0 or all_e_neg.size == 0:
            raise SystemExit(
                "No valid (+g/-g) pairs found in the given seed ID range."
            )

        num = np.sum(all_e_pos - all_e_neg, axis=0)
        denom = np.sum(all_r_pos + all_r_neg, axis=0)
        m = (num / denom) / args.shear - 1.0

        c = np.sum(all_e_pos + all_e_neg, axis=0) / np.sum(
            all_r_pos + all_r_neg, axis=0
        )

        area_arcmin2 = (args.stamp_dim * args.stamp_dim) * (
            args.pixel_scale / 60.0
        ) ** 2.0
        nsample = all_e_pos.shape[0]
        area_all_arcmin2 = area_arcmin2 * nsample
        nraw = np.sum(all_n_pos) / area_all_arcmin2

        rng = np.random.default_rng(0)
        gs = bootstrap_one(
            rng,
            all_e_pos,
            all_r_pos,
            nsamp=args.bootstrap,
        )

        ord_gs = np.sort(gs, axis=0)
        lo_idx = int(0.1587 * args.bootstrap)
        hi_idx = int(0.8413 * args.bootstrap)
        sigma_g = (ord_gs[hi_idx] - ord_gs[lo_idx]) / 2.0
        neff = (0.26 / sigma_g) ** 2.0 / area_all_arcmin2

        ms, cs = bootstrap_mc(
            rng,
            all_e_pos,
            all_e_neg,
            all_r_pos,
            all_r_neg,
            args.shear,
            nsamp=args.bootstrap,
        )
        ord_ms = np.sort(ms, axis=0)
        sigma_m = (ord_ms[hi_idx] - ord_ms[lo_idx]) / 2.0

        ord_cs = np.sort(cs, axis=0)
        sigma_c = (ord_cs[hi_idx] - ord_cs[lo_idx]) / 2.0

        print("==============================================")
        print(
            "Catalog directory: "
            f"{_shear_root(args.layout, args.target, args.shear)}"
        )
        partdir = summary_directory(
            args.layout, args.target, args.shear, version=args.version,
            tag=args.tag, summary_tag=args.summary_tag,
        )
        print(f"Partials dir: {partdir}")
        print(f"Cuts: mag_max={args.mag_max} emax={args.emax} "
              f"trace_min={args.trace_min} dg={args.dg}")
        print(f"Paired IDs (found): {all_e_pos.shape[0]}")
        print(f"Area (arcmin^2): {area_arcmin2:.3f}")
        print("m (per redshift cut):", m)
        print("c (per redshift cut):", c)
        print("n_eff (per redshift cut):", neff)
        print("n_raw (per redshift cut):", nraw)
        print("m 1-sigma (bootstrap):", sigma_m)
        print("c 1-sigma (bootstrap):", sigma_c)
        print("==============================================")


if __name__ == "__main__":
    main()
