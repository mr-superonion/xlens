#!/usr/bin/env python3
"""Aggregate shear measurements using either FlexZBoost or BPZ redshifts."""

from __future__ import annotations

import argparse
import gc
import glob
import os
from typing import Iterable, List, Optional, Tuple

import fitsio
import numpy as np


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
    "flux_gauss2",
    "dflux_gauss2_dg1",
    "dflux_gauss2_dg2",
    "i_fpfs1_e1",
    "i_fpfs1_de1_dg1",
    "i_fpfs1_e2",
    "i_fpfs1_de2_dg2",
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
    parser.add_argument(
        "--correction", action=argparse.BooleanOptionalAction, default=True
    )
    # Directory layout and naming
    parser.add_argument(
        "--pscratch",
        type=str,
        default=os.environ.get("PSCRATCH", "."),
        help="Root directory where results were written.",
    )
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
        "--z-bounds",
        type=str,
        default="0.3,0.6,0.9,1.2,1.5,1.8",
        help="Comma-separated redshift boundarys, e.g. '0.3,0.6,0.9,1.2,1.8'.",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=0.3,
        help="Ellipticity magnitude cut upper bound.",
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
        default=3850,
        help="Usable image dimension (pixels) for density/area calc.",
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=0.2,
        help="Pixel scale (arcsec/pixel).",
    )
    parser.add_argument(
        "--mag-zero",
        type=float,
        default=30.0,
        help="magnitude zero point",
    )
    # Bootstrap
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="# bootstrap resamples for m uncertainty",
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("[warn] Ignoring unknown args:", unknown_args)
    return args


def parse_zbounds(values: str) -> list[float]:
    return [float(x) for x in values.split(",")]


def base_path(pscratch, layout, target, shear):
    sd = f"shear{int(shear*100):02d}"
    return os.path.join(
        pscratch,
        f"constant_shear_{layout}-2",
        target,
        sd,
    )


def cat_read(base_dir: str, sim_id: int, mode: int) -> np.ndarray:
    main_path = os.path.join(base_dir, f"mode{mode}", f"cat-{sim_id:05d}.fits")
    main = fitsio.read(main_path, columns=colnames)
    return main


def measure_shear(src, flux_min=0.0, emax=0.3, dg=0.02, target="g1"):
    """
    Selection + response including selection response via finite differencing.

    Returns: e1, R11, e2, R22, N  (scalars for this flux_min)
    """
    emax = 0.5
    esq0 = src["fpfs_e1"] ** 2 + src["fpfs_e2"] ** 2
    m0 = (src["flux_gauss2"] > flux_min) & (esq0 < emax * emax)
    w0 = src["wsel"][m0]
    ename = "i_fpfs1"
    e1 = np.sum(w0 * src[f"{ename}_e1"][m0])
    e2 = np.sum(w0 * src[f"{ename}_e2"][m0])

    r1 = np.sum(
        src["dwsel_dg1"][m0] * src[f"{ename}_e1"][m0]
        + w0 * src[f"{ename}_de1_dg1"][m0]
    )
    r2 = np.sum(
        src["dwsel_dg2"][m0] * src[f"{ename}_e2"][m0]
        + w0 * src[f"{ename}_de2_dg2"][m0]
    )

    def sel_term(comp: int):
        comp2 = int(3 - comp)
        e = src[f"fpfs_e{comp}"]
        de = src[f"fpfs_de{comp}_dg{comp}"]
        en = src[f"fpfs_e{comp2}"]
        den = src[f"fpfs_de{comp2}_dg{comp}"]
        df = src[f"dflux_gauss2_dg{comp}"]

        esq_p = esq0 + 2.0 * dg * (e * de + en * den)
        m_p = (
            ((src["flux_gauss2"] + dg * df) > flux_min)
            & (esq_p < emax * emax)
        )
        ellp = np.sum(src["wsel"][m_p] * src[f"{ename}_e{comp}"][m_p])

        esq_m = esq0 - 2.0 * dg * (e * de + en * den)
        m_m = (
            ((src["flux_gauss2"] - dg * df) > flux_min)
            & (esq_m < emax * emax)
        )
        ellm = np.sum(src["wsel"][m_m] * src[f"{ename}_e{comp}"][m_m])
        return (ellp - ellm) / (2.0 * dg)

    r1_sel = sel_term(1)
    r2_sel = sel_term(2)
    if target == "g1":
        return {
            "e": np.array([e1]),
            "r": np.array([r1]),
            "r_sel": np.array([r1_sel]),
        }
    else:
        return {
            "e": np.array([e2]),
            "r": np.array([r2]),
            "r_sel": np.array([r2_sel]),
        }


def per_rank_work(
    ids_chunk: Iterable[int],
    base_dir: str,
    zbounds: list[float],
    flux_min: float,
    emax: float,
    dg: float,
    target: str,
    do_correction: bool = True,
    mag_zero: float = 30.0,
):
    e_pos_rows = []
    e_neg_rows = []
    r_pos_rows = []
    r_neg_rows = []
    for sim_id in ids_chunk:
        src_pos = cat_read(base_dir, sim_id, mode=40)
        out_pos = measure_shear(
            src=src_pos,
            flux_min=flux_min,
            emax=emax,
            dg=dg,
            target=target,
        )
        del src_pos
        gc.collect()
        src_neg = cat_read(base_dir, sim_id, mode=0)
        out_neg = measure_shear(
            src=src_neg,
            flux_min=flux_min,
            emax=emax,
            dg=dg,
            target=target,
        )
        del src_neg
        gc.collect()
        e_pos_rows.append(out_pos["e"])
        e_neg_rows.append(out_neg["e"])
        r_pos_rows.append(out_pos["r"] + out_pos["r_sel"])
        r_neg_rows.append(out_neg["r"] + out_neg["r_sel"])

    return (
        np.vstack(e_pos_rows),
        np.vstack(e_neg_rows),
        np.vstack(r_pos_rows),
        np.vstack(r_neg_rows),
    )


def summary_directory(outdir: str, do_correction: bool) -> str:
    partdir = os.path.join(outdir, "summary-flux-40-00")
    if not do_correction:
        partdir = partdir + "-nc"
    return partdir


def save_rank_partial(
    outdir: str,
    seed_index: int,
    e_pos: np.ndarray,
    e_neg: np.ndarray,
    r_pos: np.ndarray,
    r_neg: np.ndarray,
    ncut: int,
    do_correction: bool = True,
) -> str:
    partdir = summary_directory(outdir, do_correction)
    os.makedirs(partdir, exist_ok=True)
    path = os.path.join(partdir, f"seed_{seed_index:05d}.npz")
    np.savez_compressed(
        path,
        E_pos=e_pos,
        E_neg=e_neg,
        R_pos=r_pos,
        R_neg=r_neg,
        ncut=int(ncut),
    )
    return path


def load_and_stack_all(
    outdir: str,
    ncut_expected: Optional[int] = None,
    do_correction: bool = True,
):
    partdir = summary_directory(outdir, do_correction)
    arrays_E_pos: List[np.ndarray] = []
    arrays_E_neg: List[np.ndarray] = []
    arrays_R_pos: List[np.ndarray] = []
    arrays_R_neg: List[np.ndarray] = []
    ncut_from_file: Optional[int] = None
    print(partdir)

    for path in sorted(glob.glob(os.path.join(partdir, "*.npz"))):
        with np.load(path) as data:
            arrays_E_pos.append(data["E_pos"])
            arrays_E_neg.append(data["E_neg"])
            arrays_R_pos.append(data["R_pos"])
            arrays_R_neg.append(data["R_neg"])
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
    return E_pos_all, E_neg_all, R_pos_all, R_neg_all


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
    do_correction = args.correction
    if args.max_id <= args.min_id:
        raise SystemExit("--max-id must be > --min-id")
    base_dir = base_path(args.pscratch, args.layout, args.target, args.shear)
    zbounds = parse_zbounds(args.z_bounds)
    ncut = len(zbounds) + 1

    flux_min = -10.0
    if not args.summary:
        my_ids = np.arange(args.min_id, args.max_id, dtype=int)
        if len(my_ids) > 0:
            e_pos, e_neg, r_pos, r_neg = per_rank_work(
                my_ids,
                base_dir,
                zbounds,
                flux_min,
                args.emax,
                args.dg,
                args.target,
                do_correction=do_correction,
                mag_zero=args.mag_zero,
            )
            save_rank_partial(
                base_dir,
                int(my_ids[0]),
                e_pos,
                e_neg,
                r_pos,
                r_neg,
                ncut,
                do_correction=do_correction,
            )
    else:
        all_e_pos, all_e_neg, all_r_pos, all_r_neg = load_and_stack_all(
            base_dir,
            ncut_expected=ncut,
            do_correction=do_correction,
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
        print(f"Catalog directory: {base_dir}")
        print("flux cut")
        print(f"Paired IDs (found): {all_e_pos.shape[0]}")
        print(f"Redshift boundarys: {list(zbounds)}")
        print(f"Area (arcmin^2): {area_arcmin2:.3f}")
        print("m (per redshift cut):", m)
        print("c (per redshift cut):", c)
        print("n_eff (per redshift cut):", neff)
        print("m 1-sigma (bootstrap):", sigma_m)
        print("c 1-sigma (bootstrap):", sigma_c)
        print("==============================================")


if __name__ == "__main__":
    main()
