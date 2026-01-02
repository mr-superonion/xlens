#!/usr/bin/env python3
"""Aggregate shear measurements over individual simulation seeds using RF cuts."""

import argparse
import glob
import os
import pickle
import warnings

import fitsio
import numpy as np
from astropy.stats import sigma_clipped_stats
from numpy.lib import recfunctions as rfn

try:
    from mpi4py import MPI  # type: ignore
except Exception as exc:  # pragma: no cover - mpi4py optional
    raise SystemExit("mpi4py is required to run this script") from exc

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


_MODEL_PATH = os.path.join(os.environ["HOME"], "unrecognized_blend_gri-only.pkl")
with open(_MODEL_PATH, "rb") as _fh:
    _RF_MODEL: RandomForestClassifier = pickle.load(_fh)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Measure + aggregate from catalogs over a given seed ID range "
            "using random-forest based score cuts."
        ),
        allow_abbrev=False,
    )
    p.add_argument("--summary", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--pscratch",
        type=str,
        default=os.environ.get("PSCRATCH", "."),
        help="Root directory where results were written.",
    )
    p.add_argument(
        "--layout",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help="Layout used in path naming.",
    )
    p.add_argument(
        "--target",
        type=str,
        default="g1",
        choices=["g1", "g2"],
        help="Which component to analyze (affects R and e used).",
    )
    p.add_argument(
        "--shear",
        type=float,
        default=0.02,
        help="True shear amplitude |g| used in sims.",
    )
    p.add_argument(
        "--min-id",
        type=int,
        required=True,
        help="Minimum sim_seed (inclusive).",
    )
    p.add_argument(
        "--max-id",
        type=int,
        required=True,
        help="Maximum sim_seed (exclusive).",
    )
    p.add_argument(
        "--score-maxes",
        type=str,
        default="0.1,0.2,0.3",
        help="Comma-separated list of score cuts, e.g. '0.1,0.2,0.3'.",
    )
    p.add_argument(
        "--flux-min",
        type=float,
        default=40.0,
        help="Flux cut applied to each band before score selection.",
    )
    p.add_argument(
        "--emax",
        type=float,
        default=0.3,
        help="Ellipticity magnitude cut upper bound.",
    )
    p.add_argument(
        "--dg",
        type=float,
        default=0.02,
        help="Finite-difference step for selection response.",
    )
    p.add_argument(
        "--stamp-dim",
        type=int,
        default=3900,
        help="Usable image dimension (pixels) for density/area calc.",
    )
    p.add_argument(
        "--pixel-scale",
        type=float,
        default=0.2,
        help="Pixel scale (arcsec/pixel).",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="# bootstrap resamples for m uncertainty (done on rank 0).",
    )
    args, unknown_args = p.parse_known_args()
    if unknown_args:
        print("[warn] Ignoring unknown args:", unknown_args)
    return args


def parse_score_list(s: str):
    return [float(x) for x in s.split(",")] if s else [0.1, 0.2, 0.3]


CORE_COLUMNS = [
    "wsel",
    "dwsel_dg1",
    "dwsel_dg2",
    "fpfs_e1",
    "fpfs_de1_dg1",
    "fpfs_de1_dg2",
    "fpfs_e2",
    "fpfs_de2_dg1",
    "fpfs_de2_dg2",
]


def base_path(pscratch, layout, target, shear):
    sd = f"shear{int(shear*100):02d}"
    return os.path.join(pscratch, f"constant_shear_{layout}", target, sd)


def _to_native(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.byteorder in ("=", "|"):
        return arr
    return arr.byteswap().newbyteorder("=")


def cat_read(base_dir: str, sim_id: int, mode: int) -> np.ndarray:
    mode_dir = os.path.join(base_dir, f"mode{mode}")
    main_path = os.path.join(mode_dir, f"cat-{sim_id:05d}.fits")
    arrays = [_to_native(fitsio.read(main_path, columns=CORE_COLUMNS))]

    for band in "gri":
        cols = [
            f"{band}_flux_gauss2",
            f"{band}_flux_gauss2_err",
            f"{band}_dflux_gauss2_dg1",
            f"{band}_dflux_gauss2_dg2",
        ]
        band_path = os.path.join(mode_dir, f"cat-{sim_id:05d}-{band}.fits")
        arrays.append(_to_native(fitsio.read(band_path, columns=cols)))

    merged = rfn.merge_arrays(arrays, flatten=True, asrecarray=False, usemask=False)
    return rfn.repack_fields(merged)


def _get_score(src, comp: int = 1, dg: float = 0.0):
    mag_zero = 30.0
    rr = -2.5 / np.log(10)
    phot = []
    for band in "gri":
        flux = np.clip(src[f"{band}_flux_gauss2"], a_min=1e-30, a_max=None)
        dflux = src[f"{band}_dflux_gauss2_dg{comp}"]
        dm = (rr * dflux / flux) * dg
        phot.append(mag_zero - 2.5 * np.log10(flux) + dm)
    phot = np.vstack(phot).T
    return _RF_MODEL.predict_proba(phot)[:, 1]


def _get_esq(src, comp: int = 1, dg: float = 0.0):
    e = src[f"fpfs_e{comp}"]
    de = src[f"fpfs_de{comp}_dg{comp}"]
    comp2 = int(3 - comp)
    e2 = src[f"fpfs_e{comp2}"]
    de2 = src[f"fpfs_de{comp2}_dg{comp}"]
    esq0 = e ** 2.0 + e2 ** 2.0
    return esq0 + 2.0 * dg * (e * de + e2 * de2)


def measure_shear_with_cut(src, flux_min, emax=0.3, smax=1.0, dg=0.02):
    score = _get_score(src)
    esq0 = _get_esq(src)
    m_0 = (
        (src["g_flux_gauss2"] > flux_min)
        & (src["r_flux_gauss2"] > flux_min)
        & (src["i_flux_gauss2"] > flux_min)
        & (esq0 < emax * emax)
        & (score < smax)
    )
    nn = int(np.sum(m_0))
    if nn == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    w0 = src["wsel"][m_0]
    e1 = np.sum(w0 * src["fpfs_e1"][m_0])
    e2 = np.sum(w0 * src["fpfs_e2"][m_0])

    r1 = np.sum(
        src["dwsel_dg1"][m_0] * src["fpfs_e1"][m_0]
        + w0 * src["fpfs_de1_dg1"][m_0]
    )
    r2 = np.sum(
        src["dwsel_dg2"][m_0] * src["fpfs_e2"][m_0]
        + w0 * src["fpfs_de2_dg2"][m_0]
    )

    def sel_term(comp: int):
        esq_p = _get_esq(src, comp=comp, dg=dg)
        score_p = _get_score(src, comp=comp, dg=dg)
        gflux_p = src["g_flux_gauss2"] + dg * src[f"g_dflux_gauss2_dg{comp}"]
        rflux_p = src["r_flux_gauss2"] + dg * src[f"r_dflux_gauss2_dg{comp}"]
        iflux_p = src["i_flux_gauss2"] + dg * src[f"i_dflux_gauss2_dg{comp}"]
        m_p = (
            (gflux_p > flux_min)
            & (rflux_p > flux_min)
            & (iflux_p > flux_min)
            & (esq_p < emax * emax)
            & (score_p < smax)
        )
        ellp = np.sum(src["wsel"][m_p] * src[f"fpfs_e{comp}"][m_p])

        esq_m = _get_esq(src, comp=comp, dg=-dg)
        score_m = _get_score(src, comp=comp, dg=-dg)
        gflux_m = src["g_flux_gauss2"] - dg * src[f"g_dflux_gauss2_dg{comp}"]
        rflux_m = src["r_flux_gauss2"] - dg * src[f"r_dflux_gauss2_dg{comp}"]
        iflux_m = src["i_flux_gauss2"] - dg * src[f"i_dflux_gauss2_dg{comp}"]
        m_m = (
            (gflux_m > flux_min)
            & (rflux_m > flux_min)
            & (iflux_m > flux_min)
            & (esq_m < emax * emax)
            & (score_m < smax)
        )
        ellm = np.sum(src["wsel"][m_m] * src[f"fpfs_e{comp}"][m_m])
        return (ellp - ellm) / (2.0 * dg)

    r1_sel = sel_term(1)
    r2_sel = sel_term(2)
    return e1, (r1 + r1_sel), e2, (r2 + r2_sel), nn


def per_rank_work(ids_chunk, base_dir, score_list, flux_min, emax, dg, target):
    ncut = len(score_list)
    E_pos = []
    E_neg = []
    R_pos = []
    R_neg = []

    for sid in ids_chunk:
        pos_path = os.path.join(base_dir, "mode40", f"cat-{sid:05d}.fits")
        neg_path = os.path.join(base_dir, "mode0", f"cat-{sid:05d}.fits")
        if not (os.path.exists(pos_path) and os.path.exists(neg_path)):
            continue
        try:
            src_pos = cat_read(base_dir, sid, mode=40)
            src_neg = cat_read(base_dir, sid, mode=0)
        except (OSError, FileNotFoundError):
            print(pos_path)
            print(neg_path)
            continue

        e_pos_row = np.zeros(ncut)
        e_neg_row = np.zeros(ncut)
        R_pos_row = np.zeros(ncut)
        R_neg_row = np.zeros(ncut)

        for j, smax in enumerate(score_list):
            e1p, R1p, e2p, R2p, _ = measure_shear_with_cut(
                src_pos, flux_min, emax=emax, smax=smax, dg=dg
            )
            e1m, R1m, e2m, R2m, _ = measure_shear_with_cut(
                src_neg, flux_min, emax=emax, smax=smax, dg=dg
            )
            if target == "g1":
                e_pos_row[j] = e1p
                e_neg_row[j] = e1m
                R_pos_row[j] = R1p
                R_neg_row[j] = R1m
            else:
                e_pos_row[j] = e2p
                e_neg_row[j] = e2m
                R_pos_row[j] = R2p
                R_neg_row[j] = R2m

        E_pos.append(e_pos_row)
        E_neg.append(e_neg_row)
        R_pos.append(R_pos_row)
        R_neg.append(R_neg_row)

    if len(E_pos) == 0:
        return (np.zeros((0, ncut)),) * 4

    return (
        np.vstack(E_pos),
        np.vstack(E_neg),
        np.vstack(R_pos),
        np.vstack(R_neg),
    )


def save_rank_partial(outdir, seed_index, E_pos, E_neg, R_pos, R_neg, ncut):
    partdir = os.path.join(outdir, "summary-rf-40-00")
    os.makedirs(partdir, exist_ok=True)
    path = os.path.join(partdir, f"seed_{seed_index:05d}.npz")
    np.savez_compressed(
        path,
        E_pos=E_pos,
        E_neg=E_neg,
        R_pos=R_pos,
        R_neg=R_neg,
        ncut=np.int64(ncut),
    )
    return path


def load_and_stack_all(outdir, ncut_expected=None):
    partdir = os.path.join(outdir, "summary-rf-40-00")
    arrays_E_pos, arrays_E_neg, arrays_R_pos, arrays_R_neg = [], [], [], []
    ncut_from_file = None

    for path in sorted(glob.glob(os.path.join(partdir, "*.npz"))):
        with np.load(path) as data:
            arrays_E_pos.append(data["E_pos"])
            arrays_E_neg.append(data["E_neg"])
            arrays_R_pos.append(data["R_pos"])
            arrays_R_neg.append(data["R_neg"])
            if ncut_from_file is None:
                ncut_from_file = int(data["ncut"])

    def _stack(blocks, ncut):
        blocks = [blk for blk in blocks if blk.size > 0]
        if not blocks:
            return np.zeros((0, ncut), dtype=np.float64)
        return np.vstack(blocks)

    ncut = ncut_expected if ncut_expected is not None else (ncut_from_file or 0)
    E_pos_all = _stack(arrays_E_pos, ncut)
    E_neg_all = _stack(arrays_E_neg, ncut)
    R_pos_all = _stack(arrays_R_pos, ncut)
    R_neg_all = _stack(arrays_R_neg, ncut)
    return E_pos_all, E_neg_all, R_pos_all, R_neg_all


def bootstrap_m(rng, e_pos, e_neg, R_pos, R_neg, shear_value, nsamp=10000):
    N, ncut = e_pos.shape
    ms = np.zeros((nsamp, ncut))
    cs = np.zeros((nsamp, ncut))
    for i in range(nsamp):
        k = rng.integers(0, N, size=N, endpoint=False)
        den = np.sum(R_pos[k] + R_neg[k], axis=0)

        num_m = np.sum(e_pos[k] - e_neg[k], axis=0)
        new_gamma = num_m / den
        ms[i] = new_gamma / shear_value - 1.0

        num_c = np.sum(e_pos[k] + e_neg[k], axis=0)
        cs[i] = num_c / den
    return ms, cs


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.max_id <= args.min_id:
        raise SystemExit("--max-id must be > --min-id")

    score_list = parse_score_list(args.score_maxes)
    outdir = base_path(args.pscratch, args.layout, args.target, args.shear)
    ncut = len(score_list)

    if not args.summary:
        all_ids = np.arange(args.min_id, args.max_id, dtype=int)
        n = len(all_ids)
        base = n // size
        rem = n % size
        start = rank * base + min(rank, rem)
        stop = start + base + (1 if rank < rem else 0)
        my_ids = all_ids[start:stop]

        E_pos, E_neg, R_pos, R_neg = per_rank_work(
            my_ids,
            outdir,
            score_list,
            args.flux_min,
            args.emax,
            args.dg,
            args.target,
        )

        index = (
            int(my_ids[0]) if len(my_ids) > 0 else (args.min_id + rank)
        )
        save_rank_partial(outdir, index, E_pos, E_neg, R_pos, R_neg, ncut)
        comm.Barrier()
        return

    if rank == 0:
        all_E_pos, all_E_neg, all_R_pos, all_R_neg = load_and_stack_all(
            outdir, ncut_expected=ncut
        )

        if all_E_pos.size == 0 or all_E_neg.size == 0:
            raise SystemExit(
                "No valid (+g/-g) pairs found in the given seed ID range."
            )

        num = np.sum(all_E_pos - all_E_neg, axis=0)
        den = np.sum(all_R_pos + all_R_neg, axis=0)
        m = (num / den) / args.shear - 1.0

        c = np.sum(all_E_pos + all_E_neg, axis=0) / np.sum(
            all_R_pos + all_R_neg, axis=0
        )

        area_arcmin2 = (args.stamp_dim * args.stamp_dim) * (
            args.pixel_scale / 60.0
        ) ** 2.0

        _, _, clipped_std = sigma_clipped_stats(
            all_E_pos / np.average(all_R_pos, axis=0),
            sigma=5.0,
        )
        neff = (0.26 / clipped_std) ** 2.0 / area_arcmin2

        rng = np.random.default_rng(0)
        ms, cs = bootstrap_m(
            rng,
            all_E_pos,
            all_E_neg,
            all_R_pos,
            all_R_neg,
            args.shear,
            nsamp=args.bootstrap,
        )
        ord_ms = np.sort(ms, axis=0)
        lo_idx = int(0.1587 * args.bootstrap)
        hi_idx = int(0.8413 * args.bootstrap)
        sigma_m = (ord_ms[hi_idx] - ord_ms[lo_idx]) / 2.0

        ord_cs = np.sort(cs, axis=0)
        sigma_c = (ord_cs[hi_idx] - ord_cs[lo_idx]) / 2.0

        print("==============================================")
        print(f"Outdir: {outdir}")
        print(f"Paired IDs (found): {all_E_pos.shape[0]}")
        print(f"ID range requested: [{args.min_id}, {args.max_id})")
        print(f"Score cuts: {score_list}")
        print(f"Area (arcmin^2): {area_arcmin2:.3f}")
        print("m (per score cut):", m)
        print("c (per score cut):", c)
        print("n_eff (per score cut):", neff)
        print("m 1-sigma (bootstrap):", sigma_m)
        print("c 1-sigma (bootstrap):", sigma_c)
        print("==============================================")
    comm.Barrier()


if __name__ == "__main__":
    main()
