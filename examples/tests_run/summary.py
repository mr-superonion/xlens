#!/usr/bin/env python3
"""
Example
-------
# 128 ranks spanning group IDs 0..299:
mpirun -n 128 python summary.py \
    --emax 0.3 --layout grid \
    --target g1 --shear 0.02 \
    --group-start 0 --group-end 300
"""

import argparse
import glob
import os

import numpy as np
from astropy.stats import sigma_clipped_stats

import pyarrow as pa
import pyarrow.parquet as pq


# --- Optional MPI: works without mpirun/srun or even mpi4py installed ---
try:
    from mpi4py import MPI  # type: ignore

    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()

    def _barrier():
        _COMM.Barrier()

except Exception:

    class _FakeComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

    _COMM = _FakeComm()
    _RANK = 0
    _SIZE = 1

    def _barrier():
        return


def parse_args():
    p = argparse.ArgumentParser(
        description="measure + aggregate from catalogs over a given group range.",
        allow_abbrev=False,
    )
    # Directory layout and naming
    p.add_argument(
        "--summary", action=argparse.BooleanOptionalAction, default=False
    )
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
    # group range (each group contains 100 sim_seeds)
    p.add_argument(
        "--group-start",
        type=int,
        required=True,
        help="Minimum group_id (inclusive), maps to sim_seed >= 100*group_id",
    )
    p.add_argument(
        "--group-end",
        type=int,
        required=True,
        help="Maximum group_id (exclusive), maps to sim_seed < 100*group_id",
    )
    # Measurement config
    p.add_argument(
        "--flux-mins",
        type=str,
        default="20,40,60",
        help="Comma-separated list of flux cuts, e.g. '20,40,60'.",
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
    # Geometry for density / area
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
    # Bootstrap
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


def parse_flux_list(s: str):
    return [float(x) for x in s.split(",")] if s else [20.0, 40.0, 60.0]


def base_path(pscratch, layout, target, shear):
    sd = f"shear{int(shear*100):02d}"
    return os.path.join(
        pscratch,
        "parquet",
        f"constant_shear_{layout}",
        target,
        sd,
    )


def parquet_group_path(base_dir, group_id, mode) -> str:
    return os.path.join(
        base_dir,
        f"mode{mode}",
        f"group_id={group_id}",
        "data.parquet",
    )


def arrow_to_numpy_struct(table: pa.Table) -> np.ndarray:
    table = table.combine_chunks()
    cols = [table[c].to_numpy(zero_copy_only=False) for c in table.column_names]
    return np.rec.fromarrays(cols, names=table.column_names)


def measure_shear_with_cut(src, flux_min, emax=0.3, dg=0.02):
    """
    Selection + response including selection response via finite differencing.

    Returns: e1, R11, e2, R22, N  (scalars for this flux_min)
    """
    esq0 = src["fpfs_e1"] ** 2 + src["fpfs_e2"] ** 2
    m0 = (src["g_flux_gauss2"] > flux_min) & (esq0 < emax * emax)
    nn = int(np.sum(m0))
    if nn == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    w0 = src["wsel"][m0]
    e1 = np.sum(w0 * src["fpfs_e1"][m0])
    e2 = np.sum(w0 * src["fpfs_e2"][m0])

    r1 = np.sum(
        src["dwsel_dg1"][m0] * src["fpfs_e1"][m0]
        + w0 * src["fpfs_de1_dg1"][m0]
    )
    r2 = np.sum(
        src["dwsel_dg2"][m0] * src["fpfs_e2"][m0]
        + w0 * src["fpfs_de2_dg2"][m0]
    )

    def sel_term(comp: int):
        e = src[f"fpfs_e{comp}"]
        de = src[f"fpfs_de{comp}_dg{comp}"]
        df = src[f"g_dflux_gauss2_dg{comp}"]
        comp2 = int(3 - comp)
        e2 = src[f"fpfs_e{comp2}"]
        de2 = src[f"fpfs_de{comp2}_dg{comp}"]

        esq_p = esq0 + 2.0 * dg * (e * de + e2 * de2)
        m_p = ((src["g_flux_gauss2"] + dg * df) > flux_min) & (esq_p < emax * emax)
        ellp = np.sum(src["wsel"][m_p] * e[m_p])

        esq_m = esq0 - 2.0 * dg * (e * de + e2 * de2)
        m_m = ((src["g_flux_gauss2"] - dg * df) > flux_min) & (esq_m < emax * emax)
        ellm = np.sum(src["wsel"][m_m] * e[m_m])
        return (ellp - ellm) / (2.0 * dg)

    r1_sel = sel_term(1)
    r2_sel = sel_term(2)
    return e1, (r1 + r1_sel), e2, (r2 + r2_sel), nn


def per_rank_work(group_chunk, input_dir, flux_list, emax, dg, target):
    """
    For each group_id in ``group_chunk``, read the +g (mode40) and -g (mode0)
    catalogs aggregated over that group's 100 sim_seeds and compute the
    per-flux-cut e_pos/e_neg, R_pos/R_neg. Returns 4 arrays of shape
    (Nsamples_local, ncut).
    """
    ncut = len(flux_list)
    E_pos = []
    E_neg = []
    R_pos = []
    R_neg = []

    for group_id in group_chunk:
        ppos = parquet_group_path(input_dir, group_id, mode=40)  # +g
        pneg = parquet_group_path(input_dir, group_id, mode=0)  # -g

        if not (os.path.isfile(ppos) and os.path.isfile(pneg)):
            # Skip if pair not complete
            continue
        try:
            tbl_pos = pq.read_table(ppos)
            tbl_neg = pq.read_table(pneg)
            src_pos = arrow_to_numpy_struct(tbl_pos)
            src_neg = arrow_to_numpy_struct(tbl_neg)
        except OSError:
            print(ppos)
            print(pneg)
            continue

        e_pos_row = np.zeros(ncut)
        e_neg_row = np.zeros(ncut)
        R_pos_row = np.zeros(ncut)
        R_neg_row = np.zeros(ncut)

        for j, fmin in enumerate(flux_list):
            e1p, R1p, e2p, R2p, _ = measure_shear_with_cut(
                src_pos, fmin, emax=emax, dg=dg
            )
            e1m, R1m, e2m, R2m, _ = measure_shear_with_cut(
                src_neg, fmin, emax=emax, dg=dg
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
        z = (np.zeros((0, ncut)),) * 4
        return z

    return (
        np.vstack(E_pos),
        np.vstack(E_neg),
        np.vstack(R_pos),
        np.vstack(R_neg),
    )


def save_rank_partial(outdir, group_index, E_pos, E_neg, R_pos, R_neg, ncut):
    partdir = os.path.join(outdir, "summary-40-00")
    os.makedirs(partdir, exist_ok=True)
    path = os.path.join(partdir, f"group_{group_index:05d}.npz")
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
    partdir = os.path.join(outdir, "summary-40-00")
    arrays_E_pos, arrays_E_neg, arrays_R_pos, arrays_R_neg = [], [], [], []
    ncut_from_file = None

    paths = glob.glob(os.path.join(partdir, "group_*.npz"))
    for path in paths:
        with np.load(path) as data:
            E_pos = data["E_pos"]
            E_neg = data["E_neg"]
            R_pos = data["R_pos"]
            R_neg = data["R_neg"]
            if ncut_from_file is None:
                ncut_from_file = int(data["ncut"])
            arrays_E_pos.append(E_pos)
            arrays_E_neg.append(E_neg)
            arrays_R_pos.append(R_pos)
            arrays_R_neg.append(R_neg)

    def _stack(blocks, ncut):
        blocks = [b for b in blocks if b.size > 0]
        if len(blocks) == 0:
            return np.zeros((0, ncut), dtype=np.float64)
        return np.vstack(blocks)

    ncut = ncut_expected if ncut_expected is not None else (ncut_from_file or 0)
    E_pos_all = _stack(arrays_E_pos, ncut)
    E_neg_all = _stack(arrays_E_neg, ncut)
    R_pos_all = _stack(arrays_R_pos, ncut)
    R_neg_all = _stack(arrays_R_neg, ncut)
    return E_pos_all, E_neg_all, R_pos_all, R_neg_all


def bootstrap_m(
    rng, e_pos, e_neg, R_pos, R_neg, shear_value, nsamp=10000
):  # noqa: N802 - keep historical name
    """Bootstrap estimates of the multiplicative and additive biases.

    Parameters
    ----------
    rng : numpy.random.Generator
        RNG used to draw bootstrap indices.
    e_pos, e_neg, R_pos, R_neg : ndarray
        Arrays with shape ``(Nsamples_total, ncut)`` containing the per-object
        ellipticity and response measurements for positive/negative shear.
    shear_value : float
        True shear amplitude used in the simulations.
    nsamp : int, optional
        Number of bootstrap realizations to draw.

    Returns
    -------
    tuple of ndarray
        ``(ms, cs)`` each of shape ``(nsamp, ncut)`` containing the bootstrap
        draws of the multiplicative and additive biases respectively.
    """
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
    rank = _RANK
    size = _SIZE

    flux_list = parse_flux_list(args.flux_mins)
    ncut = len(flux_list)
    outdir = base_path(args.pscratch, args.layout, args.target, args.shear)
    input_dir = outdir

    if not args.summary:
        if args.group_end <= args.group_start:
            raise SystemExit("--group-end must be > --group-start")
        all_groups = np.arange(args.group_start, args.group_end, dtype=int)

        n = len(all_groups)
        base = n // size
        rem = n % size
        start = rank * base + min(rank, rem)
        stop = start + base + (1 if rank < rem else 0)
        my_groups = all_groups[start:stop]

        E_pos, E_neg, R_pos, R_neg = per_rank_work(
            my_groups,
            input_dir,
            flux_list,
            args.emax,
            args.dg,
            args.target,
        )

        index = (
            int(my_groups[0])
            if len(my_groups) > 0
            else (args.group_start + rank)
        )
        save_rank_partial(outdir, index, E_pos, E_neg, R_pos, R_neg, ncut)
        _barrier()
    else:
        if rank == 0:
            all_E_pos, all_E_neg, all_R_pos, all_R_neg = load_and_stack_all(
                outdir, ncut_expected=ncut
            )
            if all_E_pos.size == 0 or all_E_neg.size == 0:
                raise SystemExit(
                    "No valid (+g/-g) pairs found in the given group range."
                )

            num = np.sum(all_E_pos - all_E_neg, axis=0)  # (ncut,)
            den = np.sum(all_R_pos + all_R_neg, axis=0)
            m = (num / den) / args.shear - 1.0

            c = np.sum(all_E_pos + all_E_neg, axis=0) / np.sum(
                all_R_pos + all_R_neg, axis=0
            )

            area_arcmin2 = (args.stamp_dim * args.stamp_dim) * (
                args.pixel_scale / 60.0
            ) ** 2.0

            clipped_mean, clipped_median, clipped_std = sigma_clipped_stats(
                all_E_pos / np.average(all_R_pos, axis=0),
                sigma=5.0,
                axis=0,
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
            print(f"Input Directory: {input_dir}")
            print(f"Paired IDs (found): {all_E_pos.shape[0]}")
            print(
                "Group range requested: "
                f"[{args.group_start}, {args.group_end})"
            )
            print(
                "Seed range requested: "
                f"[{args.group_start * 100}, {args.group_end * 100})"
            )
            print(f"Flux cuts: {flux_list}")
            print(f"Area (arcmin^2): {area_arcmin2:.3f}")
            print("m (per flux cut):", m)
            print("c (per flux cut):", c)
            print("n_eff (per flux cut):", neff)
            print("m 1-sigma (bootstrap):", sigma_m)
            print("c 1-sigma (bootstrap):", sigma_c)
            print("==============================================")


if __name__ == "__main__":
    main()
