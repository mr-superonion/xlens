#!/usr/bin/env python3
import os
import argparse
import numpy as np
import fitsio
from mpi4py import MPI
from astropy.stats import sigma_clipped_stats


def parse_args():
    p = argparse.ArgumentParser(
        description="MPI: measure + aggregate from cat-%05d-mode%d.fits over a given ID range."
    )
    # Directory layout and naming
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
    # ID range
    p.add_argument(
        "--min-id",
        type=int,
        required=True,
        help="Minimum sim_seed (inclusive), e.g. 0",
    )
    p.add_argument(
        "--max-id",
        type=int,
        required=True,
        help="Maximum sim_seed (inclusive), e.g. 2047",
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
    return p.parse_args()


def parse_flux_list(s: str):
    return [float(x) for x in s.split(",")] if s else [20.0, 40.0, 60.0]


def outdir_path(pscratch, layout, target, shear):
    sd = f"shear{int(shear*100):02d}"
    return os.path.join(pscratch, f"constant_shear_{layout}", target, sd)


def cat_path(outdir, sim_id, mode):
    # cat-%05d-mode%d.fits
    return os.path.join(outdir, f"cat-{sim_id:05d}-mode{mode}.fits")


def measure_shear_flux_cut(src, flux_min, emax=0.3, dg=0.02):
    """
    Selection + response including selection response via finite differencing.

    Returns: e1, R11, e2, R22, N  (scalars for this flux_min)
    """
    esq0 = src["fpfs_e1"] ** 2 + src["fpfs_e2"] ** 2
    m0 = (src["flux"] > flux_min) & (esq0 < emax * emax)
    nn = int(np.sum(m0))
    if nn == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    w0 = src["wsel"][m0]
    e1 = float(np.sum(w0 * src["fpfs_e1"][m0]))
    e2 = float(np.sum(w0 * src["fpfs_e2"][m0]))

    r1 = float(
        np.sum(
            src["dwsel_dg1"][m0] * src["fpfs_e1"][m0]
            + w0 * src["fpfs_de1_dg1"][m0]
        )
    )
    r2 = float(
        np.sum(
            src["dwsel_dg2"][m0] * src["fpfs_e2"][m0]
            + w0 * src["fpfs_de2_dg2"][m0]
        )
    )
    def sel_term(comp: int):
        e = src[f"fpfs_e{comp}"]
        de = src[f"fpfs_de{comp}_dg{comp}"]
        df = src[f"dflux_dg{comp}"]
        comp2 = int(3 - comp)
        e2 = src[f"fpfs_e{comp2}"]
        de2 = src[f"fpfs_de{comp2}_dg{comp}"]

        esq_p = esq0 + 2.0 * dg * (e * de + e2 * de2)
        m_p = ((src["flux"] + dg * df) > flux_min) & (esq_p < emax * emax)
        ellp = np.sum(src["wsel"][m_p] * e[m_p])

        esq_m = esq0 - 2.0 * dg * (e * de + e2 * de2)
        m_m = ((src["flux"] - dg * df) > flux_min) & (esq_m < emax * emax)
        ellm = np.sum(src["wsel"][m_m] * e[m_m])
        return (ellp - ellm) / (2.0 * dg)

    r1_sel = sel_term(1)
    r2_sel = sel_term(2)
    return e1, (r1 + r1_sel), e2, (r2 + r2_sel), nn


def per_rank_work(ids_chunk, outdir, flux_list, emax, dg, target):
    """
    For each ID in ids_chunk, read +g (mode1) and -g (mode0) catalogs,
    compute per-flux-cut e_pos/e_neg, R_pos/R_neg, N_pos/N_neg.
    Returns 6 arrays of shape (Nsamples_local, ncut).
    """
    ncut = len(flux_list)
    E_pos = []
    E_neg = []
    R_pos = []
    R_neg = []

    for i, sid in enumerate(ids_chunk):
        ppos = cat_path(outdir, sid, mode=1)  # +g
        pneg = cat_path(outdir, sid, mode=0)  # -g
        if not (os.path.exists(ppos) and os.path.exists(pneg)):
            # Skip if pair not complete
            continue

        src_pos = fitsio.read(ppos)
        src_neg = fitsio.read(pneg)

        e_pos_row = np.zeros(ncut)
        e_neg_row = np.zeros(ncut)
        R_pos_row = np.zeros(ncut)
        R_neg_row = np.zeros(ncut)

        for j, fmin in enumerate(flux_list):
            e1p, R1p, e2p, R2p, Np = measure_shear_flux_cut(
                src_pos, fmin, emax=emax, dg=dg
            )
            e1m, R1m, e2m, R2m, Nm = measure_shear_flux_cut(
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


def bootstrap_m(rng, e_pos, e_neg, R_pos, R_neg, shear_value, nsamp=10000):
    """
    e_pos/e_neg/R_pos/R_neg: (Nsamples_total, ncut)
    return: ms (nsamp, ncut)
    """
    N, ncut = e_pos.shape
    ms = np.zeros((nsamp, ncut))
    for i in range(nsamp):
        k = rng.integers(0, N, size=N, endpoint=False)
        num = np.sum(e_pos[k] - e_neg[k], axis=0)
        den = np.sum(R_pos[k] + R_neg[k], axis=0)
        new_gamma = num / den
        ms[i] = new_gamma / shear_value - 1.0
    return ms


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    flux_list = parse_flux_list(args.flux_mins)
    outdir = outdir_path(
        args.pscratch, args.layout, args.target, args.shear
    )
    # Build full ID list [min_id, max_id], split across ranks
    if args.max_id < args.min_id:
        raise SystemExit("--max-id must be >= --min-id")
    all_ids = np.arange(args.min_id, args.max_id + 1, dtype=int)

    # Even split
    n = len(all_ids)
    base = n // size
    rem = n % size
    start = rank * base + min(rank, rem)
    stop = start + base + (1 if rank < rem else 0)
    my_ids = all_ids[start:stop]

    # Per-rank measurement
    E_pos, E_neg, R_pos, R_neg = per_rank_work(
        my_ids,
        outdir,
        flux_list,
        args.emax,
        args.dg,
        args.target,
    )

    # Gather to rank 0
    gathered = comm.gather((E_pos, E_neg, R_pos, R_neg), root=0)
    if rank == 0:
        # Concatenate along sample axis (skip empties)
        all_E_pos = np.vstack([g[0] for g in gathered if g[0].size])
        all_E_neg = np.vstack([g[1] for g in gathered if g[1].size])
        all_R_pos = np.vstack([g[2] for g in gathered if g[2].size])
        all_R_neg = np.vstack([g[3] for g in gathered if g[3].size])

        if all_E_pos.size == 0 or all_E_neg.size == 0:
            raise SystemExit(
                "No valid (+g/-g) pairs found in the given ID range."
            )

        # m and c per flux cut
        num = np.sum(all_E_pos - all_E_neg, axis=0)  # (ncut,)
        den = np.sum(all_R_pos + all_R_neg, axis=0)
        m = (num / den) / args.shear - 1.0

        c = np.sum(all_E_pos + all_E_neg, axis=0) / np.sum(
            all_R_pos + all_R_neg, axis=0
        )

        # area & densities
        area_arcmin2 = (args.stamp_dim * args.stamp_dim) * (
            args.pixel_scale / 60.0
        ) ** 2.0

        clipped_mean, clipped_median, clipped_std = sigma_clipped_stats(
            all_E_pos / np.average(all_R_pos, axis=0),
            sigma=5.0,
        )
        neff = (0.26 / clipped_std) ** 2.0 / area_arcmin2

        rng = np.random.default_rng(0)
        ms = bootstrap_m(
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
        mean_m = np.mean(ms, axis=0)

        # Summary
        print("==============================================")
        print(f"Outdir: {outdir}")
        print(f"Paired IDs (found): {all_E_pos.shape[0]}")
        print(f"ID range requested: [{args.min_id}, {args.max_id}]")
        print(f"Flux cuts: {flux_list}")
        print(f"Area (arcmin^2): {area_arcmin2:.3f}")
        print("m (per flux cut):", m)
        print("c (per flux cut):", c)
        print("n_eff (per flux cut):", neff)
        print("m 1-sigma (bootstrap):", sigma_m)
        print("==============================================")


if __name__ == "__main__":
    main()
