#!/usr/bin/env python3
"""Aggregate the cluster-test catalogs into a tangential shear profile.

Two phases, driven by run_bnl_summary.sh:

1. Partial mode (default): for each realization in [--min-id, --max-id)
   read the 9 per-patch catalogs (both rotations when present) written
   by process.py, rotate shapes/responses about the halo centre
   (ra=0, dec=0), and accumulate per-radial-bin sums of w*eT, w*eX,
   rT, rX, the response-weighted true tangential shear (interpolated
   from the truth catalog), and galaxy counts. One .npz partial file
   is written per job.
2. --summary: stack all partial files and print the binned profile
   gT = sum(w eT)/sum(rT), gX, the true gT, the multiplicative bias
   m = gT/gT_true - 1 per bin, with bootstrap errors over realizations.

Extra arguments injected by run_bnl_summary.sh (--emax, --width-max,
--z-bounds, --bands, --redshift, ...) are accepted and ignored, same
as bin/basic/summary.py.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
from typing import List, Optional

import common
import fitsio
import numpy as np

from xlens.analysis.cluster import HaloMcBiasMultibandPipe as ClusterPipe

colnames = [
    "wsel",
    "dwsel_dg1",
    "dwsel_dg2",
    "fpfs_e1",
    "fpfs_de1_dg1",
    "fpfs_e2",
    "fpfs_de2_dg2",
    "ra",
    "dec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure + aggregate the cluster tangential shear profile "
            "over a given realization ID range."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--summary", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="random",
        choices=["grid", "random"],
        help="Layout used in path naming.",
    )
    # ID range (realizations, NOT job indices: one realization = 9 patches)
    parser.add_argument(
        "--min-id",
        type=int,
        default=0,
        help="Minimum sim_seed (inclusive).",
    )
    parser.add_argument(
        "--max-id",
        type=int,
        default=1,
        help="Maximum sim_seed (exclusive).",
    )
    # Radial binning about the halo centre
    parser.add_argument(
        "--rmin",
        type=float,
        default=15.0,
        help="Inner edge of the radial binning [arcsec].",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=420.0,
        help="Outer edge of the radial binning [arcsec].",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=10,
        help="Number of radial bins.",
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=common.PIXEL_SCALE,
        help="Pixel scale (arcsec/pixel); must match common.PIXEL_SCALE.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="# bootstrap resamples over realizations for uncertainties",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Optional integer tag; reads catalogs from "
             "process_rot<r>-v<version>/ and writes partials to "
             "summary-v<version>/.",
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("[warn] Ignoring unknown args:", unknown_args)
    if abs(args.pixel_scale - common.PIXEL_SCALE) > 1e-8:
        raise SystemExit(
            f"--pixel-scale {args.pixel_scale} does not match the skymap "
            f"pixel scale {common.PIXEL_SCALE} in common.py"
        )
    return args


def radial_bin_edges(args) -> np.ndarray:
    return np.linspace(args.rmin, args.rmax, args.nbins + 1)


def read_realization(layout, rot, sim_seed, version=None):
    """Read the 9 per-patch catalogs and the truth catalog of one
    (realization, rotation). Returns (catalog, truth) or None if any
    file is missing."""
    proc_dir = common.process_outdir(layout, rot, version=version)
    truth_dir = common.truth_outdir(layout, rot)
    truthfname = common.truth_path(truth_dir, sim_seed)
    cat_fnames = [
        common.cat_path(proc_dir, sim_seed, patch)
        for patch in range(common.N_PATCH)
    ]
    missing = [
        p for p in (truthfname, *cat_fnames) if not os.path.isfile(p)
    ]
    if missing:
        return None
    cats = [fitsio.read(p, columns=colnames) for p in cat_fnames]
    catalog = np.concatenate(cats)
    truth = fitsio.read(
        truthfname, columns=["ra", "dec", "gamma1", "gamma2", "kappa"],
    )
    return catalog, truth


def accumulate_bins(catalog, truth, edges):
    """Per-radial-bin sums for one (realization, rotation) about the
    halo centre (common.RA_CENTER, common.DEC_CENTER)."""
    ra = np.asarray(catalog["ra"], dtype=np.float64)
    dec = np.asarray(catalog["dec"], dtype=np.float64)
    angle = ClusterPipe.position_angle_ccw_from_east(
        common.RA_CENTER, common.DEC_CENTER, ra, dec,
    ).rad
    dist = ClusterPipe.angsep(ra, dec, common.RA_CENTER, common.DEC_CENTER)

    e1 = catalog["fpfs_e1"]
    e2 = catalog["fpfs_e2"]
    w = catalog["wsel"]
    eT, eX = ClusterPipe._rotate_spin_2_vec(e1, e2, angle)
    r11, r22 = ClusterPipe._get_response_from_w_and_der(
        e1,
        e2,
        w,
        catalog["fpfs_de1_dg1"],
        catalog["fpfs_de2_dg2"],
        catalog["dwsel_dg1"],
        catalog["dwsel_dg2"],
    )
    # Responses rotate as a (diagonal) matrix, not as a spin-2 vector.
    rT, rX = ClusterPipe._rotate_spin_2_matrix(r11, r22, angle)

    # True tangential reduced shear, interpolated in radius from the
    # truth catalog (the NFW profile is isotropic about the centre).
    g1_true = truth["gamma1"] / (1.0 - truth["kappa"])
    g2_true = truth["gamma2"] / (1.0 - truth["kappa"])
    true_angle = ClusterPipe.position_angle_ccw_from_east(
        common.RA_CENTER,
        common.DEC_CENTER,
        np.asarray(truth["ra"], dtype=np.float64),
        np.asarray(truth["dec"], dtype=np.float64),
    ).rad
    gT_true_all, _ = ClusterPipe._rotate_spin_2_vec(
        g1_true, g2_true, true_angle,
    )
    true_dist = ClusterPipe.angsep(
        np.asarray(truth["ra"], dtype=np.float64),
        np.asarray(truth["dec"], dtype=np.float64),
        common.RA_CENTER,
        common.DEC_CENTER,
    )
    order = np.argsort(true_dist)
    gT_true = np.interp(
        dist, true_dist[order], gT_true_all[order],
    )

    n_bins = len(edges) - 1
    sums = {
        "sum_w_eT": np.zeros(n_bins),
        "sum_w_eX": np.zeros(n_bins),
        "sum_rT": np.zeros(n_bins),
        "sum_rX": np.zeros(n_bins),
        "sum_rT_gT_true": np.zeros(n_bins),
        "n_gal": np.zeros(n_bins),
    }
    for i in range(n_bins):
        mask = (dist >= edges[i]) & (dist < edges[i + 1])
        sums["sum_w_eT"][i] = np.sum(w[mask] * eT[mask])
        sums["sum_w_eX"][i] = np.sum(w[mask] * eX[mask])
        sums["sum_rT"][i] = np.sum(rT[mask])
        sums["sum_rX"][i] = np.sum(rX[mask])
        sums["sum_rT_gT_true"][i] = np.sum(rT[mask] * gT_true[mask])
        sums["n_gal"][i] = np.sum(mask)
    return sums


def per_rank_work(args, edges):
    """Accumulate one row per realization (rotations combined)."""
    keys = [
        "sum_w_eT", "sum_w_eX", "sum_rT", "sum_rX",
        "sum_rT_gT_true", "n_gal",
    ]
    rows = {key: [] for key in keys}
    n_done = 0
    for sim_seed in range(args.min_id, args.max_id):
        row = {key: np.zeros(len(edges) - 1) for key in keys}
        n_rot = 0
        for rot in (0, 1):
            data = read_realization(
                args.layout, rot, sim_seed, version=args.version,
            )
            if data is None:
                continue
            catalog, truth = data
            sums = accumulate_bins(catalog, truth, edges)
            for key in keys:
                row[key] += sums[key]
            n_rot += 1
            del catalog, truth
            gc.collect()
        if n_rot == 0:
            print(f"[skip] seed={sim_seed}: no complete rotation found")
            continue
        for key in keys:
            rows[key].append(row[key])
        n_done += 1
    if n_done == 0:
        return None
    return {key: np.vstack(rows[key]) for key in keys}


def save_partial(args, edges, stacked) -> str:
    partdir = common.summary_outdir(args.layout, version=args.version)
    path = os.path.join(partdir, f"seed_{args.min_id:05d}.npz")
    np.savez_compressed(path, edges=edges, **stacked)
    return path


def load_and_stack_all(args):
    partdir = common.summary_outdir(args.layout, version=args.version)
    keys = [
        "sum_w_eT", "sum_w_eX", "sum_rT", "sum_rX",
        "sum_rT_gT_true", "n_gal",
    ]
    blocks: dict[str, List[np.ndarray]] = {key: [] for key in keys}
    edges: Optional[np.ndarray] = None
    for path in sorted(glob.glob(os.path.join(partdir, "*.npz"))):
        with np.load(path) as data:
            if edges is None:
                edges = data["edges"]
            elif not np.allclose(edges, data["edges"]):
                raise SystemExit(
                    f"{path} was made with different radial bins; "
                    f"clear {partdir} and rerun the partials"
                )
            for key in keys:
                blocks[key].append(data[key])
    if edges is None:
        raise SystemExit(f"No partial files found in {partdir}")
    return edges, {key: np.vstack(blocks[key]) for key in keys}


def bootstrap_profiles(rng, stacked, nsamp):
    """Bootstrap gT and m over realizations. Returns (sigma_gT, sigma_m)."""
    n_real, n_bins = stacked["sum_w_eT"].shape
    gts = np.zeros((nsamp, n_bins))
    ms = np.zeros((nsamp, n_bins))
    for idx in range(nsamp):
        choices = rng.integers(0, n_real, size=n_real, endpoint=False)
        rt = np.sum(stacked["sum_rT"][choices], axis=0)
        gt = np.sum(stacked["sum_w_eT"][choices], axis=0) / rt
        gt_true = np.sum(stacked["sum_rT_gT_true"][choices], axis=0) / rt
        gts[idx] = gt
        ms[idx] = gt / gt_true - 1.0
    lo_idx = int(0.1587 * nsamp)
    hi_idx = int(0.8413 * nsamp)
    ord_gts = np.sort(gts, axis=0)
    ord_ms = np.sort(ms, axis=0)
    sigma_gt = (ord_gts[hi_idx] - ord_gts[lo_idx]) / 2.0
    sigma_m = (ord_ms[hi_idx] - ord_ms[lo_idx]) / 2.0
    return sigma_gt, sigma_m


def main() -> None:
    args = parse_args()
    if args.max_id <= args.min_id:
        raise SystemExit("--max-id must be > --min-id")
    edges = radial_bin_edges(args)

    if not args.summary:
        stacked = per_rank_work(args, edges)
        if stacked is None:
            print("[warn] no complete realization in the ID range; "
                  "no partial written")
            return
        path = save_partial(args, edges, stacked)
        print(f"Wrote {path} ({stacked['sum_w_eT'].shape[0]} realizations)")
    else:
        edges, stacked = load_and_stack_all(args)
        n_real = stacked["sum_w_eT"].shape[0]

        sum_rT = np.sum(stacked["sum_rT"], axis=0)
        sum_rX = np.sum(stacked["sum_rX"], axis=0)
        gT = np.sum(stacked["sum_w_eT"], axis=0) / sum_rT
        gX = np.sum(stacked["sum_w_eX"], axis=0) / sum_rX
        gT_true = np.sum(stacked["sum_rT_gT_true"], axis=0) / sum_rT
        m = gT / gT_true - 1.0
        n_gal = np.sum(stacked["n_gal"], axis=0)

        rng = np.random.default_rng(0)
        sigma_gT, sigma_m = bootstrap_profiles(
            rng, stacked, args.bootstrap,
        )

        r_mid = 0.5 * (edges[:-1] + edges[1:])
        print("==============================================")
        print(f"Catalog root: {common.summary_outdir(args.layout, version=args.version)}")
        print(f"Realizations (rot0+rot1 combined): {n_real}")
        header = (
            f"{'r[arcsec]':>10} {'n_gal':>8} {'gT':>10} {'sigma_gT':>10} "
            f"{'gX':>10} {'gT_true':>10} {'m':>10} {'sigma_m':>10}"
        )
        print(header)
        for i in range(len(r_mid)):
            print(
                f"{r_mid[i]:>10.1f} {int(n_gal[i]):>8d} {gT[i]:>10.5f} "
                f"{sigma_gT[i]:>10.5f} {gX[i]:>10.5f} {gT_true[i]:>10.5f} "
                f"{m[i]:>10.4f} {sigma_m[i]:>10.4f}"
            )
        print("==============================================")


if __name__ == "__main__":
    main()
