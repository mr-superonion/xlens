#!/usr/bin/env python3
"""Process the simulated cluster-test exposures (produced by sim.py)
into per-patch measured catalogs, one patch per job index.

The global index encodes (realization, patch):
``sim_seed, patch_id = divmod(index, 9)``. For each index, reads
exp-<band>-<seed>-p<patch>.fits per band, runs AnaCal detection +
forced FPFS measurement on the patch, keeps only ``is_primary``
sources (inner-patch region, so the 100-pixel patch overlaps are not
double counted), and writes cat-<seed>-p<patch>.fits. Indices whose
sim outputs are missing are skipped (with a note), so the scripts can
run independently and be resumed.

Detection runs on ``--detection-band`` (default 'i'); give it several
bands to detect on their inverse-variance coadd, e.g.

    process.py --band r,i,z --detection-band r,i,z --version 1

Forced measurement is per band over ``--band`` regardless, so the
output columns are the same either way.

The mode swept by run_bnl.sh IS the rotation id here (``--mode``
overrides ``--rot``), so both rotations come from one submission:

    ./run_bnl.sh --script cluster/process.py --modes "0,1" --band i \
        --index-start 0 --index-end 89 --per-task 9

(--shear/--kappa/--target injected by run_bnl.sh are accepted and
ignored.)
"""
import argparse
import gc
import os

import common
import fitsio

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    COMM.Barrier()
except ImportError:  # single-process fallback
    COMM = None
    RANK = 0
    SIZE = 1

from lsst.afw.image import ExposureF

from xlens.processor.measure_coadds import (
    MeasureCoaddsPipe,
    MeasureCoaddsPipeConfig,
)
from xlens.utils.handle import make_exposure_handles

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Process simulated cluster-lensing patches (MPI optional)",
    allow_abbrev=False,
)
parser.add_argument(
    "--start", type=int, default=0,
    help="start job index (inclusive); index = 9*sim_seed + patch_id",
)
parser.add_argument(
    "--end", type=int, default=9,
    help="end job index (exclusive)",
)
parser.add_argument(
    "--mode", type=int, default=None, choices=[0, 1],
    help="rotation id (alias of --rot, takes precedence when given; "
         "run_bnl.sh sweeps this via --modes \"0,1\")",
)
parser.add_argument(
    "--rot", type=int, default=0, choices=[0, 1], help="rotation id",
)
parser.add_argument(
    "--layout", type=str, default="random",
    choices=["grid", "random"], help="layout",
)
parser.add_argument(
    "--band", type=str, default="i",
    help="comma-separated bands list (e.g. 'r,i,z')",
)
parser.add_argument(
    "--detection-band", type=str, default="i",
    help="comma-separated bands to coadd for DETECTION (e.g. 'r,i,z'); "
         "must be a subset of --band. AnaCal removes each band's own PSF "
         "before averaging them with inverse-variance weights, so the bands "
         "need not be PSF-matched. Forced measurement still runs band by "
         "band over --band either way. Use --version to keep catalogs from "
         "different detection bands in separate directories.",
)
parser.add_argument(
    "--version", type=int, default=None,
    help="Optional integer tag; when set, catalogs go to "
         "process_rot<r>-v<version>/ instead of process_rot<r>/.",
)
args, unknown_args = parser.parse_known_args()
if unknown_args:
    print("[warn] Ignoring unknown args:", unknown_args)

rot = args.mode if args.mode is not None else args.rot
istart = args.start
iend = args.end
if iend - istart <= 0:
    raise ValueError(f"Invalid range: start={istart}, end={iend}")

bands = [b.strip() for b in args.band.split(",") if b.strip()]
if not bands:
    raise ValueError(f"Invalid --band argument: {args.band!r}")

detection_bands = [
    b.strip() for b in args.detection_band.split(",") if b.strip()
]
if not detection_bands:
    raise ValueError(
        f"Invalid --detection-band argument: {args.detection_band!r}"
    )
# Detection reads the same exposures the forced measurement does, so it can
# only use bands that were loaded.
missing_det = [b for b in detection_bands if b not in bands]
if missing_det:
    raise ValueError(
        f"--detection-band {missing_det} not in --band {bands}; "
        f"add them to --band so the exposures get loaded"
    )

if RANK == 0:
    if SIZE == 1:
        print("[Info] Running single-process (no mpirun/srun needed).")
    else:
        print(f"[Info] Running with MPI across {SIZE} ranks.")
    print(f"[Info] Detecting on {','.join(detection_bands)}; "
          f"forced measurement on {','.join(bands)}.")

# ------------------------------
# SkyMap Setup
# ------------------------------
skymap = common.build_skymap()
if RANK == 0:
    print("SkyMap created.")

# ------------------------------
# Detection (on detection_bands) + per-band forced measurement (all bands),
# with the same kernels as examples/cluster/example1_cluster_lensing.ipynb.
# ------------------------------
detect_config = MeasureCoaddsPipeConfig()
detect_config.anacal.sigma_arcsec = 0.52   # detection / e1, e2, w kernel
detect_config.fpfs.sigma_shapelets1 = 0.45  # kernel 1
detect_config.fpfs.sigma_shapelets2 = 0.55  # kernel 2
detect_config.use_sim = False
detect_config.detection_bands = detection_bands
detect_config.validate()
meas_task = MeasureCoaddsPipe(config=detect_config)

# ------------------------------
# Output layout: prepare_sim.py/sim.py write truth/exposures under sim/;
# process.py reads from there and writes catalogs under process/.
# ------------------------------
sim_dir = common.sim_outdir(args.layout, rot)
proc_dir = common.process_outdir(args.layout, rot, version=args.version)

# ------------------------------
# Work loop; the realized index set is independent of the rank count.
# ------------------------------
for index in range(istart + RANK, iend, SIZE):
    sim_seed, patch_id = common.seed_patch_from_index(index)

    outfname = common.cat_path(proc_dir, sim_seed, patch_id)
    if os.path.isfile(outfname):
        continue

    exp_fnames = {
        bb: common.exp_path(sim_dir, bb, sim_seed, patch_id) for bb in bands
    }
    # Skip if any required sim output is missing (sim.py hasn't produced
    # it yet, or was run with a different band list).
    missing = [p for p in exp_fnames.values() if not os.path.isfile(p)]
    if missing:
        print(
            f"[skip] index={index} (seed={sim_seed}, patch={patch_id}): "
            f"{len(missing)} sim file(s) missing (first: {missing[0]})"
        )
        continue

    exposures = {bb: ExposureF(exp_fnames[bb]) for bb in bands}

    handles = make_exposure_handles(
        exposures,
        skymap="test",
        tract=common.TRACT_ID,
        patch=patch_id,
        skyMap=skymap,
    )
    res = meas_task.run(
        exposure_handles_dict=handles,
        corr_array=None,
        skyMap=skymap,
        tract=common.TRACT_ID,
        patch=patch_id,
        mask=None,
        detection=None,
        # Per-(realization, patch) seed for the noise-bias-correction
        # noise; offset so it never collides with the sim noise seeds.
        seed=100000 + index,
    ).anacalCatalog

    # Keep only inner-patch (and inner-tract) sources so the 100-pixel
    # patch overlaps are not double counted when patches are combined.
    res = res[res["is_primary"]]

    tmpfname = outfname + ".tmp"
    fitsio.write(tmpfname, res, clobber=True)
    os.replace(tmpfname, outfname)

    # clean up
    del exposures, handles, res
    gc.collect()

# Ensure all ranks finish (no-op in single process)
if COMM is not None:
    COMM.Barrier()
if RANK == 0:
    print("Done.")
