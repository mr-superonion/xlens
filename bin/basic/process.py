#!/usr/bin/env python3
"""Process simulated multiband exposures (produced by sim.py) into
matched catalogs. For each seed, reads truth-<seed>.fits and
exp-<band>-<seed>.fits per band, runs detection + forced measurement +
truth match, and writes cat-<seed>.fits. Seeds whose sim outputs are
missing are skipped (with a note), so the two scripts can run
independently and be resumed.
"""
import argparse
import gc
import os

import fitsio
import numpy as np
from lsst.afw.image import ExposureF
from mpi4py import MPI

from xlens.processor.match import (
    matchPipe,
    matchPipeConfig,
)
from xlens.processor.measure_coadds import (
    MeasureCoaddsPipe,
    MeasureCoaddsPipeConfig,
)
from xlens.utils.handle import make_exposure_handles

import common

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
COMM.Barrier()

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Process simulated constant-shear exposures (MPI optional)",
    allow_abbrev=False,
)
parser.add_argument("--target", type=str, default="g1", help="test target")
parser.add_argument(
    "--mode", type=int, default=0, choices=[40, 0, 27, 9, 3, 1, 36, 4, 80],
    help="40:++++;0:----;27:+---;9:-+--;3:--+-;1:---+;36:++--;4:--++;80:0000"
)
parser.add_argument(
    "--rot", type=int, default=0, choices=[0, 1], help="rotation id",
)
parser.add_argument(
    "--start", type=int, default=0, help="start id (inclusive)",
)
parser.add_argument(
    "--end", type=int, default=10, help="end id (exclusive)",
)
parser.add_argument(
    "--shear", type=float, default=0.02, help="Shear value",
)
parser.add_argument(
    "--kappa", type=float, default=0.00, help="Kappa value",
)
parser.add_argument(
    "--layout", type=str, default="grid",
    choices=["grid", "random"], help="layout",
)
parser.add_argument(
    "--band", type=str, default="u,g,r,i,z,y",
    help="comma-separated bands list (e.g. 'r,i,z')",
)
parser.add_argument(
    "--version", type=int, default=None,
    help="Optional integer tag; when set, catalogs go to "
         "process_mode<N>-v<version>/ instead of process_mode<N>/.",
)
args, unknown_args = parser.parse_known_args()
if unknown_args:
    print("[warn] Ignoring unknown args:", unknown_args)

shear_mode = int(args.mode)
shear_value = args.shear
test_target = args.target
istart = args.start
iend = args.end
if iend - istart <= 0:
    raise ValueError(f"Invalid range: start={istart}, end={iend}")


bands = [b.strip() for b in args.band.split(",") if b.strip()]
if not bands:
    raise ValueError(f"Invalid --band argument: {args.band!r}")

if RANK == 0:
    if SIZE == 1:
        print("[Info] Running single-process (no mpirun/srun needed).")
    else:
        print(f"[Info] Running with MPI across {SIZE} ranks.")

# ------------------------------
# SkyMap Setup
# ------------------------------
skymap = common.build_skymap()
if RANK == 0:
    print("SkyMap created.")

# ------------------------------
# Detection + per-band forced measurement (all bands together)
# ------------------------------
detect_config = MeasureCoaddsPipeConfig()
detect_config.anacal.sigma_arcsec = 0.38
detect_config.anacal.force_size = True
detect_config.anacal.num_epochs = 0
detect_config.anacal.do_noise_bias_correction = True
detect_config.fpfs.do_noise_bias_correction = True
detect_config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
detect_config.use_sim = False
meas_task = MeasureCoaddsPipe(config=detect_config)

config = matchPipeConfig()
config.mag_zero = 30.0
config.mag_max_truth = 28.0
match_task = matchPipe(config=config)


# ------------------------------
# Output layout: sim.py writes truth/exposures under sim_mode<N>/;
# process.py reads from there and writes catalogs under process_mode<N>/.
# ------------------------------
sim_dir = common.sim_outdir(
    args.layout, test_target, shear_value, shear_mode,
)
proc_dir = common.process_outdir(
    args.layout, test_target, shear_value, shear_mode, version=args.version,
)


# ------------------------------
# Work loop (unique seeds per RANK if MPI)
# ------------------------------
for i in range(istart, iend):
    sim_seed = i * SIZE + RANK

    outfname = common.cat_path(proc_dir, sim_seed)
    if os.path.isfile(outfname):
        continue

    truthfname = common.truth_path(sim_dir, sim_seed)
    exp_fnames = {bb: common.exp_path(sim_dir, bb, sim_seed) for bb in bands}

    # Skip if any required sim output is missing (sim.py hasn't produced
    # it yet, or was run with a different band list).
    missing = [p for p in (truthfname, *exp_fnames.values())
               if not os.path.isfile(p)]
    if missing:
        print(
            f"[skip] seed={sim_seed}: {len(missing)} sim file(s) missing "
            f"(first: {missing[0]})"
        )
        continue

    truth_catalog = fitsio.read(truthfname)
    exposures = {bb: ExposureF(exp_fnames[bb]) for bb in bands}

    handles = make_exposure_handles(
        exposures,
        skymap="test",
        tract=common.TRACT_ID,
        patch=common.PATCH_ID,
        skyMap=skymap,
    )
    res = meas_task.run(
        exposure_handles_dict=handles,
        corr_array=None,
        skyMap=skymap,
        tract=common.TRACT_ID,
        patch=common.PATCH_ID,
        mask=None,
        detection=None,
        seed=100000 + sim_seed,
    ).anacalCatalog

    res = match_task.run(
        skyMap=skymap,
        tract=common.TRACT_ID,
        patch=common.PATCH_ID,
        catalog=res,
        dm_catalog=None,
        truth_catalog=truth_catalog,
    ).catalog
    res = res[res["wsel"] > 1e-7]
    fitsio.write(outfname, res)

    # clean up
    del exposures, handles, truth_catalog, res
    gc.collect()

# Ensure all ranks finish (no-op in single process)
COMM.Barrier()
if RANK == 0:
    print("Done.")
