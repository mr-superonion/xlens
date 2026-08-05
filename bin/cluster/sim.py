#!/usr/bin/env python3
"""Render simulated multiband exposures for the cluster test, one patch
per job index.

The global index encodes (realization, patch):
``sim_seed, patch_id = divmod(index, 9)``, so submitting indices
[0, 9*n_realizations) simulates every patch of every realization with
each job doing exactly one patch. The truth catalog of the realization
must already exist in sim_truth_rot<r>/ (run prepare_sim.py first);
indices whose truth file is missing are skipped with a note so the two
scripts can run independently and be resumed.

Each output is exp-<band>-<seed>-p<patch>.fits. Existing outputs are
reused; missing outputs are (re)generated per file.

The mode swept by run_bnl.sh IS the rotation id here (``--mode``
overrides ``--rot``), so both rotations come from one submission:

    ./run_bnl.sh --script cluster/sim.py --modes "0,1" --band i \
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

from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Simulate cluster-lensing patches (MPI optional)",
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
# Image Simulation Task
# ------------------------------
cfg_sim = MultibandSimConfig()
cfg_sim.galaxy_type = common.GALAXY_TYPE
cfg_sim.survey_name = common.SURVEY_NAME_LIST[0]
cfg_sim.draw_image_noise = True
cfg_sim.rotId = rot
sim_task = MultibandSimTask(config=cfg_sim)

truth_dir = common.truth_outdir(args.layout, rot)
outdir = common.sim_outdir(args.layout, rot)

# ------------------------------
# Work loop; the realized index set is independent of the rank count.
# ------------------------------
for index in range(istart + RANK, iend, SIZE):
    sim_seed, patch_id = common.seed_patch_from_index(index)

    truthfname = common.truth_path(truth_dir, sim_seed)
    if not os.path.isfile(truthfname):
        print(
            f"[skip] index={index} (seed={sim_seed}, patch={patch_id}): "
            f"truth catalog missing ({truthfname}); run prepare_sim.py first"
        )
        continue
    truth_catalog = fitsio.read(truthfname)

    for bb in bands:
        exp_fname = common.exp_path(outdir, bb, sim_seed, patch_id)
        if os.path.isfile(exp_fname):
            continue
        # The run seed only drives the noise realization (galaxies come
        # from the truth catalog), and the noise seed does NOT mix in
        # the patch id -- pass a per-(realization, patch) seed so the 9
        # patches get independent noise.
        exp = sim_task.run(
            tract_info=skymap[common.TRACT_ID],
            patch_id=patch_id,
            band=bb,
            seed=sim_seed * common.N_PATCH + patch_id,
            truthCatalog=truth_catalog,
        ).simExposure
        tmpfname = exp_fname + ".tmp.fits"
        exp.writeFits(tmpfname)
        os.replace(tmpfname, exp_fname)
        del exp
        gc.collect()

    del truth_catalog
    gc.collect()

# Ensure all ranks finish (no-op in single process)
if COMM is not None:
    COMM.Barrier()
if RANK == 0:
    print("Done.")
