#!/usr/bin/env python3
"""Generate the truth catalog for each realization of the cluster test.

Each realization (sim_seed) gets one truth-<seed>.fits covering the
WHOLE tract (all 9 patches), with the NFW halo of common.py at the
tract centre (ra=0, dec=0). sim.py then renders the 9 patches of a
realization from this shared truth catalog, so prepare_sim.py must run
first. Existing outputs are reused; missing outputs are (re)generated.

Job indices map directly to realizations: --start/--end select
sim_seed in [start, end).  The mode swept by run_bnl.sh IS the
rotation id here (``--mode`` overrides ``--rot``), so both rotations
of every realization come from one submission:

    ./run_bnl.sh --script cluster/prepare_sim.py --modes "0,1" \
        --index-start 0 --index-end 9 --per-task 10

(--shear/--kappa/--target/--band injected by run_bnl.sh are accepted
and ignored.)
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

from xlens.simulator.catalog import (
    CatalogHaloTask,
    CatalogHaloTaskConfig,
)

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Write cluster-lensing truth catalogs (MPI optional)",
    allow_abbrev=False,
)
parser.add_argument(
    "--start", type=int, default=0, help="start realization id (inclusive)",
)
parser.add_argument(
    "--end", type=int, default=10, help="end realization id (exclusive)",
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
args, unknown_args = parser.parse_known_args()
if unknown_args:
    print("[warn] Ignoring unknown args:", unknown_args)

rot = args.mode if args.mode is not None else args.rot
istart = args.start
iend = args.end
if iend - istart <= 0:
    raise ValueError(f"Invalid range: start={istart}, end={iend}")

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
# Truth-catalog task (NFW halo at the tract centre)
# ------------------------------
cfg_cat = CatalogHaloTaskConfig()
cfg_cat.mass = common.HALO_MASS
cfg_cat.conc = common.HALO_CONC
cfg_cat.z_lens = common.Z_LENS
cfg_cat.z_source = common.Z_SOURCE
cfg_cat.rotId = rot
cfg_cat.layout = args.layout
cfg_cat.validate()
cat_task = CatalogHaloTask(config=cfg_cat)

outdir = common.sim_outdir(args.layout, rot)

# ------------------------------
# Work loop; the realized seed set is independent of the rank count.
# ------------------------------
for sim_seed in range(istart + RANK, iend, SIZE):
    truthfname = common.truth_path(outdir, sim_seed)
    if os.path.isfile(truthfname):
        continue
    truth_catalog = cat_task.run(
        tract_info=skymap[common.TRACT_ID],
        seed=sim_seed,
    ).truthCatalog
    # Atomic write: no partial file is ever visible to sim.py jobs.
    tmpfname = truthfname + ".tmp"
    fitsio.write(tmpfname, truth_catalog, clobber=True)
    os.replace(tmpfname, truthfname)
    del truth_catalog
    gc.collect()

# Ensure all ranks finish (no-op in single process)
if COMM is not None:
    COMM.Barrier()
if RANK == 0:
    print("Done.")
