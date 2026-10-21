#!/usr/bin/env python3
"""Generate truth catalogs and simulated multiband exposures for the
constant-shear tests. Each output seed produces one truth-<seed>.fits
plus one exp-<band>-<seed>.fits per band. Existing outputs are reused;
missing outputs are (re)generated per file.
"""
import argparse
import gc
import os

import common
import fitsio
import numpy as np
from mpi4py import MPI

from xlens.simulator.catalog import (
    CatalogShearTask,
    CatalogShearTaskConfig,
)
from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
COMM.Barrier()

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Simulate constant-shear multiband exposures (MPI optional)",
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
    "--no-image-noise", action="store_true",
    help="Render the galaxies without drawing the image noise. The variance "
         "plane is still set to the survey noise variance, so addnoise.py "
         "can add the noise afterwards (the same realisation sim.py would "
         "have drawn).",
)
parser.add_argument(
    "--psf", type=str, default=None,
    help="FITS PSF stamp to simulate with, instead of the analytic Moffat "
         "the task builds from psf_fwhm_defaults. 2D: the same PSF for every "
         "band; 3D: one stamp per band in u,g,r,i,z,y order. The stamp must "
         "ALREADY include the pixel response and sum to 1 (see "
         "make_moffat_psf.py), because the galaxies are then drawn with "
         "method='no_pixel'.",
)
parser.add_argument(
    "--psf-e1", type=float, default=0.0,
    help="Distortion e1 of the analytic Moffat PSF the task builds. Ignored "
         "when --psf is given, since the stamp then carries its own shape.",
)
parser.add_argument(
    "--psf-e2", type=float, default=0.0,
    help="Distortion e2 of the analytic Moffat PSF. See --psf-e1.",
)
parser.add_argument(
    "--tag", type=str, default=None,
    help="Name for this variant; outputs go to sim_mode<N>_<tag>/ instead "
         "of sim_mode<N>/ (e.g. --no-image-noise --tag noiseless).",
)
args, unknown_args = parser.parse_known_args()
if unknown_args:
    print("[warn] Ignoring unknown args:", unknown_args)

shear_mode = int(args.mode)
shear_value = args.shear
kappa_value = args.kappa
rot_id = args.rot
test_target = args.target
istart = args.start
iend = args.end
if iend - istart <= 0:
    raise ValueError(f"Invalid range: start={istart}, end={iend}")


bands = [b.strip() for b in args.band.split(",") if b.strip()]
if not bands:
    raise ValueError(f"Invalid --band argument: {args.band!r}")
if args.layout == "random":
    extend_ratio = 1.08
elif args.layout == "grid":
    extend_ratio = 0.92
else:
    raise ValueError("Cannot support layout")

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
cfg_cat = CatalogShearTaskConfig()
cfg_cat.galaxy_type = common.GALAXY_TYPE
cfg_cat.survey_name_list = common.SURVEY_NAME_LIST
cfg_cat.z_bounds = [0.0, 0.63, 0.98, 1.48, 10.0]
cfg_cat.mode = shear_mode
cfg_cat.rotId = rot_id
cfg_cat.kappa_value = kappa_value
cfg_cat.test_value = shear_value
cfg_cat.test_target = test_target
cfg_cat.layout = args.layout
cfg_cat.extend_ratio = extend_ratio
cfg_cat.sep_arcsec = 14
cat_task = CatalogShearTask(config=cfg_cat)

psf_array = None
if args.psf:
    psf_array = fitsio.read(args.psf).astype(np.float64)
    if psf_array.ndim not in (2, 3):
        raise ValueError(f"--psf must be 2D or 3D, got shape {psf_array.shape}")
    total = psf_array.sum() if psf_array.ndim == 2 else psf_array[0].sum()
    if abs(total - 1.0) >= 1e-2:
        raise ValueError(f"--psf stamp sums to {total:.6f}; MultibandSimTask "
                         "requires |sum - 1| < 1e-2")
    if RANK == 0:
        print(f"[Info] PSF from {args.psf} {psf_array.shape}, sum {total:.6f}; "
              "galaxies drawn with method='no_pixel'.")

cfg_sim = MultibandSimConfig()
cfg_sim.galaxy_type = common.GALAXY_TYPE
cfg_sim.survey_name = common.SURVEY_NAME_LIST[0]
cfg_sim.draw_image_noise = not args.no_image_noise
cfg_sim.use_real_psf = psf_array is not None
if psf_array is None:
    cfg_sim.psf_e1 = args.psf_e1
    cfg_sim.psf_e2 = args.psf_e2
    if (args.psf_e1 or args.psf_e2) and RANK == 0:
        print(f"[Info] analytic Moffat PSF sheared to "
              f"e = ({args.psf_e1:+g}, {args.psf_e2:+g}).")
elif args.psf_e1 or args.psf_e2:
    raise ValueError("--psf-e1/--psf-e2 apply to the analytic Moffat only; "
                     "the --psf stamp already carries its own shape")
sim_task = MultibandSimTask(config=cfg_sim)


# ------------------------------
# Output layout (sim products live under sim_mode<N>/; process.py reads
# from this dir and writes catalogs under process_mode<N>/).
# ------------------------------
outdir = common.sim_outdir(args.layout, test_target, shear_value, shear_mode,
                           tag=args.tag)


# ------------------------------
# Work loop (unique seeds per RANK if MPI)
# ------------------------------
for i in range(istart, iend):
    sim_seed = i * SIZE + RANK

    truthfname = common.truth_path(outdir, sim_seed)

    if os.path.isfile(truthfname):
        truth_catalog = fitsio.read(truthfname)
    else:
        truth_catalog = cat_task.run(
            tract_info=skymap[common.TRACT_ID],
            seed=sim_seed,
        ).truthCatalog
        fitsio.write(truthfname, truth_catalog)

    exp_fnames = {bb: common.exp_path(outdir, bb, sim_seed) for bb in bands}
    for bb in bands:
        exp_fname = exp_fnames[bb]
        if os.path.isfile(exp_fname):
            continue
        exp = sim_task.run(
            tract_info=skymap[common.TRACT_ID],
            patch_id=common.PATCH_ID,
            band=bb,
            seed=sim_seed,
            truthCatalog=truth_catalog,
            psfArray=psf_array,
        ).simExposure
        exp.writeFits(exp_fname)
        del exp
        gc.collect()

    del truth_catalog
    gc.collect()

# Ensure all ranks finish (no-op in single process)
COMM.Barrier()
if RANK == 0:
    print("Done.")
