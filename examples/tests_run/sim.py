#!/usr/bin/env python3
import argparse
import gc
import os

import fitsio
import numpy as np
from lsst.skymap.discreteSkyMap import DiscreteSkyMap, DiscreteSkyMapConfig
from mpi4py import MPI

from xlens.processor.measure_coadds import (
    MeasureCoaddsPipe,
    MeasureCoaddsPipeConfig,
)
from xlens.processor.match import (
    matchPipe,
    matchPipeConfig,
)
from xlens.simulator.catalog import (
    CatalogShearTask,
    CatalogShearTaskConfig,
)
from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask
from xlens.utils.handle import make_exposure_handles

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
COMM.Barrier()

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Run constant shear simulation (MPI optional)",
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
tract_id = 0
patch_id = 0
pixel_scale = 0.2  # arcsec/pixel
config = DiscreteSkyMapConfig()
config.raList = [0.0]
config.decList = [0.0]
config.radiusList = [0.1]
config.rotation = 0.0
config.projection = "TAN"
config.patchInnerDimensions = [256, 256]
config.patchBorder = 0
config.pixelScale = pixel_scale
config.tractOverlap = 0.0
skymap = DiscreteSkyMap(config)
if RANK == 0:
    print("SkyMap created.")

# ------------------------------
# Image Simulation Task
# ------------------------------
cfg_cat = CatalogShearTaskConfig()
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

cfg_sim = MultibandSimConfig()
cfg_sim.survey_name = "lsst"
cfg_sim.draw_image_noise = True
sim_task = MultibandSimTask(config=cfg_sim)

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


# Outdir layout:
#   $PSCRATCH/constant_shear_<layout>/<target>/shearXX/
pscratch = os.environ.get("PSCRATCH", ".")
outdir = os.path.join(
    pscratch,
    f"constant_shear_{args.layout}",
    test_target,
    f"shear{int(shear_value * 100):02d}",
    f"mode{shear_mode}",
)
os.makedirs(outdir, exist_ok=True)


# ------------------------------
# Work loop (unique seeds per RANK if MPI)
# ------------------------------
for i in range(istart, iend):
    sim_seed = i * SIZE + RANK
    outfname = os.path.join(outdir, "cat-%05d.fits" % (sim_seed))
    if os.path.isfile(outfname):
        continue

    truth_catalog = cat_task.run(
        tract_info=skymap[tract_id],
        seed=sim_seed,
    ).truthCatalog
    truthfname = os.path.join(outdir, "truth-%05d.fits" % (sim_seed))
    if not os.path.isfile(truthfname):
        fitsio.write(truthfname, truth_catalog)

    exposures = {
        bb: sim_task.run(
            tract_info=skymap[tract_id],
            patch_id=patch_id,
            band=bb,
            seed=sim_seed,
            truthCatalog=truth_catalog,
        ).simExposure
        for bb in bands
    }

    handles = make_exposure_handles(
        exposures,
        skymap="test",
        tract=tract_id,
        patch=patch_id,
        skyMap=skymap,
    )
    res = meas_task.run(
        exposure_handles_dict=handles,
        corr_array=None,
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
        mask=None,
        detection=None,
        seed=100000 + sim_seed,
    ).anacalCatalog

    res = match_task.run(
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
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
