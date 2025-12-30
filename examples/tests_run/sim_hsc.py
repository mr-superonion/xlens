#!/usr/bin/env python3
import argparse
import gc
import os

import fitsio
import numpy as np
from lsst.skymap.discreteSkyMap import DiscreteSkyMap, DiscreteSkyMapConfig
import lsst.afw.image as afwImage

from mpi4py import MPI
from numpy.lib import recfunctions as rfn

from xlens.process_pipe.anacal_detect import (
    AnacalDetectPipe,
    AnacalDetectPipeConfig,
)
from xlens.process_pipe.match import (
    matchPipe,
    matchPipeConfig,
)
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
    "--end", type=int, default=2, help="end id (exclusive)",
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
    "--band", type=str, default="a",
    help="single band (g,r,i,z,y) or a for multiband",
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


band = args.band
if band not in "grizya":
    raise ValueError("Band not in [grizya]")
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
config = DiscreteSkyMapConfig()
config.raList = [0.0]
config.decList = [0.0]
config.radiusList = [0.1]
config.rotation = 0.0
config.projection = "TAN"
config.patchInnerDimensions = [4000, 4000]
config.patchBorder = 0
config.pixelScale = 0.168
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
cfg_sim.survey_name = "hsc"
cfg_sim.draw_image_noise = True
# cfg_sim.truncate_stamp_size = 65
sim_task = MultibandSimTask(config=cfg_sim)

# ------------------------------
# Detection Task
# ------------------------------
detect_config = AnacalDetectPipeConfig()
detect_config.anacal.sigma_arcsec = 0.37
detect_config.anacal.force_size = True
detect_config.anacal.num_epochs = 0
detect_config.anacal.do_noise_bias_correction = True
detect_config.do_fpfs = (args.band == "a")
detect_config.fpfs.sigma_shapelets1 = 0.37 * np.sqrt(2.0)
det_task = AnacalDetectPipe(config=detect_config)

config = matchPipeConfig()
config.mag_zero = 27.0
config.mag_max_truth = 28.0
match_task = matchPipe(config=config)


# Outdir layout:
#   $PSCRATCH/constant_shear_<layout>/<target>/shearXX/
pscratch = os.environ.get("PSCRATCH", ".")
outdir = os.path.join(
    pscratch,
    f"constant_shear_{args.layout}-2",
    test_target,
    f"shear{int(shear_value * 100):02d}",
    f"mode{shear_mode}",
)
os.makedirs(outdir, exist_ok=True)

full = fitsio.read(
    os.path.join(
        pscratch,
        "tracts_fdfc_v1_final_sims.fits"
    )
)
n_patches = len(full)

colnames = [
    "flux_gauss0",
    "dflux_gauss0_dg1",
    "dflux_gauss0_dg2",
    "flux_gauss0_err",
    "flux_gauss2",
    "dflux_gauss2_dg1",
    "dflux_gauss2_dg2",
    "flux_gauss2_err",
    "flux_gauss4",
    "dflux_gauss4_dg1",
    "dflux_gauss4_dg2",
    "flux_gauss4_err",
]


# ------------------------------
# Work loop (unique seeds per RANK if MPI)
# ------------------------------
for i in range(istart, iend):
    sim_seed = i * SIZE + RANK

    if band == "a":
        detection = None
        outfname = os.path.join(
            outdir, "cat-%05d.fits" % (sim_seed)
        )
        band_use = "i"
    else:
        detfname = os.path.join(
            outdir, "cat-%05d.fits" % (sim_seed)
        )
        if os.path.isfile(detfname):
            detection = fitsio.read(detfname)
        else:
            raise ValueError("Run detection with band=a first")
        outfname = os.path.join(
            outdir, "cat-%05d-%s.fits" % (sim_seed, band)
        )
        band_use = band
    if os.path.isfile(outfname):
        continue

    entry = full[int(sim_seed % n_patches)]
    tid = int(entry["tract"])
    patch_db = int(entry["patch"])
    patch_x = patch_db // 100
    patch_y = patch_db % 100
    pid = patch_x + patch_y * 9
    noiseCorrImage = afwImage.ImageF.readFits(
        f"{pscratch}/deepCoadd_noisecorr/{tid}/{pid}/{band_use}/noise.fits"
    )
    psfImage = afwImage.ImageF.readFits(
        f"{pscratch}/deepCoadd_psf/{tid}/{pid}/{band_use}/psf.fits"
    )

    truth_catalog = cat_task.run(
        tract_info=skymap[tract_id],
        seed=sim_seed,
    ).truthCatalog
    truthfname = os.path.join(
        outdir, "truth-%05d.fits" % (sim_seed)
    )
    if (band == "a") and (not os.path.isfile(truthfname)):
        fitsio.write(truthfname, truth_catalog)

    exposure = sim_task.run(
        tract_info=skymap[tract_id],
        patch_id=patch_id,
        band=band_use,
        seed=sim_seed,
        truthCatalog=truth_catalog,
        psfImage=psfImage,
        noiseCorrImage=noiseCorrImage,
    ).simExposure
    prep = det_task.anacal.prepare_data(
        exposure=exposure,
        seed=100000 + sim_seed,
        detection=detection,
        band=band_use,
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
        noise_corr=noiseCorrImage.getArray(),
    )
    res = det_task.run_measure(prep)
    if band != "a":
        res = rfn.repack_fields(res[colnames])
        map_dict = {name: f"{band}_" + name for name in colnames}
        res = rfn.rename_fields(res, map_dict)
    else:
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
    del prep, exposure, truth_catalog, res
    gc.collect()

# Ensure all ranks finish (no-op in single process)
COMM.Barrier()
if RANK == 0:
    print("Done.")
