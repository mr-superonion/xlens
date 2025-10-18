#!/usr/bin/env python3
import argparse
import gc
import os

import fitsio
import numpy as np
from lsst.skymap.discreteSkyMap import DiscreteSkyMap, DiscreteSkyMapConfig

# --- Optional MPI: works without mpirun/srun or even mpi4py installed ---
try:
    from mpi4py import MPI  # type: ignore
    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()

    def _barrier():
        _COMM.Barrier()
except Exception:
    # Fallback to a tiny shim so the script runs single-process
    class _FakeComm:

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1
    _COMM = _FakeComm()
    _RANK = 0
    _SIZE = 1

    def _barrier():  # no-op
        return

from numpy.lib import recfunctions as rfn

from xlens.process_pipe.anacal_detect import (
    AnacalDetectPipe,
    AnacalDetectPipeConfig,
)
from xlens.simulator.catalog import (
    CatalogShearTask,
    CatalogShearTaskConfig,
)
from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask
from xlens.process_pipe.match import (
    matchPipe,
    matchPipeConfig,
)
from xlens.utils.image import (
    combine_sim_exposures,
    estimate_noise_variance,
    generate_pure_noise,
)

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
    help="single band (g,r,i,z,y) or None for multiband",
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
if band not in "grizy":
    band = None
if args.layout == "random":
    extend_ratio = 1.08
elif args.layout == "grid":
    extend_ratio = 0.92
else:
    raise ValueError("Cannot support layout")

# ------------------------------
# MPI (or single-process) info
# ------------------------------
comm = _COMM
rank = _RANK
size = _SIZE
if rank == 0:
    if size == 1:
        print("[Info] Running single-process (no mpirun/srun needed).")
    else:
        print(f"[Info] Running with MPI across {size} ranks.")

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
config.patchInnerDimensions = [4000, 4000]
config.patchBorder = 0
config.pixelScale = pixel_scale
config.tractOverlap = 0.0
skymap = DiscreteSkyMap(config)
if rank == 0:
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
# cfg_sim.truncate_stamp_size = 65
sim_task = MultibandSimTask(config=cfg_sim)

# ------------------------------
# Detection Task
# ------------------------------
detect_config = AnacalDetectPipeConfig()
detect_config.anacal.sigma_arcsec = 0.38
detect_config.anacal.force_size = True
detect_config.anacal.num_epochs = 0
detect_config.anacal.do_noise_bias_correction = True
detect_config.do_fpfs = (band is None)
detect_config.fpfs.sigma_shapelets1 = 0.38 * np.sqrt(2.0)
det_task = AnacalDetectPipe(config=detect_config)

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


def get_exposure(truth_catalog, sim_seed, band=None):
    if band is None:
        explist = []
        noiselist = []
        for bb in ["g", "r", "i", "z"]:
            exp = sim_task.run(
                tract_info=skymap[tract_id],
                patch_id=patch_id,
                band=bb,
                seed=sim_seed,
                truthCatalog=truth_catalog,
            ).simExposure
            nx = exp.getWidth()
            ny = exp.getHeight()
            noise_variance = estimate_noise_variance(
                exposure=exp, mask_array=None,
            )
            explist.append(
                exp
            )
            noise_array = generate_pure_noise(
                ny=ny,
                nx=nx,
                pixel_scale=pixel_scale,
                seed=100000 + sim_seed,
                band=bb,
                noise_variance=noise_variance,
                noise_corr=None,
                noiseId=0,
                rotId=args.rot,
            )
            noiselist.append(noise_array)
        exposure, noise_array = combine_sim_exposures(explist, noiselist)
    else:
        exposure = sim_task.run(
            tract_info=skymap[tract_id],
            patch_id=patch_id,
            band=band,
            seed=sim_seed,
            truthCatalog=truth_catalog,
        ).simExposure
        noise_array = None
    return exposure, noise_array


# ------------------------------
# Work loop (unique seeds per rank if MPI)
# ------------------------------
for i in range(istart, iend):
    sim_seed = i * size + rank
    if band is not None:
        outfname = os.path.join(
            outdir, "cat-%05d-%s.fits" % (sim_seed, band)
        )
        detfname = os.path.join(
            outdir, "cat-%05d.fits" % (sim_seed)
        )
        if os.path.isfile(detfname):
            detection = fitsio.read(detfname)
        else:
            raise ValueError("Run detection with band=None first")
    else:
        outfname = os.path.join(
            outdir, "cat-%05d.fits" % (sim_seed)
        )
        detection = None

    if os.path.isfile(outfname) or (sim_seed >= 30000):
        continue

    truth_catalog = cat_task.run(
        tract_info=skymap[tract_id],
        seed=sim_seed,
    ).truthCatalog
    truthfname = os.path.join(
        outdir, "truth-%05d.fits" % (sim_seed)
    )
    if (band is None) and (not os.path.isfile(truthfname)):
        fitsio.write(truthfname, truth_catalog)

    exposure, noise_array = get_exposure(
        truth_catalog=truth_catalog, sim_seed=sim_seed, band=band,
    )
    prep = det_task.anacal.prepare_data(
        exposure=exposure,
        seed=100000 + sim_seed,
        noise_corr=None,
        detection=detection,
        band=band,
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
        noise_array=noise_array,
    )
    res = det_task.run_measure(prep)

    if band is not None:
        map_dict = {name: f"{band}_" + name for name in colnames}
        res = rfn.repack_fields(res[colnames])
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
_barrier()
if rank == 0:
    print("Done.")
