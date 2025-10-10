#!/usr/bin/env python3
import argparse
import gc
import os

import fitsio
import numpy as np
from lsst.skymap.discreteSkyMap import DiscreteSkyMap, DiscreteSkyMapConfig
from mpi4py import MPI

from xlens.process_pipe.anacal_detect import (
    AnacalDetectPipe,
    AnacalDetectPipeConfig,
)
from xlens.simulator.catalog import (
    CatalogShearTask,
    CatalogShearTaskConfig,
)
from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask
from numpy.lib import recfunctions as rfn

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Run constant shear simulation with MPI"
)
parser.add_argument("--target", type=str, default="g1", help="test target")
parser.add_argument(
    "--mode", type=int, default=0, choices=[40, 0, 27, 9, 3, 1, 36, 4, 80],
    help="40:++++;0:----;27:+---;9:-+--;3:--+-;1:---+;36:++--;4:--++;80:0000")
parser.add_argument(
    "--rot", type=int, default=0, choices=[0, 1], help="rotation id"
)
parser.add_argument("--start", type=int, default=0, help="start id")
parser.add_argument("--end", type=int, default=2, help="end id")
parser.add_argument("--shear", type=float, default=0.02, help="Shear value")
parser.add_argument("--kappa", type=float, default=0.00, help="Kappa value")
parser.add_argument("--layout", type=str, default="grid", help="layout")
parser.add_argument("--band", type=str, default=None, help="band")
args = parser.parse_args()

shear_mode = int(args.mode)
shear_value = args.shear
kappa_value = args.kappa
rot_id = args.rot
test_target = args.target
istart = args.start
iend = args.end
band = args.band

if args.layout == "random":
    extend_ratio = 1.08
elif args.layout == "grid":
    extend_ratio = 0.92
else:
    raise ValueError("Cannot support layout")

# ------------------------------
# MPI Setup
# ------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
cfg_sim.truncate_stamp_size = 65
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
if rank == 0:
    print("Detection task setup complete.")

samples_per_rank = iend - istart
if samples_per_rank <= 0:
    raise ValueError(f"Invalid range: start={istart}, end={iend}")

# Outdir layout:
#   $PSCRATCH/constant_shear_isolated/<target>/anacal_blends_shearXX/
pscratch = os.environ.get("PSCRATCH", ".")
outdir = os.path.join(
    pscratch,
    f"constant_shear_{args.layout}",
    test_target,
    f"shear{int(shear_value * 100):02d}",
)
os.makedirs(outdir, exist_ok=True)

# ------------------------------
# Run Simulation & Measurement
# ------------------------------
colnames = [
    "flux_gauss0",
    "dflux_gauss0_dg1",
    "dflux_gauss0_dg2",
    "flux_gauss2",
    "dflux_gauss2_dg1",
    "dflux_gauss2_dg2",
    "flux_gauss4",
    "dflux_gauss4_dg1",
    "dflux_gauss4_dg2",
]
for i in range(istart, iend):
    sim_seed = i * size + rank
    if band is not None:
        outfname = os.path.join(
            outdir, "cat-%05d-%s-mode%d.fits" % (sim_seed, band, shear_mode)
        )
        detfname = os.path.join(
            outdir, "cat-%05d-mode%d.fits" % (sim_seed, shear_mode)
        )
        if os.path.isfile(detfname):
            detection = fitsio.read(detfname)
        else:
            raise ValueError("Run detection with band=None first")
    else:
        outfname = os.path.join(
            outdir, "cat-%05d-mode%d.fits" % (sim_seed, shear_mode)
        )
        detection = None
    if os.path.isfile(outfname) or (sim_seed>=30000):
        continue
    truth_catalog = cat_task.run(
        tract_info=skymap[tract_id],
        seed=sim_seed,
    ).truthCatalog

    exposure = sim_task.run(
        tract_info=skymap[tract_id],
        patch_id=patch_id,
        band=("i" if band is None else band),
        seed=sim_seed,
        truthCatalog=truth_catalog,
    ).simExposure
    prep = det_task.anacal.prepare_data(
        exposure=exposure,
        seed=100000 + sim_seed,
        noise_corr=None,
        detection=detection,
        band=band,
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
    )
    res = det_task.run_measure(prep)
    if band is not None:
        map_dict = {name: f"{band}_" + name for name in colnames}
        res = rfn.repack_fields(res[colnames])
        res = rfn.rename_fields(res, map_dict)
    fitsio.write(
        outfname,
        res,
    )
    # clean up
    del prep, exposure, truth_catalog, res
    gc.collect()
