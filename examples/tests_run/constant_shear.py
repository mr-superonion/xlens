#!/usr/bin/env python3
import os
import argparse
import gc
import json

import numpy as np
from mpi4py import MPI

from lsst.skymap.discreteSkyMap import (
    DiscreteSkyMapConfig, DiscreteSkyMap
)
from xlens.simulator.catalog import (
    CatalogShearTask,
    CatalogShearTaskConfig,
)
from xlens.simulator.sim import (
    MultibandSimConfig, MultibandSimTask
)
from xlens.process_pipe.anacal_detect import (
    AnacalDetectPipeConfig, AnacalDetectPipe
)

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Run constant shear simulation with MPI"
)
parser.add_argument("--target", type=str, default="g1", help="test target")
parser.add_argument("--mode", type=int, default=0, choices=[0, 1, 2],
                    help="Shear mode: 0: g=-shear, 1: g=shear, 2: g=0.00")
parser.add_argument(
    "--rot", type=int, default=0, choices=[0, 1], help="rotation id"
)
parser.add_argument("--start", type=int, default=0, help="start id")
parser.add_argument("--end", type=int, default=2, help="end id")
parser.add_argument("--shear", type=float, default=0.02, help="Shear value")
parser.add_argument("--kappa", type=float, default=0.02, help="Kappa value")
parser.add_argument("--layout", type=str, default="grid", help="layout")
args = parser.parse_args()

shear_mode = args.mode
shear_value = args.shear
kappa_value = args.kappa
rot_id = args.rot
test_target = args.target
istart = args.start
iend = args.end

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
cfg_cat.z_bounds = [-0.01, 20.0]
cfg_cat.mode = shear_mode
cfg_cat.rotId = rot_id
cfg_cat.kappa_value = kappa_value
cfg_cat.test_value = shear_value
cfg_cat.test_target = test_target
cfg_cat.layout = args.layout
cat_task = CatalogShearTask(config=cfg_cat)

cfg_sim = MultibandSimConfig()
cfg_sim.survey_name = "lsst"
cfg_sim.draw_image_noise = True
sim_task = MultibandSimTask(config=cfg_sim)

# ------------------------------
# Detection Task
# ------------------------------
detect_config = AnacalDetectPipeConfig()
detect_config.anacal.force_size = False
detect_config.anacal.num_epochs = 6
detect_config.anacal.do_noise_bias_correction = True
detect_config.anacal.validate_psf = False
det_task = AnacalDetectPipe(config=detect_config)
if rank == 0:
    print("Detection task setup complete.")

samples_per_rank = iend - istart
if samples_per_rank <= 0:
    raise ValueError(f"Invalid range: start={istart}, end={iend}")

# ------------------------------
# Local Buffers (per-rank)
# ------------------------------
e1s = np.zeros(samples_per_rank, dtype=np.float64)
e2s = np.zeros(samples_per_rank, dtype=np.float64)
R1s = np.zeros(samples_per_rank, dtype=np.float64)
R2s = np.zeros(samples_per_rank, dtype=np.float64)
Ns  = np.zeros(samples_per_rank, dtype=np.float64)

# ------------------------------
# Run Simulation & Measurement
# ------------------------------
for i in range(istart, iend):
    j = i - istart
    sim_seed = i * size + rank

    truth_catalog = cat_task.run(
        tract_info=skymap[tract_id],
        seed=sim_seed,
    ).truthCatalog

    sim_result = sim_task.run(
        tract_info=skymap[tract_id],
        patch_id=patch_id,
        band="i",
        seed=sim_seed,
        truthCatalog=truth_catalog,
    )

    prep = det_task.anacal.prepare_data(
        exposure=sim_result.simExposure,
        seed=100000 + sim_seed,
        noise_corr=None,
        detection=None,
        band=None,
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
    )
    catalog = det_task.anacal.run(**prep)

    # Shear Estimation
    e1 = catalog["wsel"] * catalog["fpfs_e1"]
    de1_dg1 = (
        catalog["dwsel_dg1"] * catalog["fpfs_e1"] +
        catalog["wsel"] * catalog["fpfs_de1_dg1"]
    )

    e2 = catalog["wsel"] * catalog["fpfs_e2"]
    de2_dg2 = (
        catalog["dwsel_dg2"] * catalog["fpfs_e2"] +
        catalog["wsel"] * catalog["fpfs_de2_dg2"]
    )

    e1s[j] = np.sum(e1)
    e2s[j] = np.sum(e2)
    R1s[j] = np.sum(de1_dg1)
    R2s[j] = np.sum(de2_dg2)
    Ns[j] = float(len(catalog))

    # clean up
    del prep, sim_result, truth_catalog, catalog
    gc.collect()

# ------------------------------
# Save Results (per-rank, no Gather)
# ------------------------------
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

pp = f"_mode{shear_mode}_rot{rot_id}_rank{rank:05d}"
np.save(os.path.join(outdir, f"e1s{pp}.npy"), e1s)
np.save(os.path.join(outdir, f"e2s{pp}.npy"), e2s)
np.save(os.path.join(outdir, f"R1s{pp}.npy"), R1s)
np.save(os.path.join(outdir, f"R2s{pp}.npy"), R2s)
np.save(os.path.join(outdir, f"Ns{pp}.npy"), Ns)
print(f"[rank {rank}] wrote outputs to {outdir}")
