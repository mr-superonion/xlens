#!/usr/bin/env python3

"""
Run a constant shear simulation using MPI.

Steps:
1. Set up a DC2-like SkyMap
2. Simulate multiband shear exposures
3. Run detection and shape measurement
4. Estimate shear and aggregate results across MPI ranks
"""

import os
import argparse

import numpy as np
import matplotlib.pylab as plt
import fitsio

from mpi4py import MPI

import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs
from lsst.skymap.discreteSkyMap import (
    DiscreteSkyMapConfig, DiscreteSkyMap
)

from lsst.pipe.tasks.coaddBase import makeSkyInfo

from xlens.simulator.multiband import (
    MultibandSimShearTaskConfig, MultibandSimShearTask
)
from xlens.process_pipe.anacal_detect import (
    AnacalDetectPipeConfig, AnacalDetectPipe
)

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(
    description="Run constant shear simulation with MPI",
)
parser.add_argument(
    "--root", type=str, default="./", help="root dir"
)
parser.add_argument(
    "--mode", type=int, default=0, choices=[0, 1, 2],
    help="Shear mode: 0: g=-shear, 1: g=shear, 2: g=0.00",
)
parser.add_argument(
    "--rot", type=int, default=0, choices=[0, 1],
    help="rotation id",
)
parser.add_argument(
    "--shear", type=float, default=0.02, help="Shear value"
)
parser.add_argument(
    "--kappa", type=float, default=0.02, help="Kappa value"
)

args = parser.parse_args()

shear_mode = args.mode
shear_value = args.shear
kappa_value = args.kappa
root_dir = args.root
rot_id = args.rot

# ------------------------------
# MPI Setup
# ------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
samples_per_rank = 50

# ------------------------------
# SkyMap Setup
# ------------------------------
pixel_scale = 0.2  # arcsec/pixel
config = DiscreteSkyMapConfig()
config.raList = [0.0]
config.decList = [0.0]
config.radiusList = [6.4 / 9]
config.rotation = 0.0
config.projection = "TAN"
config.patchInnerDimensions = [4000, 4000]
config.patchBorder = 100
config.pixelScale = pixel_scale
config.tractOverlap = 0.0
skymap = DiscreteSkyMap(config)
if rank == 0:
    print("SkyMap created.")

# ------------------------------
# Image Simulation Task
# ------------------------------
sim_config = MultibandSimShearTaskConfig()
sim_config.survey_name = "lsst"
sim_config.draw_image_noise = True
sim_config.z_bounds = [-0.01, 20.0]
sim_config.mode = shear_mode
sim_config.rotId = rot_id
sim_config.kappa_value = kappa_value
sim_config.test_value = shear_value
sim_config.test_target = "g1"
sim_config.layout = "grid"

tract_id = 0
patch_id = 24
bbox = makeSkyInfo(skymap, tractId=tract_id, patchId=patch_id).bbox
wcs = skymap[tract_id][patch_id].getWcs()
sim_task = MultibandSimShearTask(config=sim_config)

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

# ------------------------------
# Buffers for MPI communication
# ------------------------------
e1s = np.empty(samples_per_rank)
e2s = np.empty(samples_per_rank)
R1s = np.empty(samples_per_rank)
R2s = np.empty(samples_per_rank)
Ns  = np.empty(samples_per_rank)

# ------------------------------
# Run Simulation & Measurement
# ------------------------------
for i in range(samples_per_rank):
    sim_seed = rank * samples_per_rank + i
    sim_result = sim_task.run(
        band="i", seed=sim_seed, boundaryBox=bbox, wcs=wcs
    )
    prep = det_task.anacal.prepare_data(
        exposure=sim_result.outputExposure,
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

    e1s[i] = np.sum(e1)
    e2s[i] = np.sum(e2)
    R1s[i] = np.sum(de1_dg1)
    R2s[i] = np.sum(de2_dg2)
    Ns[i] = len(catalog)

def gather_array(local):
    if rank == 0:
        buf = np.zeros((size, samples_per_rank), dtype=float)
    else:
        buf = None
    comm.Gather(local, buf, root=0)
    return buf

if rank == 0:
    print("Gathering results...")

e1_all = np.array(gather_array(e1s))
e2_all = np.array(gather_array(e2s))
R1_all = np.array(gather_array(R1s))
R2_all = np.array(gather_array(R2s))
N_all  = np.array(gather_array(Ns))

# ------------------------------
# Save Results
# ------------------------------
if rank == 0:
    outdir = os.path.join(
        os.environ["SCRATCH"],
        "isolated",
        root_dir,
        f"anacal_blends_shear{int(shear_value * 100):02d}"
    )
    os.makedirs(outdir, exist_ok=True)
    pp = f"_mode{shear_mode}_rot{rot_id}"
    np.save(os.path.join(outdir, f"e1s{pp}.npy"), e1_all)
    np.save(os.path.join(outdir, f"e2s{pp}.npy"), e2_all)
    np.save(os.path.join(outdir, f"R1s{pp}.npy"), R1_all)
    np.save(os.path.join(outdir, f"R2s{pp}.npy"), R2_all)
    np.save(os.path.join(outdir, f"Ns{pp}.npy"), N_all)
    print(f"Saved results to {outdir}")
