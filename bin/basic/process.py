#!/usr/bin/env python3
"""Process simulated multiband exposures (produced by sim.py) into
matched catalogs. For each seed, reads truth-<seed>.fits and
exp-<band>-<seed>.fits per band, runs detection + forced measurement +
truth match, and writes cat-<seed>.fits. Seeds whose sim outputs are
missing are skipped (with a note), so the two scripts can run
independently and be resumed.

Detection runs on ``--detection-band`` (default 'i'); give it several
bands to detect on their inverse-variance coadd, e.g.

    process.py --band r,i,z --detection-band r,i,z --version 1

Forced measurement is per band over ``--band`` regardless, so the output
columns are the same either way.
"""
import argparse
import gc
import os

import common
import fitsio
import numpy as np
from lsst.afw.image import ExposureF, ImageF
from mpi4py import MPI

from xlens.processor.match import (
    matchPipe,
    matchPipeConfig,
)
from xlens.simulator.noise import AddNoisePipe, AddNoisePipeConfig
from xlens.processor.measure_coadds import (
    MeasureCoaddsPipe,
    MeasureCoaddsPipeConfig,
)
from xlens.utils.handle import make_exposure_handles

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
    "--detection-band", type=str, default="i",
    help="comma-separated bands to coadd for DETECTION (e.g. 'r,i,z'); "
         "must be a subset of --band. AnaCal removes each band's own PSF "
         "before averaging them with inverse-variance weights, so the bands "
         "need not be PSF-matched. Forced measurement still runs band by "
         "band over --band either way. Use --version to keep catalogs from "
         "different detection bands in separate directories.",
)
parser.add_argument(
    "--renoise-mode", type=str, default="rotate",
    choices=["rotate", "spectral"],
    help="Noise-bias correction with the pure-noise image: 'rotate' adds a "
         "90-degree-rotated copy of the noise, 'spectral' adds the minimal-"
         "variance noise of the renoising draft. 'rotate' is the method the "
         "reference catalogs in process_mode<N>/ were made with.",
)
parser.add_argument(
    "--version", type=int, default=None,
    help="Optional integer tag; when set, catalogs go to "
         "process_mode<N>-v<version>/ instead of process_mode<N>/.",
)
parser.add_argument(
    "--sim-tag", type=str, default=None,
    help="Read the exposures from sim_mode<N>_<sim-tag>/ instead of "
         "sim_mode<N>/ (e.g. 'noiseless').",
)
parser.add_argument(
    "--add-noise", action="store_true",
    help="The exposures are noiseless (sim.py --no-image-noise): add the "
         "image noise here, in memory, with AddNoisePipe before measuring. "
         "The realisation is the one sim.py would have drawn for this seed "
         "(and --noise-id), so the catalogs equal those of a noisy "
         "simulation; the noisy images are never written.",
)
parser.add_argument(
    "--noise-corr", type=str, default=None,
    help="FITS noise correlation kernel normalised to 1 at its centre pixel "
         "(e.g. noise_correlation_0.fits). It is used for BOTH sides of the "
         "noise bias correction: with --add-noise the noise is generated with "
         "this correlation, and the measurement receives it so the renoising "
         "(rotate or spectral) is built from the same spectrum. Per band the "
         "kernel is scaled to that exposure's noise variance. Without it the "
         "noise is white.",
)
parser.add_argument(
    "--noise-id", type=int, default=0,
    help="noiseId of the realisation added by --add-noise (sim.py uses 0); "
         "another value is an independent noise realisation on the same "
         "galaxies.",
)
parser.add_argument(
    "--sigma-shapelets1", type=float, default=0.38 * np.sqrt(2.0),
    help="Gaussian kernel of the FPFS shapelet FLUX measurement [arcsec], "
         "the lsst_<b>_flux_fpfs1 columns the magnitude cut uses. The "
         "default 0.38*sqrt(2) = 0.5374 is the legacy value: 0.38 was the "
         "hard-coded sigma_arcsec before d7dad07.",
)
parser.add_argument(
    "--tag", type=str, default=None,
    help="Optional name for the configuration; when set, catalogs go to "
         "process_mode<N>_<tag>/ (e.g. --tag spectral -> "
         "process_mode40_spectral/). Combines with --version.",
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

detection_bands = [
    b.strip() for b in args.detection_band.split(",") if b.strip()
]
if not detection_bands:
    raise ValueError(
        f"Invalid --detection-band argument: {args.detection_band!r}"
    )
# Detection reads the same exposures the forced measurement does, so it can
# only use bands that were loaded.
missing_det = [b for b in detection_bands if b not in bands]
if missing_det:
    raise ValueError(
        f"--detection-band {missing_det} not in --band {bands}; "
        f"add them to --band so the exposures get loaded"
    )

if RANK == 0:
    if SIZE == 1:
        print("[Info] Running single-process (no mpirun/srun needed).")
    else:
        print(f"[Info] Running with MPI across {SIZE} ranks.")
    print(f"[Info] Detecting on {','.join(detection_bands)}; "
          f"forced measurement on {','.join(bands)}.")
    print(f"[Info] Renoising method: {args.renoise_mode}.")

# ------------------------------
# SkyMap Setup
# ------------------------------
skymap = common.build_skymap()
if RANK == 0:
    print("SkyMap created.")

# ------------------------------
# Detection (on detection_bands) + per-band forced measurement (all bands)
# ------------------------------
detect_config = MeasureCoaddsPipeConfig()
detect_config.renoise_mode = args.renoise_mode
detect_config.anacal.force_size = True
detect_config.anacal.num_epochs = 0
detect_config.anacal.do_noise_bias_correction = True
detect_config.fpfs.do_noise_bias_correction = True
detect_config.fpfs.sigma_shapelets1 = args.sigma_shapelets1
detect_config.use_sim = False
detect_config.detection_bands = detection_bands
detect_config.validate()
meas_task = MeasureCoaddsPipe(config=detect_config)

# One kernel for both the noise realisation and its bias correction.
noise_kernel = None
if args.noise_corr:
    noise_kernel = fitsio.read(args.noise_corr).astype(np.float64)
    kny, knx = noise_kernel.shape
    if kny % 2 == 0 or knx % 2 == 0:
        raise ValueError(f"noise correlation must have odd dimensions, got {noise_kernel.shape}")
    peak = float(np.amax(noise_kernel))
    if not np.isclose(noise_kernel[kny // 2, knx // 2], peak) or peak <= 0:
        raise ValueError("noise correlation peak must be positive and at the centre pixel")
    noise_kernel = noise_kernel / peak
    if RANK == 0:
        print(f"[Info] Noise correlation from {args.noise_corr} {noise_kernel.shape}; "
              f"used for the noise realisation and the renoising.")

BAND_ORDER = "ugrizy"


def band_corr_arrays(exposures):
    """(per-band kernel scaled to its variance, 6-band stack) or (None, None).

    The two consumers differ: AddNoisePipe reads the variance off the
    kernel's peak, while the measurement normalises the peak away and takes
    the variance from the exposure.  Scaling by each exposure's own variance
    plane satisfies both and keeps them consistent by construction.
    """
    if noise_kernel is None:
        return None, None
    per_band = {}
    stack = np.zeros((len(BAND_ORDER),) + noise_kernel.shape)
    for bb, exp in exposures.items():
        var = float(np.median(exp.getMaskedImage().variance.array))
        if not np.isfinite(var) or var <= 0:
            raise RuntimeError(f"band {bb}: variance plane is {var}, cannot scale the kernel")
        per_band[bb] = noise_kernel * var
        if bb in BAND_ORDER:
            stack[BAND_ORDER.index(bb)] = per_band[bb]
    return per_band, stack


noise_task = None
if args.add_noise:
    # the same survey / galId / rotId as sim.py, so the seed derivation
    # matches a simulation that drew its own noise
    noise_config = AddNoisePipeConfig()
    noise_config.survey_name = common.SURVEY_NAME_LIST[0]
    noise_config.galId = 0
    noise_config.rotId = args.rot
    noise_config.noiseId = args.noise_id
    noise_config.connections.noiseId = args.noise_id
    noise_config.validate()
    noise_task = AddNoisePipe(config=noise_config)
    if RANK == 0:
        print(f"[Info] Adding image noise at processing time (noiseId={args.noise_id}).")

config = matchPipeConfig()
config.mag_max_truth = common.MAG_MAX_TRUTH
# The magnitude column names of the truth catalog follow from the input
# galaxy catalog it was drawn from and the survey/band being cut on.
config.galaxy_type = common.GALAXY_TYPE
config.survey_name = common.SURVEY_NAME_LIST[0]
config.band = common.MATCH_BAND
config.validate()
match_task = matchPipe(config=config)


# ------------------------------
# Output layout: sim.py writes truth/exposures under sim_mode<N>/;
# process.py reads from there and writes catalogs under process_mode<N>/.
# ------------------------------
sim_dir = common.sim_outdir(
    args.layout, test_target, shear_value, shear_mode, tag=args.sim_tag,
)
proc_dir = common.process_outdir(
    args.layout, test_target, shear_value, shear_mode, version=args.version,
    tag=args.tag,
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
    corr_by_band, corr_array = band_corr_arrays(exposures)
    if noise_task is not None:
        for bb in bands:
            corr_img = None
            if corr_by_band is not None:
                arr = corr_by_band[bb]
                corr_img = ImageF(arr.shape[1], arr.shape[0])
                corr_img.getArray()[:, :] = arr.astype(np.float32)
            exposures[bb] = noise_task.run(
                exposure=exposures[bb], seed=sim_seed, band=bb,
                noiseCorrImage=corr_img,
            ).simExposure

    handles = make_exposure_handles(
        exposures,
        skymap="test",
        tract=common.TRACT_ID,
        patch=common.PATCH_ID,
        skyMap=skymap,
    )
    res = meas_task.run(
        exposure_handles_dict=handles,
        corr_array=corr_array,
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
    fitsio.write(outfname, res)

    # clean up
    del exposures, handles, truth_catalog, res
    gc.collect()

# Ensure all ranks finish (no-op in single process)
COMM.Barrier()
if RANK == 0:
    print("Done.")
