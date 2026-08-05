#!/usr/bin/env python3
"""Shared configuration for the cluster-lensing test scripts
(prepare_sim.py, sim.py, process.py, summary.py):

- SkyMap: a single-tract DiscreteSkyMap centred at (ra, dec) = (0, 0)
  with 3x3 = 9 patches and a 100-pixel border so neighbouring patch
  outer boxes overlap (detections are deduplicated downstream with the
  ``is_primary`` flag, which keeps only inner-patch sources).
- The NFW halo parameters of the fiducial cluster (placed at the tract
  centre, i.e. at ra=0, dec=0).
- Root output directory and per-config outdir builders.
- Filename conventions for truth, per-(band, patch) exposures, and
  per-patch catalogs.
- The job-index convention: one job index = one (realization, patch)
  pair, ``sim_seed, patch_id = divmod(index, N_PATCH)``, so each job
  processes exactly one patch.

Keeping these in one place guarantees all four scripts always agree on
the geometry, where files live, and how they are named.
"""
import os

from lsst.skymap.discreteSkyMap import DiscreteSkyMap, DiscreteSkyMapConfig

# ------------------------------
# SkyMap constants
# ------------------------------
TRACT_ID = 0
PIXEL_SCALE = 0.2  # arcsec/pixel
PATCH_INNER_DIM = 1500  # pixels (inner region, per side)
PATCH_BORDER = 100  # pixels; neighbouring patches overlap by this border
N_PATCH_SIDE = 3
N_PATCH = N_PATCH_SIDE * N_PATCH_SIDE  # 9 patches per tract
RA_CENTER = 0.0  # deg; tract (and halo) centre
DEC_CENTER = 0.0  # deg

# ------------------------------
# Input galaxy catalog
# ------------------------------
# Shared by prepare_sim.py and sim.py: the truth catalog carries the
# input-catalog property columns, so the two must always agree.
GALAXY_TYPE = "flagship2025"

# ------------------------------
# Fiducial NFW halo (at the tract centre)
# ------------------------------
HALO_MASS = 4e14  # M_sun
HALO_CONC = 3.8
Z_LENS = 0.25
Z_SOURCE = 1.0  # all source galaxies at this redshift

# ------------------------------
# Output directory (override with XLENS_CLUSTER_TEST_DIR for local tests)
# ------------------------------
OUTPUT_ROOT = os.environ.get(
    "XLENS_CLUSTER_TEST_DIR",
    "/gpfs/mnt/gpfs02/astro/astro_desc/data/simulation/image_simulation/"
    "cluster_shear_test",
)


def build_skymap():
    """Construct the single-tract 3x3-patch DiscreteSkyMap shared by all
    the cluster-test scripts.  The tract inner region is
    ``N_PATCH_SIDE * PATCH_INNER_DIM`` pixels on a side, centred at
    (RA_CENTER, DEC_CENTER)."""
    # Half-size of the tract inner region in degrees. The tract bbox
    # spans 2*radius + 1 pixels, so back off one pixel from the exact
    # half-size or the patch grid rounds up to 4x4.
    radius_deg = (
        (N_PATCH_SIDE * PATCH_INNER_DIM / 2.0 - 1.0) * PIXEL_SCALE / 3600.0
    )
    config = DiscreteSkyMapConfig()
    config.raList = [RA_CENTER]
    config.decList = [DEC_CENTER]
    config.radiusList = [radius_deg]
    config.rotation = 0.0
    config.projection = "TAN"
    config.patchInnerDimensions = [PATCH_INNER_DIM, PATCH_INNER_DIM]
    config.patchBorder = PATCH_BORDER
    config.pixelScale = PIXEL_SCALE
    config.tractOverlap = 0.0
    skymap = DiscreteSkyMap(config)
    num = skymap[TRACT_ID].getNumPatches()
    if (num.x, num.y) != (N_PATCH_SIDE, N_PATCH_SIDE):
        raise RuntimeError(
            f"skymap has {num.x}x{num.y} patches, expected "
            f"{N_PATCH_SIDE}x{N_PATCH_SIDE}; adjust radius_deg"
        )
    return skymap


def seed_patch_from_index(index):
    """Map a global job index to (sim_seed, patch_id).

    One index = one patch of one realization, so submitting indices
    [0, 9*n_realizations) covers every patch of every realization with
    one patch per job.
    """
    return divmod(int(index), N_PATCH)


def halo_tag():
    """Directory tag encoding the fiducial halo parameters."""
    return (
        f"m{HALO_MASS:.1e}_c{HALO_CONC:g}"
        f"_zl{Z_LENS:g}_zs{Z_SOURCE:g}"
    )


def _outdir(layout, rot, kind, version=None):
    """Internal: build/create the output directory for `kind` in
    {'sim_truth', 'sim', 'process'}. prepare_sim.py writes truth
    catalogs to 'sim_truth_rot<r>'; sim.py reads them and writes
    exposures to 'sim_rot<r>'; process.py reads from 'sim_rot<r>'
    and writes catalogs to 'process_rot<r>'. When `version` is set,
    `-v<version>` is appended to the leaf directory name so parallel
    processing runs can coexist (same convention as bin/basic's
    sim_mode<N>/process_mode<N>).
    """
    subdir = f"{kind}_rot{int(rot)}"
    if version is not None:
        subdir = f"{subdir}-v{int(version)}"
    outdir = os.path.join(
        OUTPUT_ROOT,
        f"cluster_{layout}",
        halo_tag(),
        subdir,
    )
    os.makedirs(outdir, exist_ok=True)
    return outdir


def truth_outdir(layout, rot):
    """Directory that holds the truth catalogs (prepare_sim.py products)."""
    return _outdir(layout, rot, "sim_truth")


def sim_outdir(layout, rot):
    """Directory that holds the per-(band, patch) exposures (sim.py
    products)."""
    return _outdir(layout, rot, "sim")


def process_outdir(layout, rot, version=None):
    """Directory that holds process.py products (per-patch catalogs).

    `version` is an optional integer tag; when set, outputs go to
    ``process_rot<r>-v<version>/`` instead of ``process_rot<r>/``.
    """
    return _outdir(layout, rot, "process", version=version)


def summary_outdir(layout, version=None):
    """Directory that holds summary.py partial files (shared by rot0/rot1)."""
    leaf = "summary"
    if version is not None:
        leaf = f"{leaf}-v{int(version)}"
    outdir = os.path.join(
        OUTPUT_ROOT,
        f"cluster_{layout}",
        halo_tag(),
        leaf,
    )
    os.makedirs(outdir, exist_ok=True)
    return outdir


# ------------------------------
# Filename conventions
# ------------------------------
def truth_path(outdir, seed):
    """Truth catalog for a given realization (covers the whole tract)."""
    return os.path.join(outdir, "truth-%05d.fits" % seed)


def exp_path(outdir, band, seed, patch):
    """Simulated exposure for a given (band, realization, patch)."""
    return os.path.join(outdir, "exp-%s-%05d-p%d.fits" % (band, seed, patch))


def cat_path(outdir, seed, patch):
    """Measured catalog for a given (realization, patch)."""
    return os.path.join(outdir, "cat-%05d-p%d.fits" % (seed, patch))
