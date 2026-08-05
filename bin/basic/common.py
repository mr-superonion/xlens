#!/usr/bin/env python3
"""Shared configuration for sim.py and process.py:

- SkyMap constants and constructor (single-tract DiscreteSkyMap).
- Root output directory and per-config outdir builder.
- Filename conventions for truth, per-band exposure, and matched
  catalog products.

Keeping these in one place guarantees sim.py and process.py always
agree on where files live and how they are named.
"""
import os

from lsst.skymap.discreteSkyMap import DiscreteSkyMap, DiscreteSkyMapConfig

# ------------------------------
# SkyMap constants
# ------------------------------
TRACT_ID = 0
PATCH_ID = 0
PIXEL_SCALE = 0.2  # arcsec/pixel

# ------------------------------
# Input galaxy catalog
# ------------------------------
# Shared by the truth-catalog task and the image simulation task: the
# truth catalog carries the input-catalog property columns, so the two
# must always agree.
GALAXY_TYPE = "flagship2025"

# ------------------------------
# Output directory
# ------------------------------
OUTPUT_ROOT = (
    "/gpfs/mnt/gpfs02/astro/astro_desc/data/simulation/image_simulation/"
    "basic_shear_test"
)


def build_skymap():
    """Construct the single-tract DiscreteSkyMap shared by sim/process."""
    config = DiscreteSkyMapConfig()
    config.raList = [0.0]
    config.decList = [0.0]
    config.radiusList = [0.1]
    config.rotation = 0.0
    config.projection = "TAN"
    config.patchInnerDimensions = [1500, 1500]
    config.patchBorder = 0
    config.pixelScale = PIXEL_SCALE
    config.tractOverlap = 0.0
    return DiscreteSkyMap(config)


def _outdir(layout, test_target, shear_value, shear_mode, kind, version=None):
    """Internal: build/create the output directory for `kind` in
    {'sim', 'process'}. sim.py writes truth + exposures to the 'sim'
    directory; process.py reads from 'sim' and writes catalogs to
    'process'. When `version` is set, `-v<version>` is appended to
    the leaf directory name so parallel processing runs can coexist.
    """
    subdir = f"{kind}_mode{shear_mode}"
    if version is not None:
        subdir = f"{subdir}-v{int(version)}"
    outdir = os.path.join(
        OUTPUT_ROOT,
        f"constant_shear_{layout}",
        test_target,
        f"shear{int(shear_value * 100):02d}",
        subdir,
    )
    os.makedirs(outdir, exist_ok=True)
    return outdir


def sim_outdir(layout, test_target, shear_value, shear_mode):
    """Directory that holds sim.py products (truth + per-band exposures)."""
    return _outdir(layout, test_target, shear_value, shear_mode, "sim")


def process_outdir(layout, test_target, shear_value, shear_mode, version=None):
    """Directory that holds process.py products (matched catalogs).

    `version` is an optional integer tag; when set, outputs go to
    ``process_mode<N>-v<version>/`` instead of ``process_mode<N>/``.
    """
    return _outdir(
        layout, test_target, shear_value, shear_mode, "process", version=version,
    )


# ------------------------------
# Filename conventions
# ------------------------------
def truth_path(outdir, seed):
    """Truth catalog for a given seed."""
    return os.path.join(outdir, "truth-%05d.fits" % seed)


def exp_path(outdir, band, seed):
    """Simulated exposure for a given (band, seed)."""
    return os.path.join(outdir, "exp-%s-%05d.fits" % (band, seed))


def cat_path(outdir, seed):
    """Final matched catalog for a given seed."""
    return os.path.join(outdir, "cat-%05d.fits" % seed)
