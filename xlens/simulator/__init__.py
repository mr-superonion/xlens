"""Simulation subpackage for generating synthetic LSST coadd images.

Provides galaxy catalog construction, image rendering with GalSim,
lensing perturbation models, layout generators, WCS utilities, and
noise injection tasks compatible with the Rubin Science Pipelines.
"""

from . import (
    bat,
    catalog,
    defaults,
    galaxies,
    layout,
    mog,
    noise,
    perturbation,
    sim,
)

__all__ = [
    "sim",
    "perturbation",
    "defaults",
    "galaxies",
    "layout",
    "catalog",
    "noise",
    "bat",
    "mog",
]
