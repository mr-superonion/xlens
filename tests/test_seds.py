# This file is part of xlens.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Round-trip unit test for :func:`xlens.simulator.seds.compute_magnitudes`.

Runs the SciPIC-exact recipe on 100 randomly-sampled Flagship 2 galaxies
with ``redshift < 1.5`` and checks that the synthesised AB magnitudes
match the catalog's stored ``_el_model3_ext`` broadband magnitudes to
within tight per-band tolerances calibrated on 10000-galaxy z<1 tests.
"""
from __future__ import annotations

import os

import fitsio
import numpy as np
import pytest

from xlens.simulator.seds import compute_magnitudes

_FITS_PATH = os.environ.get(
    "CATSIM_FITS_PATH",
    "/gpfs/mnt/gpfs02/astro/astro_desc/data/simulation/input_catalog/"
    "catsim_flagship_scripts/flagship_cosmos.fits",
)

BANDS = (
    "euclid_vis",
    "lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y",
    "euclid_nisp_y", "euclid_nisp_j", "euclid_nisp_h",
)

Z_MAX = 1.5
N_SAMPLE = 50
RNG_SEED = 1138

# Uniform per-band tolerance (mag) for both |mean| and std of the
# per-galaxy residuals. All bands achieve well under this on a
# 10000-galaxy z<1 test; this is a loose margin for the 50-galaxy
# stochastic sample.
TOL = 0.01


def _load_sample():
    if not os.path.exists(_FITS_PATH):
        pytest.skip(f"flagship catalog not found at {_FITS_PATH}")
    cat = fitsio.read(_FITS_PATH)
    cat = cat[cat["redshift"] < Z_MAX]
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(cat), N_SAMPLE, replace=False)
    return cat[idx]


def test_compute_magnitudes_roundtrip():
    """Synthesised AB mags match the catalog to <0.02 mag mean and std."""
    sample = _load_sample()

    diffs = {b: [] for b in BANDS}
    for row in sample:
        mags = compute_magnitudes(BANDS, row)
        for band, mag in mags.items():
            if np.isfinite(mag) and np.isfinite(row[band]):
                diffs[band].append(mag - row[band])

    failures = []
    for band in BANDS:
        d = np.asarray(diffs[band])
        assert d.size >= 0.5 * N_SAMPLE, (
            f"{band}: only {d.size} finite residuals from {N_SAMPLE} galaxies"
        )
        mean = float(d.mean())
        std = float(d.std())
        if abs(mean) >= TOL:
            failures.append((band, f"|mean|={abs(mean):.4f} >= {TOL}"))
        if std >= TOL:
            failures.append((band, f"std={std:.4f} >= {TOL}"))
    assert not failures, f"population-stats failures: {failures}"
