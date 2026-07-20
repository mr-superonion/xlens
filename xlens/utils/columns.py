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

"""Detection-catalog column policy.

The detection step (``anacal.run`` with ``detection=None``) emits 96
intermediate columns — raw moments, j1/j2 dilation derivatives,
gauss-aperture fluxes, etc. — most of which are unused by any
downstream code and merely duplicate columns produced by the per-band
forced measurement.

:data:`DETECTION_KEEP_COLUMNS` is the positive whitelist of columns
that are actually consumed by downstream code (set_isPrimary,
matchPipe, cluster analysis, shear-recovery utilities, tests).  Use
:func:`select_detection_columns` to project a detection catalog onto
exactly these columns before merging it with forced measurement.
"""

from __future__ import annotations

import numpy as np
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

DETECTION_KEEP_COLUMNS: tuple[str, ...] = (
    # Sky / pixel positions used by set_isPrimary and matchPipe.
    "ra",
    "dec",
    "x1",
    "x2",
    "x1_det",
    "x2_det",
    # Selection weight + shear derivatives.
    "wsel",
    "dwsel_dg1",
    "dwsel_dg2",
    # FPFS ellipticity + shear derivatives (used by shear estimator).
    "fpfs_e1",
    "fpfs_de1_dg1",
    "fpfs_de1_dg2",
    "fpfs_e2",
    "fpfs_de2_dg1",
    "fpfs_de2_dg2",
    # FPFS shapelet modes used by shear estimator.
    "fpfs_m0",
    "fpfs_dm0_dg1",
    "fpfs_dm0_dg2",
    "fpfs_m2",
    "fpfs_dm2_dg1",
    "fpfs_dm2_dg2",
    # Flags.
    "mask_value",
    "is_peak",
    "is_primary",
)


def select_detection_columns(catalog: NDArray) -> NDArray:
    """Project ``catalog`` onto :data:`DETECTION_KEEP_COLUMNS`.

    Columns listed in the keep-list but missing from the catalog are
    silently skipped.  Returns a contiguous structured array.
    """
    if catalog is None or catalog.dtype.names is None:
        return catalog
    available = set(catalog.dtype.names)
    keep = [c for c in DETECTION_KEEP_COLUMNS if c in available]
    if not keep:
        return catalog
    return np.asarray(rfn.repack_fields(catalog[keep]))


GAUSS_APERTURE_COLUMNS: tuple[str, ...] = (
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
)


def select_band_gauss_fluxes(
    catalog: NDArray,
    band: str,
) -> NDArray | None:
    """Project an anacal forced-measurement output onto the per-band
    Gaussian aperture flux columns and prefix them with ``{band}_``.

    Unlike the FPFS task, the anacal C++ task does not honour the
    ``base_column_name`` data dict entry for the gauss flux columns —
    it always emits them as plain ``flux_gauss{0,2,4}`` (and their
    ``dflux_*_dg1/2`` and ``*_err`` siblings).  So we look up the
    unprefixed names here and rename to ``{band}_flux_gauss{0,2,4}``
    on the way out.

    Returns ``None`` if no gauss columns are present.
    """
    if catalog is None or catalog.dtype.names is None:
        return None
    names = set(catalog.dtype.names)
    keep = [c for c in GAUSS_APERTURE_COLUMNS if c in names]
    if not keep:
        return None
    projected = np.asarray(rfn.repack_fields(catalog[keep]))
    mapping = {c: f"{band}_{c}" for c in keep}
    return np.asarray(rfn.rename_fields(projected, mapping))
