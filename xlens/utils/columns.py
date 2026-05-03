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


def rename_flux_to_photoz_format(
    per_band_cat: NDArray, band: str,
) -> NDArray:
    """Rename FPFS Gaussian-flux columns to the schema photoZPipe
    consumes via ``flux_name=<kernel>``.

    For any kernel in ``("fpfs", "fpfs1", "fpfs2")`` present on the
    per-band measurement, swaps the kernel and 'flux' tokens:

      ``{band}_{kernel}_flux``        -> ``{band}_flux_{kernel}``
      ``{band}_{kernel}_dflux_dg1``   -> ``{band}_dflux_{kernel}_dg1``
      ``{band}_{kernel}_dflux_dg2``   -> ``{band}_dflux_{kernel}_dg2``
      ``{band}_{kernel}_flux_err``    -> ``{band}_flux_{kernel}_err``
    """
    if per_band_cat.dtype.names is None:
        return per_band_cat
    names = set(per_band_cat.dtype.names)
    mapping: dict[str, str] = {}
    for kernel in ("fpfs", "fpfs1", "fpfs2"):
        for src, dst in (
            (f"{band}_{kernel}_flux",
             f"{band}_flux_{kernel}"),
            (f"{band}_{kernel}_dflux_dg1",
             f"{band}_dflux_{kernel}_dg1"),
            (f"{band}_{kernel}_dflux_dg2",
             f"{band}_dflux_{kernel}_dg2"),
            (f"{band}_{kernel}_flux_err",
             f"{band}_flux_{kernel}_err"),
        ):
            if src in names:
                mapping[src] = dst
    if not mapping:
        return per_band_cat
    return np.asarray(rfn.rename_fields(per_band_cat, mapping))


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
    "flux_gauss0", "dflux_gauss0_dg1", "dflux_gauss0_dg2", "flux_gauss0_err",
    "flux_gauss2", "dflux_gauss2_dg1", "dflux_gauss2_dg2", "flux_gauss2_err",
    "flux_gauss4", "dflux_gauss4_dg1", "dflux_gauss4_dg2", "flux_gauss4_err",
)


def select_band_gauss_fluxes(
    catalog: NDArray, band: str,
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
